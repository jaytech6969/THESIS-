# backend/server.py
import uvicorn
from fastapi import FastAPI, UploadFile, File
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModelForTokenClassification, pipeline
from PIL import Image
import io

app = FastAPI()

print("⏳ Loading 'The Council' models (This may take a while)...")

# --- LOAD THE EXPERTS (Heavy Models) ---

# 1. TrOCR (The Scribe - Reads Handwriting)
# Note: 'trocr-base-handwritten' is large. Ensure you have internet connection.
print("   ...Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# 2. BioBERT (The Pharmacist - Extracts Medical Entities)
print("   ...Loading BioBERT...")
ner_tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
ner_model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

print("✅ Models Loaded! Server is ready.")

@app.get("/")
def home():
    return {"status": "The Council is online."}

@app.post("/analyze_prescription")
async def analyze(file: UploadFile = File(...)):
    try:
        # 1. Read Image from upload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 2. TrOCR reads the handwriting
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = ocr_model.generate(pixel_values)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 3. BioBERT analyzes the text
        # (We convert numpy types to standard python types for JSON serialization)
        raw_entities = ner_pipeline(extracted_text)
        entities = []
        for entity in raw_entities:
            entities.append({
                "entity_group": entity['entity_group'],
                "word": entity['word'],
                "score": float(entity['score'])
            })
        
        # 4. Return structured data
        return {
            "text_transcription": extracted_text,
            "medical_analysis": entities
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Run the server on localhost port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)