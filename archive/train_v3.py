# train_v3.py
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
# --- NEW IMPORTS for the CRNN model ---
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Reshape, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1. Data Loading Functions (Unchanged) ---

def one_hot_encoded_Y(df):
    """Converts the 'MEDICINE_NAME' column to one-hot encoding."""
    return pd.get_dummies(df['MEDICINE_NAME'], dtype=int)

def load_images(image_paths, target_size=(128, 32)):
    """Loads, resizes, and normalizes images from a list of paths."""
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image {path}. Skipping.")
            continue
        
        # --- IMPORTANT: Load as Grayscale for CRNN ---
        # CRNN models often work better on grayscale (1 channel)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, target_size)
        
        # Add channel dimension
        img_expanded = np.expand_dims(img_resized, axis=-1)
        
        # Normalize
        img_norm = (img_expanded - 127.5) / 127.5
        images.append(img_norm)
        
    images = np.array(images)
    # Reshape to (num_images, height, width, 1 channel)
    images = images.reshape(images.shape[0], target_size[1], target_size[0], 1)
    return images

def load_dataset(folder_name):
    """Loads X (images) and Y (labels) from a given folder (Training, Validation, or Testing)."""
    lower_name = folder_name.lower()
    
    labels_path = os.path.join(folder_name, f"{lower_name}_labels.csv")
    images_folder = os.path.join(folder_name, f"{lower_name}_words")
    
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found at {labels_path}")
        return None, None
    if not os.path.exists(images_folder):
        print(f"Error: Images folder not found at {images_folder}")
        return None, None
        
    df = pd.read_csv(labels_path)
    Y = one_hot_encoded_Y(df)
    image_paths = [os.path.join(images_folder, filename) for filename in df['IMAGE']]
    X = load_images(image_paths) # This will now return grayscale images
    
    if folder_name == "Training":
        class_names = Y.columns.tolist()
        with open("labels.txt", "w") as f:
            for item in class_names:
                f.write(f"{item}\n")
        print(f"✅ Saved {len(class_names)} class names to labels.txt")

    return X, Y

# --- 2. Load All Datasets (Unchanged) ---
print("Loading Training data...")
x_train, y_train = load_dataset("Training")
print("Loading Validation data...")
x_val, y_val = load_dataset("Validation")
print("Loading Testing data...")
x_test, y_test = load_dataset("Testing")

if x_train is None:
    print("Failed to load training data. Exiting.")
    exit()

num_classes = len(y_train.columns)
print(f"Data loaded. Found {num_classes} classes.")

# --- 3. Data Augmentation ---
# We use a different augmentation for grayscale (1 channel)
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)
datagen.fit(x_train)

# --- 4. --- NEW CRNN Model Architecture ---
lambda_reg = 0.0001 # Slightly increased regularization for the complex model

# Define the CRNN model
input_tensor = Input(shape=(32, 128, 1)) # --- Input shape is now (32, 128, 1) ---

# --- CNN Feature Extractor ---
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
# After this, shape is (None, 4, 16, 256)

# --- Map CNN features to RNN sequence ---
# Get the shape before reshaping
shape = x.shape
# Reshape to (None, width_steps, features)
# (None, 16, 4 * 256) = (None, 16, 1024)
x = Reshape(target_shape=(shape[2], shape[1] * shape[3]))(x)
x = Dense(128, activation='relu')(x)

# --- RNN Sequence Recognizer ---
x = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(lambda_reg)))(x)
x = Bidirectional(LSTM(128, return_sequences=False, kernel_regularizer=l2(lambda_reg)))(x)

# --- Classifier Head ---
x = Dense(512, activation='relu', kernel_regularizer=l2(lambda_reg))(x)
x = Dropout(0.5)(x)
output_tensor = Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=input_tensor, outputs=output_tensor)

model.compile(
    optimizer=Adam(learning_rate=0.001), # Start with a solid LR
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
model.summary()

# --- 5. Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, # --- Increased patience for the more complex model ---
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, # Reduce LR if val_loss doesn't improve for 3 epochs
    min_lr=0.00001
)

# --- 6. Train the Model ---
print("Starting CRNN model training with data augmentation...")
batch_size = 32
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=100, 
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)
print("✅ Training complete.")

# --- 7. Evaluate and Show F1 Score (Unchanged) ---
print("Evaluating model on test data...")
test_metrics = model.evaluate(x_test, y_test)

test_loss = test_metrics[0]
test_acc = test_metrics[1]
test_precision = test_metrics[2]
test_recall = test_metrics[3]

f1_score_val = 0.0
if (test_precision + test_recall) > 0: # Avoid division by zero
    f1_score_val = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print("\n--- Test Results ---")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1 Score:  {f1_score_val:.4f}")

# --- 8. Save the Model (Unchanged) ---
model.save('prescription_model.keras')
print("\n✅ Model saved as 'prescription_model.keras'")

# --- 9. Convert to TFLite for Flutter (Unchanged) ---
print("Converting model to TensorFlow Lite (.tflite)...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open('prescription_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("✅ Model successfully converted and saved as 'prescription_model.tflite'")
except Exception as e:
    print(f"Error during TFLite conversion: {e}")

print("\nAll done!")