# train_v2.py
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # --- ADDED BatchNormalization ---
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # --- ADDED ReduceLROnPlateau ---
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator # --- ADDED ImageDataGenerator ---

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
        
        # Resize and normalize
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        images.append(img)
        
    images = np.array(images)
    # Reshape to (num_images, height, width, channels)
    images = images.reshape(images.shape[0], target_size[1], target_size[0], 3)
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
    X = load_images(image_paths)
    
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

# --- 3. --- ADDED: Data Augmentation ---
datagen = ImageDataGenerator(
    rotation_range=5,      # small rotations
    width_shift_range=0.1,  # horizontal shift
    height_shift_range=0.1, # vertical shift
    zoom_range=0.1,         # zoom in/out
    fill_mode='nearest'     # fill in new pixels
)
datagen.fit(x_train)

# --- 4. --- UPDATED: Build the CNN Model ---
lambda_reg = 0.00001
model = models.Sequential([
    Input(shape=(32, 128, 3)),  
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(), # --- ADDED ---
    MaxPooling2D(pool_size=(2, 2)), 

    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(lambda_reg)),
    BatchNormalization(), # --- ADDED ---
    MaxPooling2D(pool_size=(2, 2)), 

    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(lambda_reg)),
    BatchNormalization(), # --- ADDED ---
    MaxPooling2D(pool_size=(2, 2)),

    # --- ADDED one more layer to make the model deeper ---
    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(lambda_reg)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),

    Dense(1024, activation='relu', kernel_regularizer=l2(lambda_reg)),
    BatchNormalization(), # --- ADDED ---
    Dropout(0.55), 

    Dense(num_classes, activation='softmax', kernel_regularizer=l2(lambda_reg))  
])

# --- UPDATED optimizer with a better starting learning rate ---
model.compile(
    optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), # --- LR changed to 0.001
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
model.summary()

# --- 5. --- ADDED: New Callbacks ---
# This will stop training if the val_loss doesn't improve for 5 epochs
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, # --- Increased patience
    restore_best_weights=True
)

# This will reduce the learning rate if val_loss stops improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, # Reduce LR by 80%
    patience=2, 
    min_lr=0.00001
)

# --- 6. --- UPDATED: Train the Model ---
print("Starting model training with data augmentation...")
batch_size = 32
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size), # --- USE THE AUGMENTATION GENERATOR ---
    epochs=100, 
    steps_per_epoch=len(x_train) // batch_size, # --- Required when using a generator
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr] # --- ADDED BOTH CALLBACKS ---
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