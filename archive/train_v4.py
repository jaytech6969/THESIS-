# train_v4.py
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# --- Import augmentation layers ---
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score, precision_score, recall_score

# --- 1. Data Loading Functions (Back to RGB) ---

def one_hot_encoded_Y(df):
    return pd.get_dummies(df['MEDICINE_NAME'], dtype=int)

def load_images(image_paths, target_size=(128, 32)):
    """Loads, resizes, and normalizes images from a list of paths."""
    images = []
    for path in image_paths:
        img = cv2.imread(path) # Read as 3-channel BGR
        if img is None:
            print(f"Warning: Could not read image {path}. Skipping.")
            continue
        
        # Resize and normalize
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0 # Simple 0-1 normalization
        images.append(img)
        
    images = np.array(images)
    # Reshape to (num_images, height, width, 3 channels)
    images = images.reshape(images.shape[0], target_size[1], target_size[0], 3)
    return images

def load_dataset(folder_name):
    lower_name = folder_name.lower()
    labels_path = os.path.join(folder_name, f"{lower_name}_labels.csv")
    images_folder = os.path.join(folder_name, f"{lower_name}_words")
    
    if not os.path.exists(labels_path) or not os.path.exists(images_folder):
        print(f"Error: Data paths not found for {folder_name}")
        return None, None
        
    df = pd.read_csv(labels_path)
    Y = one_hot_encoded_Y(df)
    image_paths = [os.path.join(images_folder, filename) for filename in df['IMAGE']]
    X = load_images(image_paths) # This will now return 3-channel images
    
    if folder_name == "Training":
        class_names = Y.columns.tolist()
        with open("labels.txt", "w") as f:
            for item in class_names:
                f.write(f"{item}\n")
        print(f"✅ Saved {len(class_names)} class names to labels.txt")

    return X, Y

# --- 2. Load All Datasets ---
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

# --- 3. --- NEW: TF.DATA.DATASET ---
# This is the modern, stable way to handle training
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("✅ tf.data.Dataset created.")

# --- 4. --- NEW: Data Augmentation as Layers ---
data_augmentation = models.Sequential([
    RandomRotation(0.05, fill_mode='nearest'),
    RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest'),
    RandomZoom(height_factor=0.1, fill_mode='nearest')
], name='augmentation')

# --- 5. --- NEW: Deeper "VGG-style" CNN Model ---
lambda_reg = 0.00001
input_tensor = Input(shape=(32, 128, 3))

# --- Add augmentation layer to the model ---
x = data_augmentation(input_tensor)

# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x) # Shape: (16, 64, 64)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x) # Shape: (8, 32, 128)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x) # Shape: (4, 16, 256)

# --- Classifier Head ---
x = Flatten()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(lambda_reg))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(lambda_reg))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output_tensor = Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=input_tensor, outputs=output_tensor)

model.compile(
    optimizer=Adam(learning_rate=0.001), # Start with a solid LR
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
model.summary()

# --- 6. Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# --- 7. --- NEW: Train with tf.data.Dataset ---
print("Starting 'VGG-style' model training...")
history = model.fit(
    train_dataset, # --- Use the tf.data.Dataset
    epochs=100, 
    validation_data=val_dataset, # --- Use the tf.data.Dataset
    callbacks=[early_stopping, reduce_lr]
)
print("✅ Training complete.")

# --- 8. Evaluate and Show F1 Score ---
print("Evaluating model on test data...")
test_metrics = model.evaluate(test_dataset) # --- Use the tf.data.Dataset

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

# --- 9. Save the Model (Unchanged) ---
model.save('prescription_model.keras')
print("\n✅ Model saved as 'prescription_model.keras'")

# --- 10. Convert to TFLite for Flutter ---
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