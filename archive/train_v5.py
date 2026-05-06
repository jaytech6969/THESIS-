# train_v5.py
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# --- Import new augmentation layers ---
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom, RandomShear, RandomContrast
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score, precision_score, recall_score
# --- NEW: Import for class weights ---
from sklearn.utils import class_weight

# --- 1. Data Loading Functions (Unchanged) ---

def one_hot_encoded_Y(df):
    return pd.get_dummies(df['MEDICINE_NAME'], dtype=int)

def load_images(image_paths, target_size=(128, 32)):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image {path}. Skipping.")
            continue
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        images.append(img)
    images = np.array(images)
    images = images.reshape(images.shape[0], target_size[1], target_size[0], 3)
    return images

def load_dataset(folder_name):
    lower_name = folder_name.lower()
    labels_path = os.path.join(folder_name, f"{lower_name}_labels.csv")
    images_folder = os.path.join(folder_name, f"{lower_name}_words")
    
    if not os.path.exists(labels_path) or not os.path.exists(images_folder):
        print(f"Error: Data paths not found for {folder_name}")
        return None, None, None
        
    df = pd.read_csv(labels_path)
    Y_encoded = one_hot_encoded_Y(df)
    image_paths = [os.path.join(images_folder, filename) for filename in df['IMAGE']]
    X = load_images(image_paths)
    
    if folder_name == "Training":
        class_names = Y_encoded.columns.tolist()
        with open("labels.txt", "w") as f:
            for item in class_names:
                f.write(f"{item}\n")
        print(f"✅ Saved {len(class_names)} class names to labels.txt")

    # --- NEW: Return the original df for calculating weights ---
    return X, Y_encoded, df['MEDICINE_NAME']

# --- 2. Load All Datasets ---
print("Loading Training data...")
# --- NEW: Get y_train_labels for class weights ---
x_train, y_train, y_train_labels = load_dataset("Training")
print("Loading Validation data...")
x_val, y_val, _ = load_dataset("Validation")
print("Loading Testing data...")
x_test, y_test, _ = load_dataset("Testing")

if x_train is None:
    print("Failed to load training data. Exiting.")
    exit()

num_classes = y_train.shape[1]
print(f"Data loaded. Found {num_classes} classes.")

# --- 3. --- NEW: Calculate Class Weights ---
print("Calculating class weights...")
# Get unique class names in the order of training
class_names = np.unique(y_train_labels) 
# Compute weights
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=class_names,
    y=y_train_labels
)
# Create a dictionary mapping class index to its weight
class_weights_dict = dict(zip(range(len(weights)), weights))
print("✅ Class weights calculated.")

# --- 4. TF.DATA.DATASET (Unchanged) ---
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
print("✅ tf.data.Dataset created.")

# --- 5. --- UPDATED: Data Augmentation Layers ---
data_augmentation = models.Sequential([
    RandomRotation(0.05, fill_mode='nearest'),
    RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest'),
    RandomZoom(height_factor=0.1, fill_mode='nearest'),
    RandomShear(0.1, fill_mode='nearest'), # --- ADDED ---
    RandomContrast(0.1) # --- ADDED ---
], name='augmentation')

# --- 6. --- UPDATED: VGG-style CNN Model ---
lambda_reg = 0.00001
input_tensor = Input(shape=(32, 128, 3))
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
x = Dropout(0.4)(x) # --- UPDATED: Dropout reduced from 0.5 to 0.4 ---
x = Dense(512, activation='relu', kernel_regularizer=l2(lambda_reg))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x) # --- UPDATED: Dropout reduced from 0.5 to 0.4 ---
output_tensor = Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=input_tensor, outputs=output_tensor)

model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
model.summary()

# --- 7. Callbacks (Unchanged) ---
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

# --- 8. --- UPDATED: Train with Class Weights ---
print("Starting 'VGG-style' model training with class weights...")
history = model.fit(
    train_dataset,
    epochs=100, 
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict # --- ADDED CLASS WEIGHTS ---
)
print("✅ Training complete.")

# --- 9. Evaluate and Show F1 Score (Unchanged) ---
print("Evaluating model on test data...")
test_metrics = model.evaluate(test_dataset) 

test_loss = test_metrics[0]
test_acc = test_metrics[1]
test_precision = test_metrics[2]
test_recall = test_metrics[3]

f1_score_val = 0.0
if (test_precision + test_recall) > 0:
    f1_score_val = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print("\n--- Test Results ---")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1 Score:  {f1_score_val:.4f}")

# --- 10. Save the Model (Unchanged) ---
model.save('prescription_model.keras')
print("\n✅ Model saved as 'prescription_model.keras'")

# --- 11. Convert to TFLite for Flutter (Unchanged) ---
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