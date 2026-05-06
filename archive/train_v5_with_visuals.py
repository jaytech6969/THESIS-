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
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom, RandomShear, RandomContrast
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score, precision_score, recall_score
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

    return X, Y_encoded, df['MEDICINE_NAME']

# --- 2. Load All Datasets (Unchanged) ---
print("Loading Training data...")
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

# --- 3. Calculate Class Weights (Unchanged) ---
print("Calculating class weights...")
class_names = np.unique(y_train_labels) 
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=class_names,
    y=y_train_labels
)
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

# --- 5. Data Augmentation Layers (Unchanged) ---
data_augmentation = models.Sequential([
    RandomRotation(0.05, fill_mode='nearest'),
    RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest'),
    RandomZoom(height_factor=0.1, fill_mode='nearest'),
    RandomShear(0.1, fill_mode='nearest'),
    RandomContrast(0.1)
], name='augmentation')

# --- 6. VGG-style CNN Model (Unchanged) ---
lambda_reg = 0.00001
input_tensor = Input(shape=(32, 128, 3))
x = data_augmentation(input_tensor)

# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Classifier Head
x = Flatten()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(lambda_reg))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(lambda_reg))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
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

# --- 8. Train with Class Weights (Unchanged) ---
print("Starting 'VGG-style' model training with class weights...")
history = model.fit(
    train_dataset,
    epochs=100, 
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict
)
print("✅ Training complete.")

# --- 9. --- NEW: Plot Training History ---
print("Generating training visuals...")
history_dict = history.history
epochs_range = range(1, len(history_dict['loss']) + 1)

# Find the exact metric keys (e.g., 'precision' or 'precision_1')
precision_key = [k for k in history_dict.keys() if 'precision' in k and 'val' not in k][0]
val_precision_key = [k for k in history_dict.keys() if 'precision' in k and 'val' in k][0]
recall_key = [k for k in history_dict.keys() if 'recall' in k and 'val' not in k][0]
val_recall_key = [k for k in history_dict.keys() if 'recall' in k and 'val' in k][0]

plt.figure(figsize=(20, 10))
plt.suptitle('Model Training History', fontsize=16, y=1.02)

# Plot 1: Accuracy
plt.subplot(2, 2, 1)
plt.plot(epochs_range, history_dict['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(epochs_range, history_dict['val_accuracy'], 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# Plot 2: Loss
plt.subplot(2, 2, 2)
plt.plot(epochs_range, history_dict['loss'], 'bo-', label='Training Loss')
plt.plot(epochs_range, history_dict['val_loss'], 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

# Plot 3: Precision
plt.subplot(2, 2, 3)
plt.plot(epochs_range, history_dict[precision_key], 'bo-', label='Training Precision')
plt.plot(epochs_range, history_dict[val_precision_key], 'ro-', label='Validation Precision')
plt.title('Training and Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend(loc='lower right')
plt.grid(True)

# Plot 4: Recall
plt.subplot(2, 2, 4)
plt.plot(epochs_range, history_dict[recall_key], 'bo-', label='Training Recall')
plt.plot(epochs_range, history_dict[val_recall_key], 'ro-', label='Validation Recall')
plt.title('Training and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('training_visuals.png')
print("✅ Training visuals saved to 'training_visuals.png'")
# --- End of new section ---


# --- 10. Evaluate and Show F1 Score (Previously step 9) ---
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

# --- 11. Save the Model (Previously step 10) ---
model.save('prescription_model.keras')
print("\n✅ Model saved as 'prescription_model.keras'")

# --- 12. Convert to TFLite for Flutter (Previously step 11) ---
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