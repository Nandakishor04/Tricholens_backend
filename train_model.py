import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_DIR = os.path.join(os.getcwd(), 'actual data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')

print("Starting training process...")
print(f"Loading data from: {DATA_DIR}")

# 2. Data Preparation
# We use data augmentation to get the most out of your 157 images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['AGA', 'normal'] # Explicitly only use these two
    )

    validation_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['AGA', 'normal']
    )

    # 3. Build Model (using MobileNetV2 for speed and efficiency)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False # Start by only training the top layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax') # 2 classes: AGA and normal
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Training
    print("\nTraining Phase 1: Warming up top layers...")
    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )

    # 5. Export to TFLite
    print("\nTraining complete. Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("\nSUCCESS! New 'model.tflite' has been created and saved.")
    print("Restart your single_server.py to use the new brain.")

except Exception as e:
    print(f"\nERROR: Training failed. Make sure your folders exist. {e}")
