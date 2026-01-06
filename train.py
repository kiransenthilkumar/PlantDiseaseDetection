import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping
import json
import os

# ---------------- CONFIG ----------------
IMG_SIZE = 128            # Can increase to 160 if system allows
BATCH_SIZE = 32
EPOCHS = 5
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_PATH = "fruit_disease_model.keras"
CLASS_INDEX_PATH = "class_indices.json"
# ---------------------------------------

# Optional: reduce TensorFlow log noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = train_data.num_classes
print(f"\nNumber of classes: {NUM_CLASSES}")

# Save class indices
with open(CLASS_INDEX_PATH, "w") as f:
    json.dump(train_data.class_indices, f)

# ---------------- MobileNet Model ----------------

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model (IMPORTANT)
base_model.trainable = False

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train model
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data,
    callbacks=[early_stop]
)

# Save model
model.save(MODEL_PATH)

print("\nTraining completed successfully!")
print(f"Model saved as: {MODEL_PATH}")
print(f"Class indices saved as: {CLASS_INDEX_PATH}")
