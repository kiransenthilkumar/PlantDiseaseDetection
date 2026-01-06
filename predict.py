import tensorflow as tf
import numpy as np
import cv2
import json
import sys

MODEL_PATH = "fruit_disease_model.keras"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = 128

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping
class_names = {v: k for k, v in class_indices.items()}

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

img = cv2.imread(image_path)
if img is None:
    print("Image not found")
    sys.exit(1)

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

prediction = model.predict(img)
class_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("\nPrediction Result")
print("-----------------")
print(f"Disease / Class : {class_names[class_index]}")
print(f"Confidence      : {confidence:.2f}%")
