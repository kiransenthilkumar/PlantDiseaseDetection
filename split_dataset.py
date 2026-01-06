import os
import shutil
import random

SOURCE_DIR = "PlantVillage-Dataset/raw/color"
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

SPLIT_RATIO = 0.8  # 80% train, 20% test

random.seed(42)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_point = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_point]
    test_images = images[split_point:]

    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(TRAIN_DIR, class_name, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(TEST_DIR, class_name, img)
        )

    print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

print("\n Dataset split completed successfully!")
