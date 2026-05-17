"""
Creates a small sample dataset with only 50 images per class
Perfect for GitHub upload and demonstration!
"""

import os
import shutil
import random

# Paths
FULL_DATASET = r"C:\Users\ASUS\OneDrive\Desktop\brain_tumor_project\dataset"
SAMPLE_DATASET = r"C:\Users\ASUS\OneDrive\Desktop\brain_tumor_project\sample_dataset"

# 50 images per class
IMAGES_PER_CLASS = 50

CLASSES = ["Normal", "Benign", "Malignant"]

def create_sample():
    print("Creating sample dataset...")

    for cls in CLASSES:
        # Source folder
        src_folder = os.path.join(FULL_DATASET, cls)
        # Destination folder
        dest_folder = os.path.join(SAMPLE_DATASET, cls)

        # Create destination folder
        os.makedirs(dest_folder, exist_ok=True)

        # Get all images
        all_images = os.listdir(src_folder)

        # Pick 50 random images
        selected = random.sample(all_images, min(IMAGES_PER_CLASS, len(all_images)))

        # Copy selected images
        for img in selected:
            src  = os.path.join(src_folder, img)
            dest = os.path.join(dest_folder, img)
            shutil.copy2(src, dest)

        print(f"✅ {cls} → copied {len(selected)} images")

    print(f"\n🎉 Sample dataset created at:")
    print(f"{SAMPLE_DATASET}")
    print(f"\nImages per class:")
    for cls in CLASSES:
        count = len(os.listdir(os.path.join(SAMPLE_DATASET, cls)))
        print(f"   {cls}: {count} images")
    print(f"\nTotal: {IMAGES_PER_CLASS * len(CLASSES)} images")
    print("\nNow you can upload sample_dataset to GitHub!")

if __name__ == "__main__":
    create_sample()
