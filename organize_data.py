import os
import shutil

ARCHIVE_PATH = r"C:\Users\ASUS\OneDrive\Desktop\brain_tumor_project"
OUTPUT_PATH = r"C:\Users\ASUS\OneDrive\Desktop\brain_tumor_project\dataset"

MAPPING = {
    "notumor"    : "Normal",
    "meningioma" : "Benign",
    "glioma"     : "Malignant",
    "pituitary"  : "Malignant",
}

def organize():
    for cls in ["Normal", "Benign", "Malignant"]:
        os.makedirs(os.path.join(OUTPUT_PATH, cls), exist_ok=True)

    total = 0

    for split in ["Training", "Testing"]:
        split_path = os.path.join(ARCHIVE_PATH, split)

        if not os.path.exists(split_path):
            print(f"Could not find folder: {split_path}")
            continue

        for kaggle_folder, our_class in MAPPING.items():
            src_folder = os.path.join(split_path, kaggle_folder)

            if not os.path.exists(src_folder):
                print(f"Skipping (not found): {src_folder}")
                continue

            images = os.listdir(src_folder)
            for img_name in images:
                src  = os.path.join(src_folder, img_name)
                dest = os.path.join(OUTPUT_PATH, our_class, img_name)

                if os.path.exists(dest):
                    base, ext = os.path.splitext(img_name)
                    dest = os.path.join(OUTPUT_PATH, our_class, f"{base}_{split}{ext}")

                shutil.copy2(src, dest)
                total += 1

            print(f"Copied {len(images)} images: {split}/{kaggle_folder} -> {our_class}")

    print(f"\nDone! Total images: {total}")
    print(f"Dataset ready at: {OUTPUT_PATH}")

    print("\nImages per class:")
    for cls in ["Normal", "Benign", "Malignant"]:
        count = len(os.listdir(os.path.join(OUTPUT_PATH, cls)))
        print(f"   {cls}: {count} images")

if __name__ == "__main__":
    organize()
