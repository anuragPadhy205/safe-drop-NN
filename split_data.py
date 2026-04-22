import os
import shutil
import random

# --- 1. Define your current paths here ---
SOURCE_IMAGES = "training_set/images"
SOURCE_MASKS = "training_set/gt/semantic/label_images"

# --- 2. Define where the new clean split should go ---
BASE_OUT_DIR = "clean_dataset"
TRAIN_IMAGES = os.path.join(BASE_OUT_DIR, "train", "images")
TRAIN_MASKS = os.path.join(BASE_OUT_DIR, "train", "masks")
VAL_IMAGES = os.path.join(BASE_OUT_DIR, "val", "images")
VAL_MASKS = os.path.join(BASE_OUT_DIR, "val", "masks")

def create_dirs():
    """Creates the necessary output directories."""
    for d in [TRAIN_IMAGES, TRAIN_MASKS, VAL_IMAGES, VAL_MASKS]:
        os.makedirs(d, exist_ok=True)

def split_dataset(split_ratio=0.8):
    create_dirs()

    # Get all valid files (ignoring hidden Mac files like .DS_Store)
    images = sorted([f for f in os.listdir(SOURCE_IMAGES) if not f.startswith('.')])
    masks = sorted([f for f in os.listdir(SOURCE_MASKS) if not f.startswith('.')])

    if len(images) != len(masks):
        print(f"Error: Mismatch in file counts! Images: {len(images)} | Masks: {len(masks)}")
        return

    # Pair them up and randomize
    dataset = list(zip(images, masks))
    random.seed(42) # Keeps the random split consistent if you run it again
    random.shuffle(dataset)

    # Calculate the split index
    split_idx = int(len(dataset) * split_ratio)
    train_set = dataset[:split_idx]
    val_set = dataset[split_idx:]

    print(f"Total files: {len(dataset)} | Training: {len(train_set)} | Validation: {len(val_set)}")

    # Copy files to their new homes
    print("Copying Training files...")
    for img_name, mask_name in train_set:
        shutil.copy(os.path.join(SOURCE_IMAGES, img_name), os.path.join(TRAIN_IMAGES, img_name))
        shutil.copy(os.path.join(SOURCE_MASKS, mask_name), os.path.join(TRAIN_MASKS, mask_name))

    print("Copying Validation files...")
    for img_name, mask_name in val_set:
        shutil.copy(os.path.join(SOURCE_IMAGES, img_name), os.path.join(VAL_IMAGES, img_name))
        shutil.copy(os.path.join(SOURCE_MASKS, mask_name), os.path.join(VAL_MASKS, mask_name))

    print(f"Success! Your data is now safely split inside the '{BASE_OUT_DIR}' folder.")

if __name__ == "__main__":
    split_dataset()