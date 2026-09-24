import os
import shutil

source_dir = "archive"
target_dir = os.path.join("dataset", "Tomato_Healthy")

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

found_path = None

for root, dirs, files in os.walk(source_dir):
    root_lower = root.lower()
    # Skip non-color folders
    if "grayscale" in root_lower or "segmented" in root_lower:
        continue
    for d in dirs:
        d_lower = d.lower()
        if "tomato" in d_lower and "health" in d_lower:
            found_path = os.path.join(root, d)
            print(f"Found folder: '{d}' at '{found_path}'")
            break
    if found_path:
        break

if found_path:
    shutil.copytree(found_path, target_dir)
    img_count = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"✓ Successfully added Tomato_Healthy with {img_count} color images!")

    total_classes = len(os.listdir("dataset"))
    print(f"\nDataset readiness: {total_classes}/8 classes present in 'dataset/'.")
    if total_classes == 8:
        print("Ready to train! Run: python train.py")
else:
    print("Could not find a healthy tomato folder automatically.")
    print("Here are all tomato-related folders inside 'archive':")
    for root, dirs, files in os.walk(source_dir):
        for d in dirs:
            if "tomato" in d.lower():
                print(f"  - {os.path.join(root, d)}")