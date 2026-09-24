import os
import shutil

# Keywords required to identify each of the 8 target classes
class_rules = {
    "Apple_Scab": ["apple", "scab"],
    "Apple_Black_Rot": ["apple", "black", "rot"],
    "Apple_Rust": ["apple", "rust"],
    "Apple_Healthy": ["apple", "healthy"],
    "Tomato_Early_Blight": ["tomato", "early"],
    "Tomato_Late_Blight": ["tomato", "late"],
    "Tomato_Leaf_Mold": ["tomato", "mold"],
    "Tomato_Healthy": ["tomato", "healthy"]
}

source_dir = "archive"
target_dir = "dataset"

if not os.path.exists(source_dir):
    print(f"Error: Folder '{source_dir}' not found.")
    exit(1)

# Reset dataset directory
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir, exist_ok=True)

print("Scanning 'archive' with flexible folder matching...\n")

found_classes = {}

for root, dirs, files in os.walk(source_dir):
    root_lower = root.lower()
    # Skip grayscale and segmented folders
    if "grayscale" in root_lower or "segmented" in root_lower:
        continue

    for d in dirs:
        d_lower = d.lower()
        for target_name, keywords in class_rules.items():
            if target_name not in found_classes:
                if all(kw in d_lower for kw in keywords):
                    src_path = os.path.join(root, d)
                    dest_path = os.path.join(target_dir, target_name)

                    shutil.copytree(src_path, dest_path)
                    img_count = len([f for f in os.listdir(dest_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  ✓ {target_name}: {img_count} color images (from '{d}')")
                    found_classes[target_name] = True

print(f"\nSuccessfully organized {len(found_classes)}/8 classes.")

if len(found_classes) == 8:
    print("\nSUCCESS! All 8 classes are ready in 'dataset/'.")
    print("Next step: Run 'python train.py'")
else:
    missing = [k for k in class_rules if k not in found_classes]
    print(f"\nMissing classes: {missing}")