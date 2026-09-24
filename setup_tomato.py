import os
import shutil

source_dir = "archive"
target_dir = "dataset"

if not os.path.exists(source_dir):
    print(f"Error: Could not find '{source_dir}' folder.")
    exit(1)

# Reset dataset directory
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir, exist_ok=True)

print("Setting up Tomato-Only dataset...\n")

copied_folders = set()

for root, dirs, files in os.walk(source_dir):
    root_lower = root.lower()
    # Strictly exclude grayscale and segmented directories
    if "grayscale" in root_lower or "segmented" in root_lower:
        continue

    for d in dirs:
        if "tomato" in d.lower():
            src_path = os.path.join(root, d)

            # Standardize folder names (e.g. Tomato___Early_blight -> Tomato_Early_Blight)
            parts = d.split("___")
            if len(parts) == 2:
                clean_name = f"Tomato_{parts[1].replace('_', ' ').title().replace(' ', '_')}"
            else:
                clean_name = d

            dest_path = os.path.join(target_dir, clean_name)

            if clean_name not in copied_folders:
                shutil.copytree(src_path, dest_path)
                copied_folders.add(clean_name)
                img_count = len([f for f in os.listdir(dest_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  ✓ {clean_name}: {img_count} full-color images")

print(f"\nSUCCESS! Created Tomato-only dataset with {len(copied_folders)} classes in 'dataset/'.")
print("Next step: Run 'python train.py'")