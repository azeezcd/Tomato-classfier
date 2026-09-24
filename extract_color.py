import os
import zipfile
import shutil

# 1. Locate the .zip file in the current project directory
zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]

if not zip_files:
    print("Error: No .zip file found in this project folder!")
    print("Please place your downloaded dataset .zip file inside 'MLMoel' and try again.")
    exit(1)

zip_filename = zip_files[0]
print(f"Found zip file: '{zip_filename}'")

# 2. Reset dataset directory
if os.path.exists("dataset"):
    shutil.rmtree("dataset")

selected_classes = {
    "Apple___Apple_scab": "Apple_Scab",
    "Apple___Black_rot": "Apple_Black_Rot",
    "Apple___Cedar_apple_rust": "Apple_Rust",
    "Apple___healthy": "Apple_Healthy",
    "Tomato___Early_blight": "Tomato_Early_Blight",
    "Tomato___Late_blight": "Tomato_Late_Blight",
    "Tomato___Leaf_Mold": "Tomato_Leaf_Mold",
    "Tomato___healthy": "Tomato_Healthy"
}

# 3. Unzip into temporary folder
temp_dir = "_temp_extract"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

print(f"Unzipping '{zip_filename}'...")
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# 4. Extract full-color directories
os.makedirs("dataset", exist_ok=True)
print("\nOrganizing color leaf images...")

for src_folder_name, target_folder_name in selected_classes.items():
    found = False
    for root, dirs, files in os.walk(temp_dir):
        root_lower = root.lower()
        # Skip grayscale and segmented folders
        if "grayscale" in root_lower or "segmented" in root_lower:
            continue

        if src_folder_name in dirs:
            src_path = os.path.join(root, src_folder_name)
            dest_path = os.path.join("dataset", target_folder_name)

            shutil.copytree(src_path, dest_path)
            img_count = len([f for f in os.listdir(dest_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  ✓ {target_folder_name}: {img_count} color images")
            found = True
            break

    if not found:
        print(f"  ✗ Warning: Could not find folder for {src_folder_name}")

# Clean up temp folder
shutil.rmtree(temp_dir)

print("\nSUCCESS! Dataset ready in 'dataset/'.")