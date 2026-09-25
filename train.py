import os
import json
import time
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "dataset"

MODEL_SAVE_PATH = "plant_disease_model.pth"
MAPPING_SAVE_PATH = "class_mapping.json"

BATCH_SIZE = 32
EPOCHS = 10
WARMUP_EPOCHS = 2
IMAGE_SIZE = 224

VAL_RATIO = 0.15
SEED = 42

HEAD_LR = 0.001
FINE_TUNE_LR = 0.0001
WEIGHT_DECAY = 1e-4

# ============================================================
# DEVICE & REPRODUCIBILITY
# ============================================================
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("PLANT DISEASE TRAINING (FIELD-ROBUST)")
print("=" * 60)
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is not available. Training will run on CPU.")


# ============================================================
# 1. AUDIT DATASET
# ============================================================
def audit_and_clean_dataset(data_dir):
    print("\n" + "=" * 60)
    print("AUDITING DATASET")
    print("=" * 60)

    total_scanned = 0
    total_removed = 0
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for class_folder in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        class_scanned = 0
        class_removed = 0

        for file in os.listdir(class_path):
            if not file.lower().endswith(valid_extensions):
                continue

            file_path = os.path.join(class_path, file)
            class_scanned += 1
            total_scanned += 1

            try:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    img.load()
            except Exception:
                try:
                    os.remove(file_path)
                    class_removed += 1
                    total_removed += 1
                except Exception:
                    pass

        valid_images = class_scanned - class_removed
        print(f"{class_folder:<40} {valid_images:>6} valid | {class_removed:>4} corrupt removed")

    print("-" * 60)
    print(f"Total scanned: {total_scanned}")
    print(f"Corrupt removed: {total_removed}")
    print(f"Valid images: {total_scanned - total_removed}")


# ============================================================
# 2. CUSTOM DATASET
# ============================================================
class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image_path, label = self.dataset.samples[original_idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# 3. STRATIFIED SPLIT
# ============================================================
def stratified_split(dataset, val_ratio, seed):
    generator = torch.Generator().manual_seed(seed)
    class_indices = defaultdict(list)

    for index, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(index)

    train_indices = []
    val_indices = []

    for label, indices in class_indices.items():
        indices = torch.tensor(indices, dtype=torch.long)
        permutation = torch.randperm(len(indices), generator=generator)
        indices = indices[permutation]

        val_count = max(1, int(len(indices) * val_ratio))
        val_class = indices[:val_count]
        train_class = indices[val_count:]

        train_indices.extend(train_class.tolist())
        val_indices.extend(val_class.tolist())

    train_indices = torch.tensor(train_indices, dtype=torch.long)
    val_indices = torch.tensor(val_indices, dtype=torch.long)

    train_indices = train_indices[torch.randperm(len(train_indices), generator=generator)]
    val_indices = val_indices[torch.randperm(len(val_indices), generator=generator)]

    return train_indices.tolist(), val_indices.tolist()


# ============================================================
# 4. CLASS COUNTS
# ============================================================
def get_class_counts(dataset, indices, num_classes):
    counts = [0] * num_classes
    for idx in indices:
        _, label = dataset.samples[idx]
        counts[label] += 1
    return counts


# ============================================================
# 5. TRAINING FUNCTION
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, (correct / total) * 100


# ============================================================
# 6. VALIDATION FUNCTION
# ============================================================
def validate(model, loader, criterion, num_classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for class_idx in range(num_classes):
                mask = (labels == class_idx)
                class_tot = mask.sum().item()
                if class_tot > 0:
                    per_class_total[class_idx] += class_tot
                    per_class_correct[class_idx] += (predictions[mask] == class_idx).sum().item()

    return running_loss / total, (correct / total) * 100, per_class_correct, per_class_total


# ============================================================
# 7. MAIN PIPELINE
# ============================================================
def main():
    if not os.path.exists(DATA_DIR):
        print(f"\nERROR: '{DATA_DIR}' folder does not exist.")
        return

    audit_and_clean_dataset(DATA_DIR)

    # --------------------------------------------------------
    # REAL-WORLD HEAVY DATA AUGMENTATION PIPELINE
    # --------------------------------------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        # Widen crop scale to force focus on sub-patches / distant leaves
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        # Heavy ColorJitter to combat glare, shadows, and phone sensors
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # RandomErasing simulates fingers holding leaves or field occlusion
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), value="random")
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset structure
    full_dataset = datasets.ImageFolder(DATA_DIR)
    class_names = full_dataset.classes
    class_to_idx = full_dataset.class_to_idx
    num_classes = len(class_names)

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total images: {len(full_dataset)}")
    print(f"Number of classes: {num_classes}")
    print("\nClasses:")
    for i, class_name in enumerate(class_names):
        print(f"  {i}: {class_name}")

    # Save mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(MAPPING_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=4)
    print(f"\nSaved class mapping to '{MAPPING_SAVE_PATH}'")

    # Split
    train_indices, val_indices = stratified_split(full_dataset, VAL_RATIO, SEED)
    train_data = TransformedSubset(full_dataset, train_indices, train_transforms)
    val_data = TransformedSubset(full_dataset, val_indices, val_transforms)

    print("\n" + "=" * 60)
    print("DATA SPLIT")
    print("=" * 60)
    print(f"Training images:   {len(train_data)}")
    print(f"Validation images: {len(val_data)}")

    # Class Weights for Imbalance
    train_counts = get_class_counts(full_dataset, train_indices, num_classes)
    total_train = sum(train_counts)

    class_weights = []
    for count in train_counts:
        weight = 1.0 if count == 0 else (total_train / count) ** 0.5
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = (class_weights / class_weights.mean()).to(device)

    # DataLoaders
    cpu_count = os.cpu_count() or 2
    num_workers = min(4, cpu_count)
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    # ResNet34 Setup
    print("\n" + "=" * 60)
    print("LOADING PRETRAINED RESNET34")
    print("=" * 60)
    try:
        weights = models.ResNet34_Weights.DEFAULT
        model = models.resnet34(weights=weights)
    except AttributeError:
        model = models.resnet34(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # STAGE 1: Freeze backbone
    print("\n" + "=" * 60)
    print("STAGE 1: TRAINING CLASSIFIER HEAD")
    print("=" * 60)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.fc.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    start_time = time.time()

    # Loop
    for epoch in range(EPOCHS):
        # STAGE 2: Unfreeze Layer 4
        if epoch == WARMUP_EPOCHS:
            print("\n" + "=" * 60)
            print("STAGE 2: UNFREEZING RESNET34 LAYER 4 FOR FINE-TUNING")
            print("=" * 60)
            for param in model.layer4.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW([
                {"params": model.layer4.parameters(), "lr": FINE_TUNE_LR},
                {"params": model.fc.parameters(), "lr": HEAD_LR * 0.5}
            ], weight_decay=WEIGHT_DECAY)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS)
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, use_amp)
        val_loss, val_acc, per_class_corr, per_class_tot = validate(model, val_loader, criterion, num_classes)

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        print("\n" + "-" * 60)
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print("\nPer-class accuracy:")
        for idx, name in enumerate(class_names):
            acc = (per_class_corr[idx] / per_class_tot[idx] * 100) if per_class_tot[idx] > 0 else 0.0
            print(f"  {name:<40} {acc:6.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_without_improvement = 0
            print(f"\n✓ BEST MODEL SAVED ({best_val_acc:.2f}%)")

            # Auto-sync to GitHub on improvement
            try:
                os.system("git add class_mapping.json plant_disease_model.pth train.py 2>/dev/null")
                os.system('git commit -m "Auto-update trained model weights" 2>/dev/null')
                os.system("git push origin main 2>/dev/null &")
            except Exception:
                pass
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 3:
            print("\nEarly stopping triggered (3 epochs without improvement).")
            break

    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Time Elapsed: {total_time:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Saved Weights: {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()