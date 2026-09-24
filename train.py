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

# KEEP THESE NAMES FOR COMPATIBILITY WITH YOUR OTHER FILES
MODEL_SAVE_PATH = "plant_disease_model.pth"
MAPPING_SAVE_PATH = "class_mapping.json"

BATCH_SIZE = 32

# Total maximum epochs
EPOCHS = 10

# First 2 epochs: train only classifier
WARMUP_EPOCHS = 2

IMAGE_SIZE = 224

# Validation percentage
VAL_RATIO = 0.15

SEED = 42

# Learning rates
HEAD_LR = 0.001
FINE_TUNE_LR = 0.0001

WEIGHT_DECAY = 1e-4


# ============================================================
# DEVICE
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

    # Usually improves performance when all images have
    # the same dimensions.
    torch.backends.cudnn.benchmark = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("PLANT DISEASE TRAINING")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
else:
    print(
        "WARNING: CUDA is not available."
        " Training will be much slower on CPU."
    )


# ============================================================
# 1. AUDIT DATASET
# ============================================================

def audit_and_clean_dataset(data_dir):

    print("\n" + "=" * 60)
    print("AUDITING DATASET")
    print("=" * 60)

    total_scanned = 0
    total_removed = 0

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    for class_folder in sorted(os.listdir(data_dir)):

        class_path = os.path.join(
            data_dir,
            class_folder
        )

        if not os.path.isdir(class_path):
            continue

        class_scanned = 0
        class_removed = 0

        for file in os.listdir(class_path):

            if not file.lower().endswith(
                valid_extensions
            ):
                continue

            file_path = os.path.join(
                class_path,
                file
            )

            class_scanned += 1
            total_scanned += 1

            try:

                # Verify file
                with Image.open(file_path) as img:
                    img.verify()

                # Actually load file
                with Image.open(file_path) as img:
                    img.load()

            except Exception:

                try:
                    os.remove(file_path)

                    class_removed += 1
                    total_removed += 1

                except Exception:
                    pass

        valid_images = (
            class_scanned - class_removed
        )

        print(
            f"{class_folder:<40}"
            f"{valid_images:>6} valid | "
            f"{class_removed:>4} corrupt removed"
        )

    print("-" * 60)

    print(
        f"Total scanned: {total_scanned}"
    )

    print(
        f"Corrupt removed: {total_removed}"
    )

    print(
        f"Valid images: "
        f"{total_scanned - total_removed}"
    )


# ============================================================
# 2. CUSTOM DATASET
# ============================================================

class TransformedSubset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        transform=None
    ):

        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        original_idx = self.indices[idx]

        image_path, label = (
            self.dataset.samples[original_idx]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# 3. STRATIFIED TRAIN/VALIDATION SPLIT
# ============================================================

def stratified_split(
    dataset,
    val_ratio,
    seed
):

    generator = torch.Generator().manual_seed(
        seed
    )

    class_indices = defaultdict(list)

    # Group image indexes by class
    for index, (_, label) in enumerate(
        dataset.samples
    ):

        class_indices[label].append(index)

    train_indices = []
    val_indices = []

    for label, indices in class_indices.items():

        indices = torch.tensor(
            indices,
            dtype=torch.long
        )

        # Shuffle each class separately
        permutation = torch.randperm(
            len(indices),
            generator=generator
        )

        indices = indices[permutation]

        val_count = max(
            1,
            int(len(indices) * val_ratio)
        )

        val_class = indices[
            :val_count
        ]

        train_class = indices[
            val_count:
        ]

        train_indices.extend(
            train_class.tolist()
        )

        val_indices.extend(
            val_class.tolist()
        )

    # Shuffle final train/validation indexes
    train_indices = torch.tensor(
        train_indices,
        dtype=torch.long
    )

    val_indices = torch.tensor(
        val_indices,
        dtype=torch.long
    )

    train_indices = train_indices[
        torch.randperm(
            len(train_indices),
            generator=generator
        )
    ]

    val_indices = val_indices[
        torch.randperm(
            len(val_indices),
            generator=generator
        )
    ]

    return (
        train_indices.tolist(),
        val_indices.tolist()
    )


# ============================================================
# 4. CLASS COUNTS
# ============================================================

def get_class_counts(
    dataset,
    indices,
    num_classes
):

    counts = [0] * num_classes

    for idx in indices:

        _, label = dataset.samples[idx]

        counts[label] += 1

    return counts


# ============================================================
# 5. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    use_amp
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    loss = running_loss / total

    accuracy = (
        correct / total
    ) * 100

    return loss, accuracy


# ============================================================
# 6. VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    num_classes
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    per_class_correct = [
        0
    ] * num_classes

    per_class_total = [
        0
    ] * num_classes

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            if device.type == "cuda":

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

            else:

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct_mask = (
                predictions == labels
            )

            correct += (
                correct_mask.sum().item()
            )

            total += labels.size(0)

            # Per-class accuracy
            for class_idx in range(
                num_classes
            ):

                mask = (
                    labels == class_idx
                )

                class_total = (
                    mask.sum().item()
                )

                if class_total > 0:

                    per_class_total[
                        class_idx
                    ] += class_total

                    per_class_correct[
                        class_idx
                    ] += (
                        predictions[mask]
                        == class_idx
                    ).sum().item()

    loss = (
        running_loss / total
    )

    accuracy = (
        correct / total
    ) * 100

    return (
        loss,
        accuracy,
        per_class_correct,
        per_class_total
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.exists(DATA_DIR):

        print(
            f"\nERROR: '{DATA_DIR}' folder does not exist."
        )

        return

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    audit_and_clean_dataset(
        DATA_DIR
    )

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    train_transforms = transforms.Compose([

        transforms.Resize(
            (256, 256)
        ),

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.85, 1.0)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            15
        ),

        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    val_transforms = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    full_dataset = datasets.ImageFolder(
        DATA_DIR
    )

    class_names = (
        full_dataset.classes
    )

    class_to_idx = (
        full_dataset.class_to_idx
    )

    num_classes = len(
        class_names
    )

    print("\n" + "=" * 60)
    print("DATASET")
    print("=" * 60)

    print(
        f"Total images: {len(full_dataset)}"
    )

    print(
        f"Number of classes: {num_classes}"
    )

    print("\nClasses:")

    for i, class_name in enumerate(
        class_names
    ):

        print(
            f"  {i}: {class_name}"
        )

    # --------------------------------------------------------
    # SAVE CLASS MAPPING
    # --------------------------------------------------------

    idx_to_class = {
        value: key
        for key, value
        in class_to_idx.items()
    }

    with open(
        MAPPING_SAVE_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            idx_to_class,
            f,
            indent=4
        )

    print(
        f"\nSaved mapping to "
        f"'{MAPPING_SAVE_PATH}'"
    )

    # --------------------------------------------------------
    # STRATIFIED SPLIT
    # --------------------------------------------------------

    train_indices, val_indices = (
        stratified_split(
            full_dataset,
            VAL_RATIO,
            SEED
        )
    )

    train_data = TransformedSubset(
        full_dataset,
        train_indices,
        train_transforms
    )

    val_data = TransformedSubset(
        full_dataset,
        val_indices,
        val_transforms
    )

    print("\n" + "=" * 60)
    print("DATA SPLIT")
    print("=" * 60)

    print(
        f"Training images:   {len(train_data)}"
    )

    print(
        f"Validation images: {len(val_data)}"
    )

    # --------------------------------------------------------
    # CLASS BALANCING
    # --------------------------------------------------------

    train_counts = get_class_counts(
        full_dataset,
        train_indices,
        num_classes
    )

    print("\nTraining class counts:")

    for i, class_name in enumerate(
        class_names
    ):

        print(
            f"{class_name:<40}"
            f"{train_counts[i]}"
        )

    total_train = sum(
        train_counts
    )

    class_weights = []

    for count in train_counts:

        if count == 0:
            weight = 1.0

        else:
            weight = (
                total_train / count
            ) ** 0.5

        class_weights.append(
            weight
        )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    class_weights = (
        class_weights
        / class_weights.mean()
    )

    class_weights = (
        class_weights.to(device)
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    cpu_count = os.cpu_count() or 2

    num_workers = min(
        4,
        cpu_count
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        )
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        )
    )

    print(
        f"\nDataLoader workers: "
        f"{num_workers}"
    )

    # --------------------------------------------------------
    # LOAD PRETRAINED RESNET34
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LOADING RESNET34")
    print("=" * 60)

    try:

        weights = (
            models.ResNet34_Weights.DEFAULT
        )

        model = models.resnet34(
            weights=weights
        )

    except AttributeError:

        # Compatibility with older torchvision
        model = models.resnet34(
            pretrained=True
        )

    # KEEP SAME FINAL ARCHITECTURE
    # FOR COMPATIBILITY WITH OTHER FILES

    in_features = (
        model.fc.in_features
    )

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            in_features,
            num_classes
        )
    )

    model = model.to(device)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    # --------------------------------------------------------
    # STAGE 1
    #
    # FREEZE ENTIRE RESNET
    # TRAIN ONLY CLASSIFIER
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 1: TRAINING CLASSIFIER")
    print("=" * 60)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(
        model.fc.parameters(),
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # MIXED PRECISION
    # --------------------------------------------------------

    use_amp = (
        device.type == "cuda"
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda"
        )
        if use_amp
        else None
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_val_acc = 0.0

    best_model_state = None

    epochs_without_improvement = 0

    start_time = time.time()

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        EPOCHS
    ):

        # ----------------------------------------------------
        # STAGE 2
        #
        # UNFREEZE ONLY LAYER 4
        # ----------------------------------------------------

        if epoch == WARMUP_EPOCHS:

            print("\n" + "=" * 60)
            print(
                "STAGE 2: FINE-TUNING RESNET34 LAYER 4"
            )
            print("=" * 60)

            # Unfreeze only deepest feature block
            for param in (
                model.layer4.parameters()
            ):
                param.requires_grad = True

            # Layer 4 gets smaller learning rate
            # than classifier

            optimizer = optim.AdamW(
                [
                    {
                        "params":
                            model.layer4.parameters(),
                        "lr":
                            FINE_TUNE_LR
                    },

                    {
                        "params":
                            model.fc.parameters(),
                        "lr":
                            HEAD_LR * 0.5
                    }
                ],
                weight_decay=WEIGHT_DECAY
            )

            scheduler = (
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(
                        1,
                        EPOCHS
                        - WARMUP_EPOCHS
                    )
                )
            )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                use_amp
            )
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        (
            val_loss,
            val_acc,
            per_class_correct,
            per_class_total
        ) = validate(
            model,
            val_loader,
            criterion,
            num_classes
        )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        if epoch >= WARMUP_EPOCHS:

            scheduler.step()

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print("\n" + "-" * 60)

        print(
            f"Epoch "
            f"[{epoch + 1:02d}/{EPOCHS:02d}]"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_acc:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_acc:.2f}%"
        )

        print("\nPer-class accuracy:")

        for class_idx, class_name in (
            enumerate(class_names)
        ):

            if per_class_total[
                class_idx
            ] > 0:

                class_accuracy = (
                    per_class_correct[
                        class_idx
                    ]
                    /
                    per_class_total[
                        class_idx
                    ]
                ) * 100

            else:

                class_accuracy = 0.0

            print(
                f"  "
                f"{class_name:<35}"
                f"{class_accuracy:6.2f}%"
            )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_model_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            epochs_without_improvement = 0

            print(
                f"\n✓ BEST MODEL SAVED"
                f" ({best_val_acc:.2f}%)"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement >= 3
        ):

            print(
                "\nValidation accuracy has "
                "not improved for 3 epochs."
            )

            print(
                "Stopping early."
            )

            break

    # ========================================================
    # FINISH
    # ========================================================

    total_time = (
        time.time() - start_time
    ) / 60

    # Restore best model in memory
    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Training time: "
        f"{total_time:.2f} minutes"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.2f}%"
    )

    print(
        f"Model: {MODEL_SAVE_PATH}"
    )

    print(
        f"Mapping: {MAPPING_SAVE_PATH}"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

