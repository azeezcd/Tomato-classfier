
import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms, models

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "dataset"
MODEL_PATH = "plant_disease_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_RATIO = 0.15
SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("VALIDATION MODEL DIAGNOSTIC")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# DATASET
# ============================================================

full_dataset = datasets.ImageFolder(
    DATA_DIR
)

class_names = full_dataset.classes
num_classes = len(class_names)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, "->", name)


# ============================================================
# SAME STRATIFIED SPLIT USED DURING TRAINING
# ============================================================

def stratified_split(dataset, val_ratio, seed):

    generator = torch.Generator().manual_seed(seed)

    class_indices = defaultdict(list)

    for index, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(index)

    train_indices = []
    val_indices = []

    for label, indices in class_indices.items():

        indices = torch.tensor(
            indices,
            dtype=torch.long
        )

        permutation = torch.randperm(
            len(indices),
            generator=generator
        )

        indices = indices[permutation]

        val_count = max(
            1,
            int(len(indices) * val_ratio)
        )

        val_indices.extend(
            indices[:val_count].tolist()
        )

        train_indices.extend(
            indices[val_count:].tolist()
        )

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
# VALIDATION DATASET
# ============================================================

class ValidationSubset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        transform
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        original_index = self.indices[index]

        image_path, label = (
            self.dataset.samples[original_index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        return image, label


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose([

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


# ============================================================
# CREATE SAME VALIDATION SPLIT
# ============================================================

_, val_indices = stratified_split(
    full_dataset,
    VAL_RATIO,
    SEED
)

val_dataset = ValidationSubset(
    full_dataset,
    val_indices,
    val_transform
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print("\nValidation images:", len(val_dataset))


# ============================================================
# LOAD SAME RESNET34
# ============================================================

model = models.resnet34(
    weights=None
)

in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(
        in_features,
        num_classes
    )
)

model = model.to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    state_dict
)

model.eval()


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = torch.zeros(
    num_classes,
    num_classes,
    dtype=torch.int64
)

correct = 0
total = 0


# ============================================================
# VALIDATION
# ============================================================

with torch.inference_mode():

    for images, labels in val_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                outputs = model(images)

        else:
            outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        for real, pred in zip(
            labels.cpu(),
            predictions.cpu()
        ):
            confusion[
                real,
                pred
            ] += 1


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = (
    correct / total
) * 100

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Validation accuracy: {accuracy:.2f}%"
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\nPER-CLASS RESULTS")

for i, class_name in enumerate(
    class_names
):

    class_total = confusion[
        i
    ].sum().item()

    class_correct = confusion[
        i, i
    ].item()

    if class_total > 0:

        class_accuracy = (
            class_correct
            / class_total
        ) * 100

    else:
        class_accuracy = 0.0

    print(
        f"{class_name:<45}"
        f"{class_accuracy:6.2f}%"
        f"  ({class_correct}/{class_total})"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "Rows = REAL CLASS"
)

print(
    "Columns = PREDICTED CLASS\n"
)

# Header
print(
    " " * 18
    + " ".join(
        f"{i:>6}"
        for i in range(num_classes)
    )
)

for i, row in enumerate(confusion):

    print(
        f"{i:>2} "
        f"{class_names[i][:14]:<16}"
        + " ".join(
            f"{value.item():6d}"
            for value in row
        )
    )


# ============================================================
# MOST COMMON CONFUSIONS
# ============================================================

print("\n" + "=" * 70)
print("MOST COMMON CONFUSIONS")
print("=" * 70)

confusions = []

for real in range(num_classes):

    for pred in range(num_classes):

        if real != pred:

            amount = confusion[
                real,
                pred
            ].item()

            if amount > 0:

                confusions.append(
                    (
                        amount,
                        class_names[real],
                        class_names[pred]
                    )
                )

confusions.sort(
    reverse=True
)

for amount, real, pred in confusions[:10]:

    print(
        f"{amount:>4} images:"
        f" {real}"
        f" -> {pred}"
    )

print("\n" + "=" * 70)

