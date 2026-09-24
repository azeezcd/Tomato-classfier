
import json

import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "plant_disease_model.pth"
MAPPING_PATH = "class_mapping.json"

IMAGE_SIZE = 224

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(
    MAPPING_PATH,
    "r",
    encoding="utf-8"
) as f:

    idx_to_class = json.load(f)

# JSON keys are strings, convert them back to int
idx_to_class = {
    int(key): value
    for key, value in idx_to_class.items()
}


num_classes = len(idx_to_class)


# ============================================================
# CREATE SAME RESNET34 MODEL
# ============================================================

model = models.resnet34(
    weights=None
)

in_features = model.fc.in_features

# MUST MATCH TRAINING CODE
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(
        in_features,
        num_classes
    )
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    state_dict
)

model = model.to(device)

model.eval()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

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
# PREDICTION FUNCTION
# ============================================================

def get_prediction(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.inference_mode():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    results = []

    for index, probability in enumerate(probabilities):

        results.append({
            "class": idx_to_class[index],
            "confidence": probability.item() * 100
        })

    results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return results

