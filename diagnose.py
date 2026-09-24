
import os
import random

from predict import get_prediction


DATASET_DIR = "dataset"


# ------------------------------------------------------------
# Test one random image from every class
# ------------------------------------------------------------

print("=" * 70)
print("TESTING ONE IMAGE FROM EVERY TRAINING CLASS")
print("=" * 70)

classes = sorted(
    folder
    for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(
        os.path.join(DATASET_DIR, folder)
    )
)

for class_name in classes:

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    images = [
        file
        for file in os.listdir(class_dir)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    if not images:
        continue

    image_name = random.choice(images)

    image_path = os.path.join(
        class_dir,
        image_name
    )

    result = get_prediction(
        image_path
    )

    print("\n" + "-" * 70)

    print(
        "REAL CLASS:   ",
        class_name
    )

    print(
        "IMAGE:        ",
        image_name
    )

    print(
        "PREDICTION:   ",
        result["class"]
    )

    print(
        "CONFIDENCE:   ",
        f"{result['confidence']:.2f}%"
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
