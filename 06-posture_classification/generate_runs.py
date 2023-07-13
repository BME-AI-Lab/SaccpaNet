import shutil


def copy_and_write(representation, params):
    shutil.copytree("codes", f"runs/{representation}", dirs_exist_ok=True)
    with open(f"runs/{representation}/search_configs.py", "w") as f:
        f.write(params)


MODELS = [
    "ResNet34",
    "Resnet50",
    "Resnet101",
    "Resnet152",
    "EfficientNetB0",
    "EfficientNetB2",
    "EfficientNetB4",
    "EfficientNetB7",
    "ECA34",
    "ECA50",
    "ECA101",
    "ECA152",
    "SACCPA34",
    "SACCPA50",
    "SACCPA101",
    "SACCPA152",
]


def search_range():
    for models in MODELS:
        representation = f"{models}"
        params_str = f"CLASSIFICATION_MODELS = {models}\n"
        copy_and_write(representation, params_str)


if __name__ == "__main__":
    search_range()
