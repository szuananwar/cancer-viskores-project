import os
import numpy as np
from PIL import Image

INPUT_DIR = "data/archive/Training"
OUTPUT_DIR = "data/simulated_3d"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 128
DEPTH = 64
MAX_IMAGES_PER_CLASS = 1400

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_volume_from_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    slice2d = np.array(img).astype(np.float32) / 255.0

    volume = []

    for z in range(DEPTH):
        scale = np.sin(np.pi * z / DEPTH)
        simulated_slice = slice2d * scale
        volume.append(simulated_slice)

    volume = np.stack(volume, axis=0)
    return volume

for class_name in CLASSES:
    class_input = os.path.join(INPUT_DIR, class_name)
    class_output = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_output, exist_ok=True)

    files = [
        f for f in os.listdir(class_input)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files = files[:MAX_IMAGES_PER_CLASS]

    for filename in files:
        input_path = os.path.join(class_input, filename)
        volume = create_volume_from_image(input_path)

        base = os.path.splitext(filename)[0]
        output_path = os.path.join(class_output, base + ".npy")

        np.save(output_path, volume)

    print(f"{class_name}: saved {len(files)} simulated 3D volumes")

print("Done creating simulated 3D volumes.")
