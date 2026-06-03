import os
import numpy as np

INPUT_DIR = "data/simulated_3d"
OUTPUT_DIR = "data/vtk"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def write_legacy_vtk(volume, output_path):
    depth, height, width = volume.shape

    with open(output_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Simulated MRI Volume\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {width} {height} {depth}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n")
        f.write(f"POINT_DATA {width * height * depth}\n")
        f.write("SCALARS MRI float 1\n")
        f.write("LOOKUP_TABLE default\n")

        flat = volume.flatten(order="F")
        for value in flat:
            f.write(f"{value}\n")

for class_name in CLASSES:
    class_input = os.path.join(INPUT_DIR, class_name)
    class_output = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_output, exist_ok=True)

    files = [f for f in os.listdir(class_input) if f.endswith(".npy")]

    for filename in files:
        volume = np.load(os.path.join(class_input, filename))
        base = os.path.splitext(filename)[0]
        output_path = os.path.join(class_output, base + ".vtk")
        write_legacy_vtk(volume, output_path)

    print(f"{class_name}: converted {len(files)} volumes to legacy VTK")

print("Done converting NPY to VTK.")
