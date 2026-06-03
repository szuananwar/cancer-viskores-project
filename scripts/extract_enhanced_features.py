import os
import numpy as np
import pandas as pd

VISKORES_CSV = "data/features/viskores_features.csv"
SIMULATED_DIR = "data/simulated_3d"
OUTPUT_FILE = "data/features/enhanced_features.csv"

df = pd.read_csv(VISKORES_CSV)

enhanced_rows = []

for _, row in df.iterrows():
    file_path = row["file"]
    class_name = row["class"]

    vtk_file = os.path.basename(file_path)
    npy_file = vtk_file.replace(".vtk", ".npy")

    npy_path = os.path.join(
        SIMULATED_DIR,
        class_name,
        npy_file
    )

    volume = np.load(npy_path)

    mean_intensity = np.mean(volume)
    median_intensity = np.median(volume)
    std_intensity = np.std(volume)
    min_intensity = np.min(volume)
    max_intensity = np.max(volume)

    q25_intensity = np.percentile(volume, 25)
    q75_intensity = np.percentile(volume, 75)
    intensity_range = max_intensity - min_intensity
    iqr_intensity = q75_intensity - q25_intensity

    total_voxels = volume.size
    voxel_count = np.count_nonzero(volume > 0.1)
    high_intensity_voxels = np.count_nonzero(volume > 0.6)

    voxel_ratio = voxel_count / total_voxels
    high_intensity_ratio = high_intensity_voxels / total_voxels

    isosurface_points = row["isosurface_points"]
    isosurface_cells = row["isosurface_cells"]

    point_cell_ratio = isosurface_points / max(isosurface_cells, 1)
    cell_density = isosurface_cells / max(voxel_count, 1)
    point_density = isosurface_points / max(voxel_count, 1)

    enhanced_rows.append({
        "class": class_name,

        "isosurface_points": isosurface_points,
        "isosurface_cells": isosurface_cells,

        "mean_intensity": mean_intensity,
        "median_intensity": median_intensity,
        "std_intensity": std_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,

        "q25_intensity": q25_intensity,
        "q75_intensity": q75_intensity,
        "intensity_range": intensity_range,
        "iqr_intensity": iqr_intensity,

        "voxel_count": voxel_count,
        "high_intensity_voxels": high_intensity_voxels,
        "voxel_ratio": voxel_ratio,
        "high_intensity_ratio": high_intensity_ratio,

        "point_cell_ratio": point_cell_ratio,
        "cell_density": cell_density,
        "point_density": point_density
    })

enhanced_df = pd.DataFrame(enhanced_rows)

enhanced_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(enhanced_df)} samples")
print(f"Output: {OUTPUT_FILE}")

print("\nColumns:")
print(enhanced_df.columns.tolist())
