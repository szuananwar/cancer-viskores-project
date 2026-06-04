import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, filters, morphology

os.makedirs("results", exist_ok=True)

CLASS_NAME = "glioma"
sample_file = sorted(os.listdir(f"data/simulated_3d/{CLASS_NAME}"))[0]
volume_path = f"data/simulated_3d/{CLASS_NAME}/{sample_file}"

volume = np.load(volume_path)

middle_slice = volume[volume.shape[0] // 2]
projection = np.max(volume, axis=0)

# Create tumor-like mask from high-intensity region
threshold = np.percentile(volume, 92)
mask = volume > threshold

# Clean the mask
mask = morphology.remove_small_objects(mask, min_size=500)
mask = morphology.binary_closing(mask)

# Extract isosurface from binary mask
verts, faces, normals, values = measure.marching_cubes(mask.astype(float), level=0.5)

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(middle_slice, cmap="gray")
ax1.set_title("2D MRI Slice")
ax1.axis("off")

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(projection, cmap="gray")
ax2.set_title("Simulated 3D Volume\nMaximum Projection")
ax2.axis("off")

ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.plot_trisurf(
    verts[:, 2],
    verts[:, 1],
    faces,
    verts[:, 0],
    linewidth=0.1,
    antialiased=True,
    alpha=0.95
)

ax3.set_title("Extracted Tumor Isosurface")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")
ax3.view_init(elev=25, azim=45)
ax3.set_box_aspect((1, 1, 0.7))

plt.suptitle(
    "Example 3D Tumor Rendering from Simulated MRI Volume",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig("results/example_3d_rendering.png", dpi=300)
plt.close()

print("Saved: results/example_3d_rendering.png")
