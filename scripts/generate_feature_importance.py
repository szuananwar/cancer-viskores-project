import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/features/enhanced_features.csv")

features = [
    "isosurface_points",
    "isosurface_cells",
    "mean_intensity",
    "median_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "q25_intensity",
    "q75_intensity",
    "intensity_range",
    "iqr_intensity",
    "voxel_count",
    "high_intensity_voxels",
    "voxel_ratio",
    "high_intensity_ratio",
    "point_cell_ratio",
    "cell_density",
    "point_density"
]

X = df[features]
y = df["class"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

importance_df.to_csv(
    "results/feature_importance.csv",
    index=False
)

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)
plt.title("Random Forest Feature Importance - Viskores 3D Feature Model")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=300)
plt.close()

print("Saved: results/feature_importance.png")
print("Saved: results/feature_importance.csv")
