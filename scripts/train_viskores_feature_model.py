import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/features/enhanced_features.csv")

X = df[
    [
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
]

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
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="weighted")
recall = recall_score(y_test, preds, average="weighted")
f1 = f1_score(y_test, preds, average="weighted")

report = classification_report(
    y_test,
    preds,
    target_names=encoder.classes_
)

print(report)

with open("results/viskores_feature_classification_report.txt", "w") as f:
    f.write(report)

metrics = pd.DataFrame([{
    "model": "Viskores 3D Feature Random Forest",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])

metrics.to_csv("results/viskores_feature_metrics.csv", index=False)

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)
plt.title("Confusion Matrix - Viskores 3D Feature Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("results/viskores_confusion_matrix.png", dpi=300)
plt.close()

print("Viskores feature model results saved.")
