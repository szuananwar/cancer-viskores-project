import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_DIR = "data/archive/Training"
TEST_DIR = "data/archive/Testing"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

os.makedirs("results", exist_ok=True)

train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_data.class_indices.keys())

model = models.Sequential([
    Input(shape=(128, 128, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

pred_probs = model.predict(test_data)
preds = np.argmax(pred_probs, axis=1)
true_labels = test_data.classes

report = classification_report(
    true_labels,
    preds,
    target_names=class_names
)

print(report)

with open("results/classification_report.txt", "w") as f:
    f.write(report)

accuracy = accuracy_score(true_labels, preds)
precision = precision_score(true_labels, preds, average="weighted")
recall = recall_score(true_labels, preds, average="weighted")
f1 = f1_score(true_labels, preds, average="weighted")

metrics_df = pd.DataFrame([{
    "model": "Baseline 2D CNN",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])

metrics_df.to_csv("results/baseline_2d_metrics.csv", index=False)

class_report_dict = classification_report(
    true_labels,
    preds,
    target_names=class_names,
    output_dict=True
)

class_metrics_df = pd.DataFrame(class_report_dict).transpose()
class_metrics_df.to_csv("results/baseline_2d_class_metrics.csv")

cm = confusion_matrix(true_labels, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix - Baseline 2D CNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy - Baseline 2D CNN")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/accuracy_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss - Baseline 2D CNN")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=300)
plt.close()

f1_scores = [
    class_report_dict[class_name]["f1-score"]
    for class_name in class_names
]

plt.figure(figsize=(8, 6))
plt.bar(class_names, f1_scores)
plt.title("F1-Score by Class - Baseline 2D CNN")
plt.xlabel("Class")
plt.ylabel("F1-Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/class_f1_scores.png", dpi=300)
plt.close()

model.save("results/baseline_2d_cnn.keras")

print("All baseline results saved in the results folder.")
