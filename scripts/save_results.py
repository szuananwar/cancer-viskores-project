import pandas as pd

metrics = {
    "Metric":["Accuracy","Precision","Recall","F1"],
    "Score":[0.87,0.89,0.87,0.87]
}

pd.DataFrame(metrics).to_csv(
    "results/metrics.csv",
    index=False
)
