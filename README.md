# Forest Fires ML Project

This repository analyzes the **forestfires.csv** dataset and provides two tasks:
1. **Regression**: predict `log1p(area)` (stabilizes variance for skewed target).
2. **Classification**: predict `fire_occurred` (whether any area burned).

## Why these targets & algorithms?
- **log1p(area)** makes errors more proportional and reduces the impact of extreme outliers.
- **Random Forests** handle non-linearities, interactions, and mixed feature types well with minimal tuning.
- Baselines and metrics are saved for transparency.

## Folder Structure
```
forestfires-ml-project/
├── data/
│   └── forestfires.csv
├── scripts/
│   └── forestfires_ml.py
└── outputs/
    ├── eda_summary.json
    ├── regression_metrics.json
    ├── classification_metrics.json
    ├── report.txt
    └── plots/
        ├── area_hist.png
        ├── log_area_hist.png
        └── confusion_matrix.png
```

## How to run locally
```bash
pip install -r requirements.txt
python scripts/forestfires_ml.py
```

## Quick EDA
- Shape, columns, dtypes, and null counts are saved in `outputs/eda_summary.json`.

## Metrics
- **Regression**: RMSE (log-scale), R², baseline RMSE (mean-predictor).
- **Classification**: Accuracy, Precision, Recall, F1, Confusion Matrix.

## Push to GitHub
1. Create a new GitHub repo (empty).
2. In a terminal:
```bash
cd forestfires-ml-project
git init
git add .
git commit -m "Initial commit: forest fires EDA + models"
git branch -M main
git remote add origin https://github.com/<USERNAME>/<REPO>.git
git push -u origin main
```
3. Add a link to the repo in your report if needed.

## Notes
- If your dataset columns differ from UCI names, adjust `numeric_cols`/`categorical_cols` in the script.