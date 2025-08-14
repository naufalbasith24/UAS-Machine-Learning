# forestfires_ml.py
# Reproducible pipeline for UCI Forest Fires dataset
import json, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "forestfires.csv")
OUT = os.path.join(HERE, "outputs")
PLOTS = os.path.join(OUT, "plots")

os.makedirs(OUT, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

df = pd.read_csv(DATA)
df['fire_occurred'] = (df['area'] > 0).astype(int)
df['log_area'] = np.log1p(df['area'])

numeric_cols = [c for c in df.columns if c not in ['month','day','area','log_area','fire_occurred'] and pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in ['month','day'] if c in df.columns]

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
preprocessor = ColumnTransformer([("num", numeric_transformer, numeric_cols),("cat", categorical_transformer, categorical_cols)])

# Regression
X_reg = df[numeric_cols + categorical_cols]
y_reg = df['log_area']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_model = Pipeline([("preprocess", preprocessor), ("model", RandomForestRegressor(n_estimators=300, random_state=42))])
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)
rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred_r)))
r2 = float(r2_score(y_test_r, y_pred_r))
baseline_rmse = float(np.sqrt(mean_squared_error(y_test_r, np.full_like(y_test_r, y_train_r.mean()))))
with open(os.path.join(OUT, "regression_metrics.json"), "w") as f:
    json.dump({"rmse_log": rmse, "r2": r2, "baseline_rmse_log": baseline_rmse}, f, indent=2)

# Classification
X_clf = df[numeric_cols + categorical_cols]
y_clf = df['fire_occurred']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
clf_model = Pipeline([("preprocess", preprocessor), ("model", RandomForestClassifier(n_estimators=400, random_state=42))])
clf_model.fit(X_train_c, y_train_c)
y_pred_c = clf_model.predict(X_test_c)
acc = float(accuracy_score(y_test_c, y_pred_c))
prec = float(precision_score(y_test_c, y_pred_c, zero_division=0))
rec = float(recall_score(y_test_c, y_pred_c))
f1 = float(f1_score(y_test_c, y_pred_c))
cm = confusion_matrix(y_test_c, y_pred_c).tolist()
with open(os.path.join(OUT, "classification_metrics.json"), "w") as f:
    json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}, f, indent=2)

# Plots
plt.figure(); df['area'].plot(kind='hist', bins=40, title='Distribution of Burned Area (ha)'); plt.xlabel('area (ha)'); plt.ylabel('count'); plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'area_hist.png'), dpi=160); plt.close()
plt.figure(); df['log_area'].plot(kind='hist', bins=40, title='Distribution of log1p(Area)'); plt.xlabel('log1p(area)'); plt.ylabel('count'); plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'log_area_hist.png'), dpi=160); plt.close()

# Confusion matrix plot
import numpy as np
cm_arr = np.array(cm)
plt.figure(); plt.imshow(cm_arr, interpolation='nearest'); plt.title('Confusion Matrix (fire_occurred)'); plt.xlabel('Predicted'); plt.ylabel('Actual')
for (i, j), v in np.ndenumerate(cm_arr):
    plt.text(j, i, str(v), ha='center', va='center')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'confusion_matrix.png'), dpi=160); plt.close()

with open(os.path.join(OUT, "report.txt"), "w") as f:
    f.write(f"RMSE_log={rmse:.4f}, Baseline_RMSE_log={baseline_rmse:.4f}, R2={r2:.4f}\nAccuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}\nConfusion Matrix: {cm}")
print("Done. Outputs saved under 'outputs/'.")