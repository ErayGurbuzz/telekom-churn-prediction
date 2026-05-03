import matplotlib
matplotlib.use("Agg")  # Grafik penceresi açılmasını engeller

import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

# --- PATH AYARLARI ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.pkl"
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Evaluate başladı")

# --- DATA ---
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"]

# --- MODEL ---
pipeline = joblib.load(MODEL_PATH)
print("Model yüklendi:", MODEL_PATH)

y_pred = pipeline.predict(X)
y_proba = pipeline.predict_proba(X)[:, 1]

# =========================
# 1. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix.png")
plt.close()

print("Confusion matrix kaydedildi")

# =========================
# 2. ROC CURVE
# =========================
RocCurveDisplay.from_predictions(y, y_proba)

plt.title("ROC Curve")
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_curve.png")
plt.close()

print("ROC curve kaydedildi")

# =========================
# 3. PRECISION-RECALL CURVE
# =========================
PrecisionRecallDisplay.from_predictions(y, y_proba)

plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(FIG_DIR / "precision_recall_curve.png")
plt.close()

print("Precision-Recall curve kaydedildi")

# =========================
# 4. FEATURE IMPORTANCE
# =========================
print("Feature importance başladı")

model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessor"]

print("Kullanılan model:", type(model))

feature_names = preprocessor.get_feature_names_out()

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    importance_type = "tree_feature_importance"
elif hasattr(model, "coef_"):
    importances = abs(model.coef_[0])
    importance_type = "logistic_regression_coefficients"
else:
    raise ValueError("Bu model feature importance veya coef_ desteklemiyor.")

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

top_features = feat_imp.head(15)

print("Importance tipi:", importance_type)
print(top_features)

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()

plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")

plt.tight_layout()
plt.savefig(FIG_DIR / "feature_importance.png")
plt.close()

print("Feature importance kaydedildi")
print("Evaluate tamamlandı")

# =========================
# 5. MODEL COMPARISON
# =========================


METRICS_PATH = BASE_DIR / "models" / "metrics.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# JSON oku
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics_data = json.load(f)

results = pd.DataFrame(metrics_data["results"])

# =========================
# 📊 TEMİZ MODEL KARŞILAŞTIRMA
# =========================
plt.figure(figsize=(12, 6))

bar_width = 0.15
x = range(len(results))

metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

for i, metric in enumerate(metrics):
    plt.bar(
        [p + i * bar_width for p in x],
        results[metric],
        width=bar_width,
        label=metric
    )

    
    for j, val in enumerate(results[metric]):
        plt.text(
            j + i * bar_width, 
            val + 0.01, 
            f"{val:.3f}", 
            ha='center', 
            fontsize=8
        )

plt.xticks(
    [p + bar_width*2 for p in x],
    results["model_name"],
    rotation=30,
    ha="right"
)

plt.ylim(0, 1)
plt.title("Model Performance Comparison")

plt.ylabel("Score")
plt.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison_clean.png")
plt.close()

print("Temiz model karşılaştırma grafiği kaydedildi")