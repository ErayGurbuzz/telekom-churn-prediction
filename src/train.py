import json
from pathlib import Path

import joblib
import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- PROJE DIZIN YAPILANDIRMASI ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges sütunundaki geçersiz değerleri sayıya çevirip NaN yapar
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Hedef değişkeni binary formata (0-1) çevirir
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Sayısal ve kategorik veriler için ön işleme hattını (pipeline) kurar."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Sayısal veriler: Eksik tamamlama (median) ve standartlaştırma
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Kategorik veriler: Eksik tamamlama (en sık tekrar eden) ve One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Modelin başarısını ölçer ve tüm metrikleri sözlük olarak döner."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "classification_report": classification_report( #Scikit-learn bunu sözlük olarak da döndürebiliyor. JSON’a yazmak ve sonradan raporda kullanmak daha kolay olur.
            y_test, y_pred, zero_division=0, output_dict=True
        ),
    }
    return metrics

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    
    # 1. Veri Yükleme ve Bölme
    df = load_data(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Stratify kullanımı ile sınıfların dengeli dağılmasını sağlıyoruz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = build_preprocessor(X)

    # 2. Sınıf Ağırlıklarını Hesaplama (Dengesiz veri çözümü için)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos

    # 3. Model Listesi
    candidate_models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=-1
        ),
        "catboost": CatBoostClassifier(
            auto_class_weights="Balanced",
            random_state=42,
            verbose=0
        ),
    }
    
    # 4. Eğitim ve Yarıştırma 
    results = []
    best_name, best_pipeline, best_f1 = None, None, -1.0

    for name, model in candidate_models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)

        # Şampiyon modeli F1-Score bazında seçiyoruz
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name = name
            best_pipeline = pipeline

    # 5. Kayıt İşlemleri
    joblib.dump(best_pipeline, MODEL_DIR / "churn_pipeline.pkl")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"Eğitim tamamlandı. Şampiyon Model: {best_name.upper()}")

if __name__ == "__main__":
    main()