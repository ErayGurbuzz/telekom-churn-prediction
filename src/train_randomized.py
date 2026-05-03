import argparse
import json
import time
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --- PROJE DIZIN YAPILANDIRMASI ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"

# RandomizedSearchCV sonuçlarını ayrı klasörde tutuyoruz.
EXPERIMENT_DIR = MODEL_DIR / "experiments" / "randomized"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # TotalCharges sütunundaki geçersiz değerleri sayıya çevirip NaN yapar.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Hedef değişkeni binary formata çevirir.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # customerID model için anlamlı olmadığı için çıkarılır.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }

    return metrics


def get_search_spaces(scale_pos_weight: float) -> dict:
    """
    Her model için RandomizedSearchCV parametre alanlarını tanımlar.

    Not:
    Pipeline içinde model adımı "model" olduğu için parametreler
    model__parametre_adi şeklinde yazılır.
    """

    search_spaces = {
        "logistic_regression": {
            "model": LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
            "params": {
                "model__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
                "model__penalty": ["l1", "l2"],
            },
        },

        "random_forest": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
            "params": {
                "model__n_estimators": [200, 300, 500, 700],
                "model__max_depth": [5, 8, 10, 15, 20, None],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", None],
            },
        },

        "xgboost": {
            "model": XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=1,
            ),
            "params": {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "model__min_child_weight": [1, 3, 5, 7],
            },
        },

        "lightgbm": {
            "model": LGBMClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbosity=-1,
                n_jobs=1,
            ),
            "params": {
                "model__n_estimators": [100, 200, 300, 500],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__num_leaves": [15, 31, 50, 70],
                "model__max_depth": [-1, 5, 10, 15],
                "model__min_child_samples": [10, 20, 30, 50],
            },
        },

        "catboost": {
            "model": CatBoostClassifier(
                auto_class_weights="Balanced",
                random_state=42,
                verbose=0,
                allow_writing_files=False,
                thread_count=1,
            ),
            "params": {
                "model__iterations": [200, 300, 500],
                "model__depth": [4, 6, 8, 10],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__l2_leaf_reg": [1, 3, 5, 7, 9],
            },
        },
    }

    return search_spaces


def main(n_iter: int, cv_splits: int, promote: bool) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    start_time = time.time()

    # 1. Veri yükleme
    df = load_data(DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3. Ön işleme pipeline'ı
    preprocessor = build_preprocessor(X)

    # 4. Dengesiz veri için pozitif sınıf ağırlığı
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos

    # 5. Cross-validation yapısı
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42,
    )

    search_spaces = get_search_spaces(scale_pos_weight)

    results = []
    best_name = None
    best_pipeline = None
    best_f1 = -1.0

    # 6. Bütün modeller için RandomizedSearchCV
    for name, config in search_spaces.items():
        print(f"\n--- {name.upper()} için RandomizedSearchCV başladı ---")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", config["model"]),
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            n_iter=n_iter,
            scoring="f1",
            cv=cv,
            random_state=42,
            n_jobs=1,
            verbose=1,
            refit=True,
        )

        model_start_time = time.time()

        search.fit(X_train, y_train)

        model_elapsed_time = round(time.time() - model_start_time, 2)

        tuned_pipeline = search.best_estimator_

        metrics = evaluate_model(
            name=name,
            pipeline=tuned_pipeline,
            X_test=X_test,
            y_test=y_test,
        )

        metrics["best_params"] = search.best_params_
        metrics["best_cv_score"] = round(search.best_score_, 4)
        metrics["search_method"] = "RandomizedSearchCV"
        metrics["n_iter"] = n_iter
        metrics["cv_splits"] = cv_splits
        metrics["training_time_seconds"] = model_elapsed_time

        results.append(metrics)

        print(f"{name} bitti.")
        print(f"Best CV F1: {metrics['best_cv_score']}")
        print(f"Test F1: {metrics['f1_score']}")
        print(f"Süre: {model_elapsed_time} saniye")

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name = name
            best_pipeline = tuned_pipeline

    total_elapsed_time = round(time.time() - start_time, 2)

    output = {
        "search_method": "RandomizedSearchCV",
        "best_model": best_name,
        "best_test_f1_score": best_f1,
        "n_iter": n_iter,
        "cv_splits": cv_splits,
        "total_training_time_seconds": total_elapsed_time,
        "results": results,
    }

    # 7. Deney sonucunu ayrı klasöre kaydet
    experiment_model_path = EXPERIMENT_DIR / "churn_pipeline_randomized.pkl"
    experiment_metrics_path = EXPERIMENT_DIR / "metrics_randomized.json"

    joblib.dump(best_pipeline, experiment_model_path)

    with open(experiment_metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nRandomizedSearchCV tamamlandı.")
    print(f"En iyi model: {best_name.upper()}")
    print(f"En iyi test F1-score: {best_f1}")
    print(f"Toplam süre: {total_elapsed_time} saniye")
    print(f"Model kaydedildi: {experiment_model_path}")
    print(f"Metrikler kaydedildi: {experiment_metrics_path}")

    # 8. İsteğe bağlı olarak production modelini güncelle
    if promote:
        production_model_path = MODEL_DIR / "churn_pipeline.pkl"
        production_metrics_path = MODEL_DIR / "metrics.json"

        joblib.dump(best_pipeline, production_model_path)

        with open(production_metrics_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print("\nPromote işlemi yapıldı.")
        print("Yeni tuned model artık production modeli olarak kaydedildi.")
        print(f"Production model: {production_model_path}")
        print(f"Production metrics: {production_metrics_path}")
    else:
        print("\nProduction modeli değiştirilmedi.")
        print("Mevcut models/churn_pipeline.pkl aynı kaldı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Her model için denenecek rastgele parametre kombinasyonu sayısı.",
    )

    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Cross-validation fold sayısı.",
    )

    parser.add_argument(
        "--promote",
        action="store_true",
        help="En iyi tuned modeli models/churn_pipeline.pkl üzerine yazar.",
    )

    args = parser.parse_args()

    main(
        n_iter=args.n_iter,
        cv_splits=args.cv,
        promote=args.promote,
    )