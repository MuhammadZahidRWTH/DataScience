import os
import json
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "data/transactions.csv"
MODEL_DIR = "model"
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "fraud-detection"

PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 49,
    "eval_metric": "aucpr",
    "use_label_encoder": False,
    "random_state": 42,
}


def load_and_split(path):
    df = pd.read_csv(path)
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def build_pipeline(params):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(**params)),
    ])


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
    }


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH)
    print(f"train: {len(X_train)} rows | test: {len(X_test)} rows | fraud rate: {y_train.mean():.2%}")

    with mlflow.start_run(run_name="xgboost-fraud-v1") as run:

        mlflow.log_params(PARAMS)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("fraud_rate", round(float(y_train.mean()), 4))

        pipeline = build_pipeline(PARAMS)
        pipeline.fit(X_train, y_train)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        mlflow.log_metric("cv_roc_auc_mean", round(cv_scores.mean(), 4))
        mlflow.log_metric("cv_roc_auc_std", round(cv_scores.std(), 4))
        print(f"cv roc-auc: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)

        print("\n── test metrics ─────────────────────────")
        for k, v in metrics.items():
            print(f"  {k:<20} {v}")

        if metrics["recall"] < 0.5:
            print("\nwarning: recall below 0.5 — consider retuning scale_pos_weight")

        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=["legit", "fraud"])
        report_path = f"{MODEL_DIR}/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        feature_names = X_train.columns.tolist()
        importances = pipeline.named_steps["model"].feature_importances_
        fi = dict(sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1], reverse=True
        ))
        fi_path = f"{MODEL_DIR}/feature_importance.json"
        with open(fi_path, "w") as f:
            json.dump(fi, f, indent=2)
        mlflow.log_artifact(fi_path)

        model_path = f"{MODEL_DIR}/pipeline.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        metadata = {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "params": PARAMS,
        }
        with open(f"{MODEL_DIR}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nrun id: {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    train()