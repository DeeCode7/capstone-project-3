"""Training, evaluation, quality gating, and model serialization."""

import json
import logging
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.config import (
    ALL_FEATURES,
    BEST_MODEL_PATH,
    METRICS_PATH,
    MLP_MODEL_PATH,
    MLP_PARAMS,
    MODELS_DIR,
    QUALITY_GATE_MACRO_F1,
    RF_MODEL_PATH,
    RF_PARAMS,
    TARGET_COLUMN,
)
from src.data_loader import load_datasets
from src.preprocessor import build_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_pipelines() -> dict[str, Pipeline]:
    """Instantiate end-to-end ML and DL pipelines."""
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(**RF_PARAMS)),
        ]
    )

    mlp_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", MLPClassifier(**MLP_PARAMS)),
        ]
    )

    return {
        "RandomForest (ML)": rf_pipeline,
        "MLP Neural Net (DL)": mlp_pipeline,
    }


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict[str, Any]:
    """Compute comprehensive multi-class evaluation metrics."""
    y_pred = pipeline.predict(X_test)
    class_names = list(label_encoder.classes_)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": report,
        "confusion_matrix": cm,
        "labels": class_names,
    }


def train_and_evaluate() -> dict[str, Any]:
    """Run full training workflow, enforce acceptance gate, and persist artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df, data_source = load_datasets()
    X_train = train_df[ALL_FEATURES]
    X_test = test_df[ALL_FEATURES]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[TARGET_COLUMN])
    y_test = label_encoder.transform(test_df[TARGET_COLUMN])

    logger.info(
        "Dataset loaded (%s) | Train rows: %d | Test rows: %d | Classes: %s",
        data_source,
        len(train_df),
        len(test_df),
        list(label_encoder.classes_),
    )

    pipelines = build_pipelines()
    results: dict[str, Any] = {
        "data_source": data_source,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "models": {},
    }

    best_model_name = ""
    best_macro_f1 = -1.0
    best_pipeline: Pipeline | None = None

    for name, pipeline in pipelines.items():
        logger.info("Training %s...", name)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test, label_encoder)
        results["models"][name] = metrics

        logger.info(
            "%s -> Accuracy: %.4f | Macro-F1: %.4f | Weighted-F1: %.4f",
            name,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
        )

        target_path = RF_MODEL_PATH if "RandomForest" in name else MLP_MODEL_PATH
        joblib.dump(
            {
                "model_name": name,
                "pipeline": pipeline,
                "label_encoder": label_encoder,
                "metrics": metrics,
            },
            target_path,
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_model_name = name
            best_pipeline = pipeline

    passed_gate = best_macro_f1 >= QUALITY_GATE_MACRO_F1
    results["best_model"] = best_model_name
    results["best_macro_f1"] = best_macro_f1
    results["quality_gate_threshold"] = QUALITY_GATE_MACRO_F1
    results["passed_quality_gate"] = passed_gate

    if not passed_gate:
        logger.warning(
            "Quality Gate Warning: Best Macro-F1 (%.4f) is below threshold (%.2f).",
            best_macro_f1,
            QUALITY_GATE_MACRO_F1,
        )
    else:
        logger.info(
            "Quality Gate Passed! Promoting %s (Macro-F1: %.4f) to best_model.joblib",
            best_model_name,
            best_macro_f1,
        )

    joblib.dump(
        {
            "model_name": best_model_name,
            "pipeline": best_pipeline,
            "label_encoder": label_encoder,
            "metrics": results["models"][best_model_name],
        },
        BEST_MODEL_PATH,
    )

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_and_evaluate()
