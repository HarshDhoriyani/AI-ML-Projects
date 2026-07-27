"""Evaluate a trained pipeline on the held-out Adult Census Income test set."""

import logging

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src import config
from src.data_loader import load_test_data
from src.preprocessing import split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(save_plots: bool = True):
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. Run `python main.py train` first."
        )

    pipeline = joblib.load(config.MODEL_PATH)

    logger.info("Loading test data...")
    df = load_test_data()
    X_test, y_test = split_features_target(df)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])
    auc = roc_auc_score(y_test, y_proba)

    print("\n=== Classification Report ===")
    print(report)
    print(f"ROC-AUC: {auc:.4f}")

    if save_plots:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"], ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(config.MODELS_DIR / "confusion_matrix.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title("ROC Curve")
        fig.tight_layout()
        fig.savefig(config.MODELS_DIR / "roc_curve.png", dpi=150)
        plt.close(fig)

        logger.info("Saved plots to %s", config.MODELS_DIR)

    return {"classification_report": report, "roc_auc": auc}


if __name__ == "__main__":
    evaluate_model()
