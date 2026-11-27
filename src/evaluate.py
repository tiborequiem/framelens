"""Metrics for binary news-language bias classification."""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred):
    """Return accuracy, macro F1 and metrics for both classes (0 and 1)."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "precision_non_biased": float(precision[0]),
        "recall_non_biased": float(recall[0]),
        "f1_non_biased": float(f1[0]),
        "precision_biased": float(precision[1]),
        "recall_biased": float(recall[1]),
        "f1_biased": float(f1[1]),
    }
