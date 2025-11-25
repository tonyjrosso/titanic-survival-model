from titanic_survival_model.metrics.schema import (
    Metrics,
    InputMetrics,
    PredictionMetrics,
    PerformanceMetrics,
)

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def compute_metrics(y_pred, y_score, y_truth=None):

    # --- Input metrics ---
    input_metrics = InputMetrics(
        batch_size=len(y_pred),
        positive_count=int(y_truth.sum()) if y_truth is not None else None,
        negative_count=int((y_truth == 0).sum()) if y_truth is not None else None,
        positive_rate=float(y_truth.mean()) if y_truth is not None else None,
    )

    # --- Prediction metrics ---
    prediction_metrics = PredictionMetrics(
        positive_rate=float(y_pred.mean()),
        positive_count=int(y_pred.sum()),
        negative_count=int((y_pred == 0).sum()),
        min_score=float(y_score.min()),
        mean_score=float(y_score.mean()),
        max_score=float(y_score.max()),
        score_std=float(y_score.std())
    )

    # --- Performance metrics ---
    if y_truth is not None:
        # compute supervised metrics...
        perf = PerformanceMetrics(
            confusion_matrix=confusion_matrix(y_truth, y_pred).tolist(),
            accuracy=accuracy_score(y_truth, y_pred),
            roc_auc=roc_auc_score(y_truth, y_score),
            pr_auc=average_precision_score(y_truth, y_score),
            precision=precision_score(y_truth, y_pred),
            recall=recall_score(y_truth, y_pred),
            f1=f1_score(y_truth, y_pred),
            logloss=log_loss(y_truth, y_score),
            roc_curve=[x.tolist() for x in roc_curve(y_truth, y_score)],
            pr_curve=[x.tolist() for x in precision_recall_curve(y_truth, y_score)]
        )
    else:
        perf = PerformanceMetrics(
            confusion_matrix=None, accuracy=None, roc_auc=None, pr_auc=None,
            precision=None, recall=None, f1=None, logloss=None,
            roc_curve=None, pr_curve=None
        )

    # --- Build final metrics object ---
    return Metrics(
        input=input_metrics,
        prediction=prediction_metrics,
        performance=perf
    )