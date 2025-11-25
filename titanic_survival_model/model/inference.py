from titanic_survival_model.metrics.evaluation import compute_metrics
from titanic_survival_model.metrics.schema import Prediction

def predict(model, X, y=None):
    """
    Return the model prediction, scores, and, if ground truth has been provided,
    performance metrics.
    """
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:,1]
    metrics = compute_metrics(
        y_pred,
        y_score,
        y
    )
    return Prediction(
        metrics= metrics,
        predictions= y_pred,
        scores= y_score
    )