from titanic_survival_model.config.load_config import CONFIG
from titanic_survival_model.metrics.evaluation import compute_metrics
from titanic_survival_model.model.registry import save_model

from sklearn.model_selection import train_test_split

def train(model, X, y, config=None):
    """Train a new model."""

    # Pull defaults from YAML if config not supplied
    if config is None:
        config = CONFIG["training"]
    else:
        # fill in missing keys with defaults
        defaults = CONFIG["training"]
        config = {**defaults, **config}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    model.fit(X_train, y_train)

    metrics = compute_metrics(
        model.predict(X_test),
        model.predict_proba(X_test)[:, 1],
        y_test
    )
    
    save_model(model, metadata=metrics)

    return model, metrics
