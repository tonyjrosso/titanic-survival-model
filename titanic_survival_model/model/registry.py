from dataclasses import asdict
import joblib
import json
import os

from titanic_survival_model.config.load_config import CONFIG

artifact_dir = CONFIG["paths"]["artifact_dir"]

def _get_next_model_id():
    """Return the next unused sequential model ID."""
    files = os.listdir(artifact_dir)
    ids = [
        int(f.removeprefix("model_id_").removesuffix(".joblib"))
        for f in files
        if f.startswith("model_id_") and f.endswith(".joblib")
    ]
    return max(ids) + 1 if ids else 0


def save_model(model, metadata):
    """Save the model and return the assigned model ID."""
    model_id = _get_next_model_id()

    # Save model
    joblib.dump(model, f"{artifact_dir}/model_id_{model_id}.joblib")

    # Save metadata (metrics, config, dataset snapshot, etc.)
    with open(f"{artifact_dir}/model_id_{model_id}_info.json", "w") as f:
        json.dump(asdict(metadata), f, indent=4)

    return model_id

def load_model(model_id):
    """Returns the model with corresponding model ID."""
    return joblib.load(f"{artifact_dir}/model_id_{model_id}.joblib")