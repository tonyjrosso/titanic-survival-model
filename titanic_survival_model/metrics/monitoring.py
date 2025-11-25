import os
import json

from titanic_survival_model.config.load_config import CONFIG
from titanic_survival_model.metrics.schema import PerformanceHistory

artifact_dir = CONFIG["paths"]["artifact_dir"]

def collect_performance_history():
    """Load performance metrics from all available models."""

    history = []
    
    for file in os.listdir(artifact_dir):

        if not (file.startswith("model_id_") and file.endswith("_info.json")):
            continue
        
        path = os.path.join(artifact_dir, file)

        try:
            with open(path, 'r') as f:
                model_info = json.load(f)
        except FileNotFoundError:
            print(f"Error: '{file}' not found. Please create the file.")
            continue
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{file}'.")
            continue
        
        history += [
            PerformanceHistory(
                index = int(
                    file.removeprefix("model_id_").removesuffix("_info.json")
                ),
                roc_auc = model_info["performance"]["roc_auc"],
                pr_auc = model_info["performance"]["pr_auc"],
                f1 = model_info["performance"]["f1"],
                log_loss = model_info["performance"]["logloss"]
            )
        ]
        
    return history