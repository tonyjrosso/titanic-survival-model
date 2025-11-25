import os
import yaml


CONFIG_PATH = os.path.join("titanic_survival_model","config", "settings.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)