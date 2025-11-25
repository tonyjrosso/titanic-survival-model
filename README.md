# Titanic Survival Prediction – Modular ML Pipeline

A clean, production-style machine learning project demonstrating:

- modular project architecture
- reproducible training pipeline
- model registry
- metrics computation
- visualizations (with/without ground truth)
- clean model retraining workflow
- dataset loading and preprocessing

This project uses the Titanic dataset to build a logistic regression classifier predicting passenger survival.

## Project Structure

```
titanic_survival_model/
│
├── model/
│   ├── inference.py          # predict() and predict_with_report()
│   ├── training.py           # train() with CV, test split, metrics, saving
│   ├── registry.py           # load/save model artifacts + metadata
│   └── artifacts/            # serialized models + metadata files
│
├── metrics/
│   ├── evaluation.py         # compute_metrics() producing structured metrics
│   ├── monitoring.py         # collects performance history over model versions
│   ├── visualization.py      # score/ROC/PR/confusion plots
│   └── schema.py             # schemas for metrics
│
├── data/
│   └── loader.py       # loads + preprocesses Titanic dataset
│
├── config/
│   ├── settings.yaml         # default training + path configs
│   └── load_config.py        # loads configs for other modules
│
├── notebooks/
│   └── Titanic_Demo.ipynb    # walkthrough for training + evaluation
│
└── requirements.txt
```

## Example Outputs

The notebook walks through:

- training a logistic regression model
- computing metrics
- visualizing ROC/PR/CM
- monitoring score distribution
- evaluating performance history over retraining cycles

Example figures (ROC curve, PR curve, score distribution etc.) are produced using `metrics/visualization.py`.

## How to Run
1. Install dependencies
```
pip install -r requirements.txt
```

2. Run the demo notebook
```
jupyter notebook notebooks/Titanic_Demo.ipynb
```

3. Or train a new model programmatically
```
from model.training import train
from data.loader import load_titanic
from sklearn.linear_model import LogisticRegression

X, y = load_titanic()
model, metrics = train(LogisticRegression(), X, y)
```

## Configuration

Edit config/settings.yaml to change:

- train/test split
- random seed
- artifact directory

This keeps your code clean and your experiments reproducible.

## Model Artifacts

Trained models are saved under `model/artifacts/`

## Acknowledgements

Dataset from:
> https://github.com/datasciencedojo/datasets/blob/master/titanic.csv
