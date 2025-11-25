from titanic_survival_model.metrics.monitoring import collect_performance_history

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


# -----------------------------------------------------------------------------
# Utility: Convert a dict-like metrics block into a table format for display
# -----------------------------------------------------------------------------

def _dict_to_table(data_dict):
    """Convert dict items into table-friendly row format."""
    return [[k.replace('_', ' ').title(), v] for k, v in data_dict.items()]


# -----------------------------------------------------------------------------
# Section 1 — VISUALIZATION WITHOUT GROUND TRUTH
# (Score-only performance visualization)
# -----------------------------------------------------------------------------

def plot_score_overview(prediction):
    """
    Plot:
      - numeric prediction score statistics (table)
      - prediction score distribution (histogram)
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # ---- Left: numeric metrics table ----
    summary_dict = {
        "Batch Size": prediction.metrics.input.batch_size,
        "Positive Rate": round(prediction.metrics.prediction.positive_rate, 3),
        "Mean Score": round(prediction.metrics.prediction.mean_score, 3),
        "Score Std": round(prediction.metrics.prediction.score_std, 3),
        "Min Score": round(prediction.metrics.prediction.min_score, 3),
        "Max Score": round(prediction.metrics.prediction.max_score, 3),
    }

    table_data = _dict_to_table(summary_dict)
    ax[0].axis("off")
    ax[0].table(cellText=table_data, loc="center")
    ax[0].set_title("Prediction Score Statistics")

    # ---- Right: histogram of score distribution ----
    ax[1].hist(prediction.scores, bins=20, color='steelblue', alpha=0.8)
    ax[1].set_title("Prediction Score Distribution")
    ax[1].set_xlabel("Score")
    ax[1].set_ylabel("Frequency")

    fig.tight_layout()
    plt.close(fig)
    return fig


# -----------------------------------------------------------------------------
# Section 2 — VISUALIZATION WITH GROUND TRUTH
# (Performance metrics, ROC, PR, Confusion Matrix)
# -----------------------------------------------------------------------------

def plot_metric_cards(metrics):
    """
    Display key supervised metrics (ROC AUC, PR AUC, F1, Precision, Recall,
    Log Loss) in a numeric summary table.
    """
    perf = metrics.performance

    summary_dict = {
        "Precision": round(perf.precision, 3),
        "Recall": round(perf.recall, 3),
        "F1 Score": round(perf.f1, 3),
        "ROC AUC": round(perf.roc_auc, 3),
        "PR AUC": round(perf.pr_auc, 3),
        "Log Loss": round(perf.logloss, 3),
    }

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    ax.table(cellText=_dict_to_table(summary_dict), loc="center")
    ax.set_title("Performance Metrics Summary")

    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_confusion_matrix(metrics):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(4, 4))
    cm = np.array(metrics.performance.confusion_matrix)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_roc_curve(metrics):
    """Plot ROC curve."""
    fpr, tpr, _ = metrics.performance.roc_curve
    auc = metrics.performance.roc_auc

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_pr_curve(metrics):
    """Plot Precision–Recall curve."""
    precision, recall, _ = metrics.performance.pr_curve
    auc = metrics.performance.pr_auc

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, lw=2, color="seagreen")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve (AUC={auc:.3f})")
    fig.tight_layout()
    plt.close(fig)
    return fig


# -----------------------------------------------------------------------------
# Section 3 — VISUALIZATION WITH GROUND TRUTH
# (Performance metrics, ROC, PR, Confusion Matrix)
# -----------------------------------------------------------------------------

def plot_roc_auc_history(history):
    """Plot history of ROC AUC across models."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for model in history:
        ax.scatter(model.index, model.roc_auc, color="seagreen")

    ax.set_xlabel("Model ID")
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"History of ROC AUC across models")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_pr_auc_history(history):
    """Plot history of PR AUC across models."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for model in history:
        ax.scatter(model.index, model.pr_auc, color="seagreen")

    ax.set_xlabel("Model ID")
    ax.set_ylabel("PR AUC")
    ax.set_title(f"History of PR AUC across models")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_f1_history(history):
    """Plot history of F1 score across models."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for model in history:
        ax.scatter(model.index, model.f1, color="seagreen")

    ax.set_xlabel("Model ID")
    ax.set_ylabel("F1 score")
    ax.set_title(f"History of F1 score across models")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_log_loss_history(history):
    """Plot history of log loss across models."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for model in history:
        ax.scatter(model.index, model.log_loss, color="seagreen")

    ax.set_xlabel("Model ID")
    ax.set_ylabel("Log loss")
    ax.set_title(f"History of log loss across models")
    fig.tight_layout()
    plt.close(fig)
    return fig


# -----------------------------------------------------------------------------
# Section 4 — MASTER FUNCTIONS
# -----------------------------------------------------------------------------

def visualize_without_ground_truth(prediction):
    """
    Returns a figure object that visualizes model output without labels.
    """
    return plot_score_overview(prediction)


def visualize_with_ground_truth(prediction):
    """
    Returns a list of figure objects for:
      - score overview
      - metric summary
      - confusion matrix
      - PR curve
      - ROC curve
    """
    figs = []
    figs.append(plot_score_overview(prediction))
    figs.append(plot_metric_cards(prediction.metrics))
    figs.append(plot_confusion_matrix(prediction.metrics))
    figs.append(plot_pr_curve(prediction.metrics))
    figs.append(plot_roc_curve(prediction.metrics))
    return figs


def visualize_performance_history(history):
    """
    Returns a list of figure objects for:
      - history of ROC AUC
      - history of PR AUC
      - history of F1 score
      - history of log loss
    for all models that have been trained.
    """
    figs = []
    figs.append(plot_roc_auc_history(history))
    figs.append(plot_pr_auc_history(history))
    figs.append(plot_f1_history(history))
    figs.append(plot_log_loss_history(history))
    return figs