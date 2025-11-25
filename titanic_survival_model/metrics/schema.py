from dataclasses import dataclass
from typing import List, Optional

@dataclass
class InputMetrics:
    batch_size: int
    positive_count: Optional[int]
    negative_count: Optional[int]
    positive_rate: Optional[float]

@dataclass
class PredictionMetrics:
    positive_rate: float
    positive_count: int
    negative_count: int
    min_score: float
    mean_score: float
    max_score: float
    score_std: float

@dataclass
class PerformanceMetrics:
    confusion_matrix: Optional[List[List[int]]]
    accuracy: Optional[float]
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    logloss: Optional[float]
    roc_curve: Optional[List[List[float]]]
    pr_curve: Optional[List[List[float]]]

@dataclass
class Metrics:
    input: InputMetrics
    prediction: PredictionMetrics
    performance: PerformanceMetrics

@dataclass
class PerformanceHistory:
    index: int
    roc_auc: float
    pr_auc: float
    f1: float
    log_loss: float


@dataclass
class Prediction:
    metrics: Metrics
    predictions: List[int]
    scores: List[float]