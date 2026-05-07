
from pydantic import BaseModel


class TrainMetrics(BaseModel):
    MAE_pct: float
    RMSE_pct: float
    MAPE: float
    DA: float
    DA_naive: float
    DA_conf: float
    delta_DA: float


class ModelMetricsResponse(BaseModel):
    model_version: str
    ticker: str
    target: str
    architecture: str
    framework: str
    window_size: int
    n_features: int
    features: list[str]
    normalization: str
    confidence_threshold: float
    train_metrics: TrainMetrics