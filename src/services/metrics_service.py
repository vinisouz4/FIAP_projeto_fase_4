from typing import List, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: List[float],
    y_pred: List[float],
) -> dict:
    """
    Calcula MAE, MSE, RMSE, MAPE e R² entre valores reais e previstos.

    Retorna dict com as métricas (None quando não aplicável).
    """
    yt = np.array(y_true, dtype=np.float64)
    yp = np.array(y_pred, dtype=np.float64)

    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]

    mae  = float(mean_absolute_error(yt, yp))
    mse  = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100)
    r2: Optional[float] = float(r2_score(yt, yp)) if n > 1 else None

    return {
        "mae":  round(mae,  6),
        "mse":  round(mse,  6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 4),
        "r2":   round(r2, 6) if r2 is not None else None,
    }