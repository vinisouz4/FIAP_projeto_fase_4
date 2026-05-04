"""
Rota de predição — /predict
Compatível com o modelo BiLSTM treinado em train_models.ipynb
"""


import numpy as np
import pandas as pd

from src.models.predict_models import DailyOHLCV




# ---------------------------------------------------------------------------
# Helpers — espelham exatamente o pré-processamento do notebook
# ---------------------------------------------------------------------------

def _build_features(history: list[DailyOHLCV]) -> np.ndarray:
    """
    Recria as mesmas 13 features do notebook a partir do histórico OHLCV.
    Retorna array (n_valid_rows, 13) sem NaNs.
    """

    df = pd.DataFrame([d.model_dump() for d in history])

    df["RETURN"] = df["close"].pct_change()
    df["RETURN_2"] = df["close"].pct_change(2)
    df["RETURN_5"] = df["close"].pct_change(5)
    df["LOG_RETURN"] = np.log(df["close"] / df["close"].shift(1))
    df["HIGH_LOW_PCT"] = (df["high"] - df["low"]) / df["close"]
    df["OPEN_CLOSE"] = (df["close"] - df["open"]) / df["open"]
    df["VOLATILITY_5"] = df["RETURN"].rolling(5).std()
    df["VOLATILITY_20"] = df["RETURN"].rolling(20).std()
    df["VOL_CHANGE"] = df["volume"].pct_change()
    df["MOMENTUM_5"] = df["RETURN"].rolling(5).sum()
    df["MOMENTUM_10"] = df["RETURN"].rolling(10).sum()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    features = [
        "RETURN", "RETURN_2", "RETURN_5", "LOG_RETURN",
        "HIGH_LOW_PCT", "OPEN_CLOSE",
        "VOLATILITY_5", "VOLATILITY_20",
        "VOL_CHANGE", "MOMENTUM_5", "MOMENTUM_10",
        "RSI", "MACD_HIST",
    ]
    df = df[features].dropna().reset_index(drop=True)
    return df.values.astype(np.float32)


def _window_normalize(window: np.ndarray) -> np.ndarray:
    """Z-score por janela — idêntico ao create_sequences_window_norm do notebook."""
    std = window.std(axis=0)
    std[std < 1e-8] = 1.0
    return (window - window.mean(axis=0)) / std


def _build_input(feature_matrix: np.ndarray, window_size: int) -> np.ndarray:
    """
    Pega os últimos `window_size` passos, normaliza e retorna
    o tensor (1, window_size, n_features) pronto para inferência.
    """
    if len(feature_matrix) < window_size:
        raise ValueError(
            f"Dados insuficientes após cálculo de features. "
            f"Necessário >= {window_size} linhas válidas, obteve {len(feature_matrix)}."
        )
    window = feature_matrix[-window_size:].copy()
    window_norm = _window_normalize(window)
    return window_norm[np.newaxis, ...]  # (1, 30, 13)


