"""
services/predictor.py

Lógica de inferência: prepara o input, roda o modelo e retorna previsões.
Suporta previsão de múltiplos passos (autoregressive / recursive forecasting).
"""

import time
from typing import List, Optional

import numpy as np

from src.services import model_loader
from src.services.metrics_service import compute_metrics


def _window_size() -> int:
    """Retorna o número de timesteps esperado pelo modelo."""
    model = model_loader.get_model()
    shape = model.input_shape  # ex: (None, 10, 1) para LSTM univariado
    if len(shape) == 3:
        return shape[1]
    return int(shape[-1])


def _prepare_input(prices: List[float]) -> np.ndarray:
    """
    Ajusta a série para o tamanho da janela do modelo.
    Retorna array com shape (1, timesteps, 1).
    """
    window = _window_size()
    arr = np.array(prices, dtype=np.float32)

    # Trunca ou faz padding à esquerda (edge padding)
    if len(arr) > window:
        arr = arr[-window:]
    elif len(arr) < window:
        arr = np.pad(arr, (window - len(arr), 0), mode="edge")

    return arr.reshape(1, window, 1)


def run_prediction(
    prices: List[float],
    steps: int = 1,
    actual_prices: Optional[List[float]] = None,
) -> dict:
    """
    Executa a previsão para `steps` passos futuros de forma autoregressiva:
    cada previsão é adicionada à janela para alimentar o próximo passo.

    Parâmetros
    ----------
    prices: série histórica de preços (janela de entrada)
    steps: quantos períodos futuros prever
    actual_prices: ground-truth opcional para calcular métricas

    Retorna
    -------
    dict com predictions, steps, inference_time_ms e metrics (opcional)
    """
    model = model_loader.get_model()
    buffer = list(prices)
    predictions: List[float] = []

    t0 = time.perf_counter()

    for _ in range(steps):
        x = _prepare_input(buffer)
        raw = model.predict(x, verbose=0)        # (1, 1) ou (1,)
        pred = round(float(raw.flatten()[0]), 6)
        predictions.append(pred)
        buffer.append(pred)                       # alimenta próximo passo

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    result: dict = {
        "predictions": predictions,
        "steps": steps,
        "inference_time_ms": elapsed_ms,
        "metrics": None,
    }

    if actual_prices:
        result["metrics"] = compute_metrics(actual_prices, predictions)

    return result