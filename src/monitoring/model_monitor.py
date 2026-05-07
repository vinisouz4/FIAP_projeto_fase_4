"""
Monitoramento de modelo com MLflow
Rastreia drift de predições e directional accuracy em produção.
Registra cada predição em um run de produção separado do treino.
"""

import time
import threading
from collections import deque
from datetime import datetime, timezone

import mlflow

from src.log.logs import LoggerHandler
from src.services.model_loader_services import get_metadata

logger = LoggerHandler(__name__)

# ---------------------------------------------------------------------------
# Buffer em memória para acumular predições antes de logar no MLflow
# Evita uma chamada HTTP ao MLflow a cada request
# ---------------------------------------------------------------------------

_BUFFER_SIZE = 50          
_FLUSH_INTERVAL = 300      # flush a cada 5 minutos se o buffer não encher

_lock = threading.Lock()
_buffer: deque = deque(maxlen=_BUFFER_SIZE * 2)
_last_flush: float = time.time()


class PredictionRecord:
    """Representa uma predição registrada em produção."""

    __slots__ = (
        "timestamp", "ticker", "predicted_return",
        "direction", "confidence", "latency_ms",
    )

    def __init__(
        self,
        ticker: str,
        predicted_return: float,
        direction: str,
        confidence: str,
        latency_ms: float,
    ):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.ticker = ticker
        self.predicted_return = predicted_return
        self.direction = direction
        self.confidence = confidence
        self.latency_ms = latency_ms


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def log_prediction(record: PredictionRecord) -> None:
    """
    Adiciona uma predição ao buffer.
    O flush para o MLflow ocorre automaticamente quando o buffer atinge
    _BUFFER_SIZE ou após _FLUSH_INTERVAL segundos.
    """
    with _lock:
        _buffer.append(record)
        if _should_flush():
            _flush_to_mlflow()


def force_flush() -> None:
    """Força o envio imediato do buffer ao MLflow (use no shutdown da app)."""
    with _lock:
        if _buffer:
            _flush_to_mlflow()


# ---------------------------------------------------------------------------
# Lógica interna
# ---------------------------------------------------------------------------

def _should_flush() -> bool:
    global _last_flush
    elapsed = time.time() - _last_flush
    return len(_buffer) >= _BUFFER_SIZE or elapsed >= _FLUSH_INTERVAL


def _flush_to_mlflow() -> None:
    """
    Agrega as predições do buffer e loga métricas no MLflow.
    Mantém um run de produção aberto (run_name='production').
    """
    global _last_flush

    if not _buffer:
        return

    records = list(_buffer)
    _buffer.clear()
    _last_flush = time.time()

    try:
        metadata = get_metadata()
        experiment_name = f"production_{metadata.get('ticker', 'AAPL')}"
        mlflow.set_experiment(experiment_name)

        directions = [r.direction for r in records]
        latencies = [r.latency_ms for r in records]
        returns = [r.predicted_return for r in records]
        high_conf = [r for r in records if r.confidence == "HIGH"]

        n = len(records)
        up_pct = directions.count("UP") / n * 100
        down_pct = directions.count("DOWN") / n * 100
        neutral_pct = directions.count("NEUTRAL") / n * 100
        avg_latency = sum(latencies) / n
        avg_return = sum(returns) / n
        high_conf_pct = len(high_conf) / n * 100

        # Detecta drift simples: desvio na distribuição de direção
        # Um modelo estável deve ter UP/DOWN próximos ao histórico de treino
        direction_bias = abs(up_pct - down_pct)

        with mlflow.start_run(run_name="production", nested=False):
            mlflow.log_metrics({
                "prod_predictions_count":    n,
                "prod_up_pct": round(up_pct, 2),
                "prod_down_pct": round(down_pct, 2),
                "prod_neutral_pct": round(neutral_pct, 2),
                "prod_avg_latency_ms": round(avg_latency, 2),
                "prod_avg_predicted_return": round(avg_return, 4),
                "prod_high_confidence_pct": round(high_conf_pct, 2),
                "prod_direction_bias": round(direction_bias, 2),
            })

            # Alerta de drift: bias > 30pp pode indicar regime de mercado diferente
            if direction_bias > 30:
                mlflow.set_tag("drift_alert", f"direction_bias={direction_bias:.1f}pp")
                logger.WARNING(
                    f"[DRIFT] Viés de direção elevado: {direction_bias:.1f}pp "
                    f"(UP={up_pct:.1f}% / DOWN={down_pct:.1f}%)"
                )

            mlflow.set_tag("batch_timestamp", datetime.now(timezone.utc).isoformat())
            mlflow.set_tag("model_version", metadata.get("model_version", "unknown"))

        logger.INFO(
            f"[MLflow] {n} predições logadas — "
            f"avg_latency={avg_latency:.1f}ms, bias={direction_bias:.1f}pp"
        )

    except Exception as exc:
        logger.error(f"[MLflow] Falha ao logar predições: {exc}")


# ---------------------------------------------------------------------------
# Background thread para flush periódico
# ---------------------------------------------------------------------------

def _start_background_flusher() -> None:
    """Inicia thread daemon que faz flush a cada _FLUSH_INTERVAL segundos."""

    def _loop():
        while True:
            time.sleep(_FLUSH_INTERVAL)
            with _lock:
                if _buffer:
                    logger.INFO("[MLflow] Flush periódico iniciado")
                    _flush_to_mlflow()

    t = threading.Thread(target=_loop, daemon=True, name="mlflow-flusher")
    t.start()
    logger.INFO(f"[MLflow] Background flusher iniciado (intervalo={_FLUSH_INTERVAL}s)")


# Inicia automaticamente ao importar o módulo
_start_background_flusher()