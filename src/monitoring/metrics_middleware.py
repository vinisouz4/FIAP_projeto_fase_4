"""
Middleware de monitoramento — Prometheus
Rastreia: tempo de resposta, contagem de requisições, status codes,
utilização de CPU/memória e métricas específicas do modelo.
"""

import time
import psutil
import os
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.log.logs import LoggerHandler

logger = LoggerHandler(__name__)



# Contagem total de requisições por rota e status
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total de requisições recebidas",
    ["method", "endpoint", "status_code"],
)

# Histograma de latência (buckets em segundos)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Duração das requisições em segundos",
    ["method", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Requisições em andamento
REQUESTS_IN_PROGRESS = Gauge(
    "api_requests_in_progress",
    "Requisições sendo processadas no momento",
    ["method", "endpoint"],
)

# CPU e memória do processo
CPU_USAGE = Gauge("process_cpu_percent", "Uso de CPU do processo (%)")
MEMORY_USAGE = Gauge("process_memory_mb", "Uso de memória RSS do processo (MB)")
MEMORY_PERCENT = Gauge("process_memory_percent", "Uso de memória do processo (%)")

# Métricas específicas do predict
PREDICT_COUNT = Counter(
    "model_predict_total",
    "Total de predições realizadas",
    ["direction", "confidence"],
)

PREDICT_LATENCY = Histogram(
    "model_predict_duration_seconds",
    "Tempo de inferência do modelo",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

PREDICT_RETURN = Histogram(
    "model_predicted_return_pct",
    "Distribuição dos retornos previstos (%)",
    buckets=[-3.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 3.0],
)

# Disponibilidade do modelo
MODEL_LOADED = Gauge("model_loaded", "1 se o modelo está carregado, 0 caso contrário")


# ---------------------------------------------------------------------------
# Coleta de recursos do sistema
# ---------------------------------------------------------------------------

_process = psutil.Process(os.getpid())


def collect_system_metrics() -> None:
    """Atualiza as gauges de CPU e memória — chamado a cada requisição."""
    try:
        CPU_USAGE.set(_process.cpu_percent(interval=None))
        mem = _process.memory_info()
        MEMORY_USAGE.set(mem.rss / 1024 / 1024)
        MEMORY_PERCENT.set(_process.memory_percent())
    except Exception as exc:
        logger.WARNING(f"Falha ao coletar métricas de sistema: {exc}")


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Intercepta todas as requisições e registra métricas de latência,
    contagem e status no Prometheus.
    """

    # Rotas que não precisam ser rastreadas individualmente
    _SKIP_PATHS = {"/metrics", "/health", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Ignora rotas de infra
        if path in self._SKIP_PATHS:
            return await call_next(request)

        method = request.method
        endpoint = self._normalize_path(path)

        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        collect_system_metrics()

        start = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            logger.ERROR(f"Erro não tratado em {endpoint}: {exc}")
            raise
        finally:
            duration = time.perf_counter() - start

            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).inc()

            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()

    @staticmethod
    def _normalize_path(path: str) -> str:
        """
        Agrupa rotas com IDs dinâmicos para evitar cardinalidade infinita.
        Ex: /items/123 → /items/{id}
        """
        parts = path.split("/")
        normalized = []
        for part in parts:
            if part.isdigit() or (len(part) > 20 and "-" in part):
                normalized.append("{id}")
            else:
                normalized.append(part)
        return "/".join(normalized)


# ---------------------------------------------------------------------------
# Endpoint /metrics (expõe para o Prometheus scrape)
# ---------------------------------------------------------------------------

async def metrics_endpoint(request: Request) -> Response:
    """Expõe métricas no formato Prometheus text."""
    collect_system_metrics()
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Helpers para uso na rota /predict
# ---------------------------------------------------------------------------

def record_prediction(
    direction: str,
    confidence: str,
    predicted_return_pct: float,
    duration_seconds: float,
) -> None:
    """
    Registra métricas de uma predição individual.
    Chame ao final da rota /predict após a inferência.
    """
    PREDICT_COUNT.labels(direction=direction, confidence=confidence).inc()
    PREDICT_LATENCY.observe(duration_seconds)
    PREDICT_RETURN.observe(predicted_return_pct)


def set_model_status(loaded: bool) -> None:
    """Atualiza a gauge que indica se o modelo está disponível."""
    MODEL_LOADED.set(1 if loaded else 0)