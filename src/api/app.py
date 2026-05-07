from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.routes.auth_routes import router_auth
from src.routes.update_databases import router_update
from src.routes.health_routes import router as router_health
from src.routes.loader_model import router as router_loader_model
from src.routes.predict_routes import router as router_predict
from src.routes.metrics_routes import router as router_metrics

from src.services.model_loader_services import load_artifacts, is_loaded
from src.monitoring.metrics_middleware import (
    PrometheusMiddleware,
    metrics_endpoint,
    set_model_status,
)
from src.monitoring.model_monitor import force_flush



@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    load_artifacts()
    set_model_status(is_loaded())
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────
    force_flush()   # garante que predições pendentes sejam logadas no MLflow


def create_app():
    app = FastAPI(
        title="FIAP Projeto Fase 4",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Middleware de métricas — deve ser adicionado antes das rotas
    app.add_middleware(PrometheusMiddleware)

    # Endpoint /metrics consumido pelo Prometheus (fora do prefix /api
    # para compatibilidade com o scraper padrão)
    app.add_route("/metrics", metrics_endpoint)

    app.include_router(
        router_auth,
        prefix="/api",
        tags=["Auth"],
    )

    app.include_router(
        router_update,
        prefix="/api",
        tags=["Update Databases"],
    )

    app.include_router(
        router_loader_model,
        prefix="/api",
        tags=["Modelo"],
    )

    app.include_router(
        router_predict,
        prefix="/api",
        tags=["Modelo"],
    )

    app.include_router(
        router_metrics,
        prefix="/api",
        tags=["Modelo"],
    )

    app.include_router(
        router_health,
        prefix="/api",
        tags=["Health"],
    )

    return app