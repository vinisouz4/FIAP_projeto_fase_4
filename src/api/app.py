from fastapi import FastAPI
from src.routes.auth_routes import router_auth  
from src.routes.update_databases import router_update
from src.routes.health_routes import router as router_health
from src.routes.loader_model import router as router_loader_model

"""
Arquivo responsável pela organização das rotas da API
"""


def create_app():
    app = FastAPI(title="FIAP Projeto Fase 4", version="1.0.0")

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
        tags=["Model Loader"],
    )

    app.include_router(
        router_health,
        prefix="/api",
        tags=["Health"],
    )

    return app