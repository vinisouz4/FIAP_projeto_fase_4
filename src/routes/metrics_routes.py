from fastapi import APIRouter, HTTPException


from src.services.model_loader_services import get_metadata, is_loaded
from src.models.metrics_models import ModelMetricsResponse, TrainMetrics



router = APIRouter(prefix="/metrics")



@router.get("/metrics", response_model=ModelMetricsResponse, summary="Métricas e metadados do modelo")
async def get_metrics():
    """
    Retorna as métricas de treino e os metadados de configuração do modelo carregado.

    - **MAE_pct**: erro absoluto médio em % 
    - **RMSE_pct**: raiz do erro quadrático médio em %
    - **DA**: directional accuracy — % de acertos de direção (UP/DOWN)
    - **DA_naive**: DA de um modelo ingênuo (baseline)
    - **delta_DA**: ganho de DA sobre o baseline
    """
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")

    metadata = get_metadata()

    if not metadata:
        raise HTTPException(status_code=404, detail="Metadados não encontrados.")

    return ModelMetricsResponse(
        model_version=metadata["model_version"],
        ticker=metadata["ticker"],
        target=metadata["target"],
        architecture=metadata["architecture"],
        framework=metadata["framework"],
        window_size=metadata["window_size"],
        n_features=metadata["n_features"],
        features=metadata["features"],
        normalization=metadata["normalization"],
        confidence_threshold=metadata["confidence_threshold"],
        train_metrics=TrainMetrics(**metadata["train_metrics"]),
    )