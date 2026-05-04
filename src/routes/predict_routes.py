from fastapi import APIRouter, HTTPException, Depends


from src.models.predict_models import PredictRequest, PredictResponse
from src.services.predict_service import _build_features, _window_normalize, _build_input
from src.services.model_loader_services import get_model, get_metadata
from src.services.auth_services import current_user

router = APIRouter(prefix="/predict")


user_dependency = Depends(current_user)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

 
@router.post("/", response_model=PredictResponse, summary="Previsão de retorno t+1")
async def predict(body: PredictRequest, user: str = user_dependency):
    """
    Retorna a previsão de retorno percentual para o próximo dia útil.
 
    - **history**: mínimo de 56 registros OHLCV ordenados do mais antigo ao mais recente.
    - **predicted_return_pct**: retorno esperado em % (positivo = alta, negativo = queda).
    - **direction**: `UP` / `DOWN` / `NEUTRAL` (NEUTRAL quando |pred| < 0.05%).
    - **confidence**: `HIGH` quando |pred| > `confidence_threshold` do metadata (0.3%).
    """
    # ── Carrega modelo e metadata ────────────────────────────────────────────
    model = get_model()
    metadata = get_metadata()
 
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")
 
    # ── Validações de negócio ────────────────────────────────────────────────
    if body.ticker.upper() != metadata.get("ticker", "AAPL").upper():
        raise HTTPException(
            status_code=422,
            detail=f"Ticker '{body.ticker}' não suportado. Modelo treinado para '{metadata['ticker']}'.",
        )
 
    window_size: int = metadata["window_size"]  # 30
    threshold: float = metadata["confidence_threshold"]  # 0.003
 
    # ── Pré-processamento ────────────────────────────────────────────────────
    try:
        feature_matrix = _build_features(body.history)
        X = _build_input(feature_matrix, window_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro no pré-processamento: {exc}")
 
    # ── Inferência ───────────────────────────────────────────────────────────
    try:
        raw_pred: float = float(model.predict(X, verbose=0).flatten()[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {exc}")
 
    # ── Pós-processamento ────────────────────────────────────────────────────
    predicted_pct = round(raw_pred * 100, 4)
 
    if abs(raw_pred) < 0.0005:          # < 0.05% → sem sinal claro
        direction = "NEUTRAL"
    elif raw_pred > 0:
        direction = "UP"
    else:
        direction = "DOWN"
 
    confidence = "HIGH" if abs(raw_pred) >= threshold else "LOW"
 
    return PredictResponse(
        ticker=body.ticker.upper(),
        predicted_return_pct=predicted_pct,
        direction=direction,
        confidence=confidence,
        model_version=metadata.get("model_version", "2.0.0"),
    )