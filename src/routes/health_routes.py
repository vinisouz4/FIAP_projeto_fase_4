from fastapi import APIRouter, Depends

from src.services.model_loader_services import (
    get_model,
    is_loaded
)

from src.services.auth_services import current_user
from src.models.health_models import HealthResponse

user_dependency = Depends(current_user)


router = APIRouter()
 
@router.get("/health", response_model=HealthResponse)
def health(user: str = user_dependency):
    """Verifica se o modelo está carregado e a API está operacional."""
    loaded = is_loaded()
    m = get_model()

    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded = loaded,
        model_input_shape = str(m.input_shape) if m else None,
        model_output_shape = str(m.output_shape) if m else None
    )