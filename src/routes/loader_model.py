from fastapi import APIRouter, Depends
 
from src.services.model_loader_services import (
    load_artifacts,
    get_model,
    is_loaded
)

from src.services.auth_services import current_user

user_dependency = Depends(current_user)

router = APIRouter()
 
@router.get("/load_model")
def load_model(user: str = user_dependency):
    """Endpoint para carregar o modelo e o scaler na memória."""
    load_artifacts()
    return {
        "status": "ok",
        "model_loaded": is_loaded(),
        "model_input_shape": str(get_model().input_shape) if get_model() else None,
        "model_output_shape": str(get_model().output_shape) if get_model() else None
    }