from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_input_shape: Optional[str]
    model_output_shape: Optional[str]
 