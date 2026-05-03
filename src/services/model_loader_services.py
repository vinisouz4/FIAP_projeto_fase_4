import os
import tensorflow as tf
import json

from src.log.logs import LoggerHandler

logger = LoggerHandler(__name__)

MODEL_ARTIFACTS_PATH = os.getenv("MODEL_ARTIFACTS_PATH", "src/model_artifacts")
 
# Instâncias globais — populadas por load_artifacts()
model = None
metadata: dict = {}
 
 
def load_artifacts() -> None:
    """Carrega o model.keras e o inference_metadata.json para memória."""
    global model, metadata
 
    # ── model.keras ───────────────────────────────────────────
    model_path = os.path.join(MODEL_ARTIFACTS_PATH, "model.keras")
 
    if not os.path.exists(model_path):
        logger.error(f"model.keras não encontrado em: {MODEL_ARTIFACTS_PATH}")
        return
 
    model = tf.keras.models.load_model(model_path)
    logger.INFO(f"Modelo carregado: {model_path}")
    logger.INFO(f"input_shape : {model.input_shape}")
    logger.INFO(f"output_shape: {model.output_shape}")

    # ── inference_metadata.json ───────────────────────────────
    metadata_path = os.path.join(MODEL_ARTIFACTS_PATH, "inference_metadata.json")
 
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.INFO(f"Metadados carregados: {list(metadata.keys())}")
    else:
        logger.WARNING("inference_metadata.json não encontrado — metadados vazios.")
 
 
def get_model():
    return model
 
 
def get_metadata() -> dict:
    return metadata
 
 
def is_loaded() -> bool:
    return model is not None