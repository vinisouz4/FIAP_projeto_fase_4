import os
import threading
import json

import tensorflow as tf

from src.log.logs import LoggerHandler

logger = LoggerHandler(__name__)

MODEL_ARTIFACTS_PATH = os.getenv("MODEL_ARTIFACTS_PATH", "src/model_artifacts")

# Instâncias globais — populadas por load_artifacts()
_model = None
_metadata: dict = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_artifacts() -> bool:
    """
    Carrega o model.keras e o inference_metadata.json para memória.
    Retorna True se ambos foram carregados com sucesso, False caso contrário.
    Thread-safe: pode ser chamado no startup sem risco de race condition.
    """
    global _model, _metadata

    model_path    = os.path.join(MODEL_ARTIFACTS_PATH, "model.keras")
    metadata_path = os.path.join(MODEL_ARTIFACTS_PATH, "inference_metadata.json")

    # ── model.keras ───────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        logger.error(f"model.keras não encontrado em: {MODEL_ARTIFACTS_PATH}")
        return False

    try:
        loaded = tf.keras.models.load_model(model_path)
    except Exception as exc:
        logger.error(f"Falha ao carregar model.keras: {exc}")
        return False

    # Valida que o modelo aceita o shape esperado antes de aceitar em produção
    try:
        _validate_model(loaded, metadata_path)
    except ValueError as exc:
        logger.error(f"Modelo inválido: {exc}")
        return False

    # ── inference_metadata.json ───────────────────────────────────────────
    loaded_meta: dict = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                loaded_meta = json.load(f)
            logger.INFO(f"Metadados carregados: {list(loaded_meta.keys())}")
        except Exception as exc:
            logger.WARNING(f"Falha ao ler inference_metadata.json: {exc}")
    else:
        logger.WARNING("inference_metadata.json não encontrado — metadados vazios.")

    # Atualiza globals de forma atômica
    with _lock:
        _model    = loaded
        _metadata = loaded_meta

    logger.INFO(f"Modelo carregado: {model_path}")
    logger.INFO(f"input_shape : {_model.input_shape}")
    logger.INFO(f"output_shape: {_model.output_shape}")

    # Notifica o Prometheus que o modelo está disponível
    try:
        from src.monitoring.metrics_middleware import set_model_status
        set_model_status(True)
    except ImportError:
        pass  # monitoramento opcional — não bloqueia o carregamento

    return True


def unload_artifacts() -> None:
    """Libera o modelo da memória (útil em testes ou hot-reload)."""
    global _model, _metadata
    with _lock:
        _model    = None
        _metadata = {}

    try:
        from src.monitoring.metrics_middleware import set_model_status
        set_model_status(False)
    except ImportError:
        pass

    logger.INFO("Modelo descarregado da memória.")


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------

def get_model() -> tf.keras.Model | None:
    with _lock:
        return _model


def get_metadata() -> dict:
    with _lock:
        return _metadata.copy()


def is_loaded() -> bool:
    with _lock:
        return _model is not None


# ---------------------------------------------------------------------------
# Validação interna
# ---------------------------------------------------------------------------

def _validate_model(model: tf.keras.Model, metadata_path: str) -> None:
    """
    Verifica se o modelo carregado é compatível com o metadata.
    Lança ValueError se encontrar inconsistência.
    """
    input_shape = model.input_shape   # (None, window_size, n_features)

    if len(input_shape) != 3:
        raise ValueError(
            f"Shape de entrada inesperado: {input_shape}. "
            "Esperado: (None, window_size, n_features)."
        )

    if not os.path.exists(metadata_path):
        return  # sem metadata não há o que validar

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return

    expected_window   = meta.get("window_size")
    expected_features = meta.get("n_features")

    _, actual_window, actual_features = input_shape

    if expected_window and actual_window != expected_window:
        raise ValueError(
            f"window_size do modelo ({actual_window}) diverge do metadata ({expected_window})."
        )

    if expected_features and actual_features != expected_features:
        raise ValueError(
            f"n_features do modelo ({actual_features}) diverge do metadata ({expected_features})."
        )

    logger.INFO(
        f"Modelo validado — window_size={actual_window}, n_features={actual_features}"
    )