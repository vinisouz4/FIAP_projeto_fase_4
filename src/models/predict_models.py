from pydantic import BaseModel, Field

class DailyOHLCV(BaseModel):
    """Um dia de dados OHLCV para o ticker."""
    open: float = Field(..., description="Preço de abertura")
    high: float = Field(..., description="Preço máximo")
    low: float = Field(..., description="Preço mínimo")
    close: float = Field(..., description="Preço de fechamento")
    volume: float = Field(..., description="Volume negociado")


class PredictRequest(BaseModel):
    """
    Corpo da requisição.
    Envie exatamente window_size + 26 dias de dados OHLCV ordenados do mais antigo
    ao mais recente para que todas as features (MACD_26, RSI_14, etc.) sejam
    calculadas sem NaN na janela final de 30 dias.
    Valor mínimo seguro: 56 dias (26 para MACD + 30 de janela).
    """
    ticker: str = Field("AAPL", description="Ticker do ativo (apenas AAPL suportado na v2)")
    history: list[DailyOHLCV] = Field(
        ...,
        min_length=56,
        description="Histórico OHLCV ordenado crescente por data (mínimo 56 registros)",
    )


class PredictResponse(BaseModel):
    ticker: str
    predicted_return_pct: float = Field(..., description="Retorno previsto para t+1 em %")
    direction: str = Field(..., description="UP | DOWN | NEUTRAL")
    confidence: str = Field(..., description="HIGH | LOW  (|pred| > confidence_threshold)")
    model_version: str