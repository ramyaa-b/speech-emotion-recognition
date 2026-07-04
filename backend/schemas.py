from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    attention_weights: list[float]  # per time-frame weight, for visualizing focus


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
