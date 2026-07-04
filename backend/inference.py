"""
Model loading and inference logic, kept separate from main.py so the model
is loaded exactly once (at import time / app startup) rather than per
request — reloading a Keras model on every call would make the API
unusably slow and is the kind of mistake this separation is meant to avoid.
"""

import io
import os

import joblib
import librosa
import numpy as np

from src.feature_extraction import extract_frame_features, load_feature_config
from src.model_def import build_model

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model"))

EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

_state = {"loaded": False}


def load_artifacts():
    """Load model + preprocessing artifacts once. Safe to call multiple times."""
    if _state["loaded"]:
        return

    cfg = load_feature_config(os.path.join(MODEL_DIR, "feature_config.json"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    n_steps = cfg["max_frames"]
    n_feats = cfg["n_features"]
    model, attn_model = build_model(input_shape=(n_steps, n_feats))
    model.load_weights(os.path.join(MODEL_DIR, "emotion_model.h5"))

    class_names = [EMOTIONS[c] for c in label_encoder.classes_]

    _state.update(
        cfg=cfg,
        scaler=scaler,
        model=model,
        attn_model=attn_model,
        class_names=class_names,
        loaded=True,
    )


def is_loaded() -> bool:
    return _state["loaded"]


def predict_emotion(audio_bytes: bytes) -> dict:
    """Run inference on raw audio bytes. Returns a plain dict (see schemas.PredictionResponse)."""
    if not _state["loaded"]:
        load_artifacts()

    cfg = _state["cfg"]
    scaler = _state["scaler"]
    model = _state["model"]
    attn_model = _state["attn_model"]
    class_names = _state["class_names"]

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=cfg["sr"], duration=cfg["duration"])
    target_len = int(cfg["sr"] * cfg["duration"])
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    feats = extract_frame_features(y, sr, cfg)
    n_steps, n_feats = feats.shape
    flat_scaled = scaler.transform(feats.reshape(-1, n_feats))
    feats_scaled = flat_scaled.reshape(1, n_steps, n_feats)

    probs = model.predict(feats_scaled, verbose=0)[0]
    attn = attn_model.predict(feats_scaled, verbose=0)[0, :, 0]

    top_idx = int(np.argmax(probs))

    return {
        "predicted_emotion": class_names[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": {name: float(p) for name, p in zip(class_names, probs)},
        "attention_weights": [float(a) for a in attn],
    }
