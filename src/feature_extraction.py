"""
Shared audio feature extraction.

This module is intentionally the single source of truth for how raw audio
becomes model input. It's used at inference time by app.py, and mirrors
exactly what the training notebook (notebooks/Emotion_Recognition_v2_training.ipynb)
does when building the dataset. If you retrain the model with different
features, update this file and the notebook together — the two are only as
consistent as you keep them.
"""

import json
import os

import librosa
import numpy as np


def load_feature_config(config_path: str) -> dict:
    """Load the exact feature/audio config saved by the training notebook."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_audio(file_path: str, cfg: dict) -> tuple[np.ndarray, int]:
    """Load an audio file, resample, and pad/truncate to a fixed duration."""
    y, sr = librosa.load(file_path, sr=cfg["sr"], duration=cfg["duration"])
    target_len = int(cfg["sr"] * cfg["duration"])
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr


def extract_frame_features(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Return a (max_frames, n_features) matrix of frame-level audio features:
    MFCC + delta + delta-delta, chroma, spectral contrast, zero-crossing
    rate, RMS energy. Padded/truncated to a fixed number of time steps so
    every clip produces an identically-shaped array.
    """
    hop = cfg["hop_length"]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg["n_mfcc"], hop_length=hop)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)

    feats = np.vstack([mfcc, mfcc_delta, mfcc_delta2, chroma, contrast, zcr, rms])
    feats = feats.T  # (n_frames, n_features)

    max_frames = cfg["max_frames"]
    if feats.shape[0] < max_frames:
        pad_width = max_frames - feats.shape[0]
        feats = np.pad(feats, ((0, pad_width), (0, 0)), mode="constant")
    else:
        feats = feats[:max_frames, :]

    return feats.astype(np.float32)


def extract_features_from_file(file_path: str, cfg: dict) -> np.ndarray:
    """Convenience wrapper: file path -> (max_frames, n_features) array."""
    y, sr = load_audio(file_path, cfg)
    return extract_frame_features(y, sr, cfg)
