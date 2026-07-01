"""
Speech Emotion Recognition — Streamlit app.

Loads the model artifacts produced by notebooks/Emotion_Recognition_v2_training.ipynb
(model/emotion_model.h5 + scaler.pkl + label_encoder.pkl + feature_config.json)
and runs inference on an uploaded audio clip, using the exact same feature
extraction as training (src/feature_extraction.py) so there's no
train/inference preprocessing drift.
"""

import io
import os

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.feature_extraction import extract_frame_features, load_audio, load_feature_config
from src.model_def import build_model

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

EMOTION_EMOJI = {
    "neutral": "😐", "calm": "🙂", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    cfg = load_feature_config(os.path.join(MODEL_DIR, "feature_config.json"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    n_steps = cfg["max_frames"]
    n_feats = cfg["n_features"]
    model, attn_model = build_model(input_shape=(n_steps, n_feats))
    model.load_weights(os.path.join(MODEL_DIR, "emotion_model.h5"))

    class_names = [EMOTIONS[c] for c in label_encoder.classes_]
    return cfg, scaler, label_encoder, model, attn_model, class_names


def predict(audio_bytes: bytes, cfg, scaler, model, attn_model, class_names):
    # librosa can read directly from a file-like object for common formats
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=cfg["sr"], duration=cfg["duration"])
    target_len = int(cfg["sr"] * cfg["duration"])
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    feats = extract_frame_features(y, sr, cfg)  # (time_steps, n_features)

    n_steps, n_feats = feats.shape
    flat = feats.reshape(-1, n_feats)
    flat_scaled = scaler.transform(flat)
    feats_scaled = flat_scaled.reshape(1, n_steps, n_feats)

    probs = model.predict(feats_scaled, verbose=0)[0]
    attn = attn_model.predict(feats_scaled, verbose=0)[0, :, 0]

    return probs, attn, y, sr


def plot_probabilities(probs, class_names):
    order = np.argsort(probs)[::-1]
    fig = go.Figure(
        go.Bar(
            x=[probs[i] for i in order],
            y=[f"{EMOTION_EMOJI[class_names[i]]} {class_names[i]}" for i in order],
            orientation="h",
            marker=dict(color=[probs[i] for i in order], colorscale="Sunsetdark"),
        )
    )
    fig.update_layout(
        xaxis_title="Confidence",
        xaxis=dict(range=[0, 1]),
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def plot_waveform_attention(y, sr, attn, cfg):
    fig, ax1 = plt.subplots(figsize=(9, 2.6))
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax1.plot(times, y, color="#888", linewidth=0.5)
    ax1.set_ylabel("waveform")
    ax1.set_xlabel("time (s)")

    ax2 = ax1.twinx()
    attn_times = np.linspace(0, cfg["duration"], num=len(attn))
    ax2.plot(attn_times, attn, color="crimson", linewidth=2)
    ax2.fill_between(attn_times, attn, alpha=0.25, color="crimson")
    ax2.set_ylabel("attention weight", color="crimson")
    ax2.tick_params(axis="y", colors="crimson")

    plt.title("Where the model focused when predicting")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="centered")

    st.title("🎤 Speech Emotion Recognition")
    st.caption(
        "CNN + BiLSTM + attention, trained on RAVDESS with a speaker-independent "
        "split — 58% accuracy on 5 held-out speakers the model never trained on "
        "(random guessing on 8 classes = 12.5%)."
    )

    cfg, scaler, label_encoder, model, attn_model, class_names = load_artifacts()

    uploaded = st.file_uploader(
        "Upload a short speech clip (wav / mp3 / ogg / flac / m4a)",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
    )

    if uploaded is not None:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes)

        with st.spinner("Analyzing..."):
            probs, attn, y, sr = predict(audio_bytes, cfg, scaler, model, attn_model, class_names)

        top_idx = int(np.argmax(probs))
        top_emotion = class_names[top_idx]
        st.markdown(f"### {EMOTION_EMOJI[top_emotion]} Predicted: **{top_emotion.capitalize()}** ({probs[top_idx]*100:.1f}% confidence)")

        st.plotly_chart(plot_probabilities(probs, class_names), use_container_width=True)

        with st.expander("Why did the model predict this? (attention over time)"):
            st.pyplot(plot_waveform_attention(y, sr, attn, cfg))
            st.caption(
                "The red line shows which moments in the clip the model weighted "
                "most heavily — it doesn't just average the whole clip, it learns "
                "to focus on the most emotionally salient frames."
            )

        with st.expander("Model limitations"):
            st.markdown(
                "- Trained on **acted** emotional speech (RAVDESS), not spontaneous "
                "speech — real-world tone tends to be subtler.\n"
                "- Weakest on **fearful** (23% recall) and **happy** (38% recall); "
                "strongest on **surprised** (95% recall) and **calm** (85% recall).\n"
                "- Common confusions: happy↔surprised, sad↔calm, fearful↔angry/disgust "
                "— these overlap in pitch and energy, which is exactly the kind of "
                "distinction that's hard from audio alone."
            )
    else:
        st.info("Upload an audio clip to get started.")


if __name__ == "__main__":
    main()
