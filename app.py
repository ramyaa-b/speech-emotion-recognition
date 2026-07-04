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
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.feature_extraction import extract_frame_features, load_feature_config
from src.model_def import build_model
from src.ui_theme import (
    CURRENT,
    EMBER,
    EMOTION_EMOJI,
    INK,
    MUTED,
    PANEL,
    PAPER,
    THEME_CSS,
    emotion_color,
    prediction_badge_html,
    waveform_divider_html,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
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
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=cfg["sr"], duration=cfg["duration"])
    target_len = int(cfg["sr"] * cfg["duration"])
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    feats = extract_frame_features(y, sr, cfg)

    n_steps, n_feats = feats.shape
    flat = feats.reshape(-1, n_feats)
    flat_scaled = scaler.transform(flat)
    feats_scaled = flat_scaled.reshape(1, n_steps, n_feats)

    probs = model.predict(feats_scaled, verbose=0)[0]
    attn = attn_model.predict(feats_scaled, verbose=0)[0, :, 0]

    return probs, attn, y, sr


def plot_probabilities(probs, class_names):
    order = np.argsort(probs)[::-1]
    labels = [f"{EMOTION_EMOJI[class_names[i]]}  {class_names[i]}" for i in order]
    values = [probs[i] for i in order]
    colors = [emotion_color(class_names[i]) for i in order]

    fig = go.Figure(
        go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors))
    )
    fig.update_layout(
        xaxis_title=None,
        xaxis=dict(range=[0, 1], gridcolor="#262C39", tickfont=dict(color=MUTED)),
        yaxis=dict(tickfont=dict(color=PAPER, size=14)),
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def plot_waveform_attention(y, sr, attn, cfg, predicted_emotion):
    accent = emotion_color(predicted_emotion)

    fig, ax1 = plt.subplots(figsize=(9, 2.6))
    fig.patch.set_facecolor(INK)
    ax1.set_facecolor(INK)

    times = np.linspace(0, len(y) / sr, num=len(y))
    ax1.plot(times, y, color=MUTED, linewidth=0.5)
    ax1.set_ylabel("waveform", color=MUTED, fontsize=9)
    ax1.set_xlabel("time (s)", color=MUTED, fontsize=9)
    ax1.tick_params(colors=MUTED, labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color("#262C39")

    ax2 = ax1.twinx()
    attn_times = np.linspace(0, cfg["duration"], num=len(attn))
    ax2.plot(attn_times, attn, color=accent, linewidth=2)
    ax2.fill_between(attn_times, attn, alpha=0.25, color=accent)
    ax2.set_ylabel("attention", color=accent, fontsize=9)
    ax2.tick_params(axis="y", colors=accent, labelsize=8)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="centered")
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    st.markdown('<div class="ser-hero-title">🎤 Speech Emotion Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ser-hero-sub">CNN + BiLSTM with attention, trained on RAVDESS with a '
        "speaker-independent split — evaluated on five actors the model never heard during "
        "training, not a random shuffle that leaks voice identity into the test set.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(waveform_divider_html(), unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="ser-stat-row">
            <div class="ser-stat"><div class="ser-stat-num">58.0%</div><div class="ser-stat-label">Test accuracy</div></div>
            <div class="ser-stat"><div class="ser-stat-num">12.5%</div><div class="ser-stat-label">Random baseline</div></div>
            <div class="ser-stat"><div class="ser-stat-num">5</div><div class="ser-stat-label">Held-out speakers</div></div>
        </div>
        """,
        unsafe_allow_html=True,
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

        st.markdown(prediction_badge_html(top_emotion, probs[top_idx]), unsafe_allow_html=True)

        st.plotly_chart(plot_probabilities(probs, class_names), use_container_width=True)

        with st.expander("Why did the model predict this? (attention over time)"):
            st.pyplot(plot_waveform_attention(y, sr, attn, cfg, top_emotion))
            st.markdown(
                '<div class="ser-caption">The colored line shows which moments in the clip the '
                "model weighted most heavily — it doesn't just average the whole clip, it learns "
                "to focus on the most emotionally salient frames. Line color matches the "
                "predicted emotion's arousal group (ember = high-arousal, teal = low-arousal).</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Model limitations"):
            st.markdown(
                f"""
                <div class="ser-caption">
                &bull; Trained on <b style="color:{PAPER}">acted</b> emotional speech (RAVDESS), not spontaneous speech — real-world tone tends to be subtler.<br>
                &bull; Weakest on <b style="color:{EMBER}">fearful</b> (23% recall) and <b style="color:{EMBER}">happy</b> (38% recall); strongest on <b style="color:{EMBER}">surprised</b> (95% recall) and <b style="color:{CURRENT}">calm</b> (85% recall).<br>
                &bull; Common confusions cluster <i>within</i> arousal groups, not across them — happy&harr;surprised (both high-arousal), sad&harr;calm (both low-arousal) — the acoustic signal for arousal is clearer than for valence.
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="ser-caption">Upload an audio clip to get started.</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
