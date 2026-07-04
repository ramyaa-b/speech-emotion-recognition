"""
Speech Emotion Recognition, thin Streamlit client.

Unlike app.py at the repo root (which loads the model directly), this
version calls the FastAPI backend's /predict endpoint: a decoupled,
API-first setup where the model only needs to be loaded once, in the
backend process, regardless of how many frontend clients connect.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

# allow importing src/ui_theme.py when run from the frontend/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ui_theme import (
    EMOTION_EMOJI,
    INK,
    MUTED,
    PAPER,
    THEME_CSS,
    emotion_color,
    prediction_badge_html,
    section_box_html,
    waveform_divider_html,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def plot_probabilities(probabilities: dict):
    items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    labels = [f"{EMOTION_EMOJI.get(k, '')}  {k}" for k, _ in items]
    values = [v for _, v in items]
    colors = [emotion_color(k) for k, _ in items]

    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors)))
    fig.update_layout(
        xaxis=dict(range=[0, 1], gridcolor="#262C39", tickfont=dict(color=MUTED)),
        yaxis=dict(tickfont=dict(color=PAPER, size=14)),
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def plot_attention(attention_weights: list, predicted_emotion: str, duration: float = 3.0):
    accent = emotion_color(predicted_emotion)
    fig, ax = plt.subplots(figsize=(9, 2.2))
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(INK)

    times = np.linspace(0, duration, num=len(attention_weights))
    ax.plot(times, attention_weights, color=accent, linewidth=2)
    ax.fill_between(times, attention_weights, alpha=0.25, color=accent)
    ax.set_xlabel("time (seconds)", color=MUTED, fontsize=9)
    ax.set_ylabel("what mattered most", color=accent, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#262C39")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🔊", layout="centered")
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    st.markdown('<div class="ser-hero-title">Speech Emotion Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ser-hero-sub">Upload a short voice clip and see what emotion it sounds like.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(waveform_divider_html(), unsafe_allow_html=True)

    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if not health.get("model_loaded"):
            st.warning("The model is still starting up. Try again in a moment.")
    except requests.exceptions.RequestException:
        st.error(f"Can't reach the backend at `{API_URL}`. Make sure it's running.")
        st.stop()

    uploaded = st.file_uploader(
        "Upload a short audio clip (wav, mp3, ogg, flac, or m4a)",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
    )

    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        st.audio(audio_bytes)

        with st.spinner("Listening..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, audio_bytes, uploaded.type or "audio/wav")},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Something went wrong getting a prediction: {e}")
                st.stop()

        top = result["predicted_emotion"]
        conf = result["confidence"]

        st.markdown(prediction_badge_html(top, conf), unsafe_allow_html=True)

        st.plotly_chart(plot_probabilities(result["probabilities"]), use_container_width=True)

        st.markdown(
            section_box_html(
                "What part of your voice mattered most",
                "The highlighted line below shows which part of your clip had the "
                "biggest effect on the result.",
            ),
            unsafe_allow_html=True,
        )
        st.pyplot(plot_attention(result["attention_weights"], top))
    else:
        st.markdown(
            '<div class="ser-caption">Upload an audio clip to get started.</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
