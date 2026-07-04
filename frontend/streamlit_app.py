"""
Speech Emotion Recognition — thin Streamlit client.

Unlike app.py at the repo root (which loads the model directly), this
version calls the FastAPI backend's /predict endpoint — a decoupled,
API-first architecture where the model only needs to be loaded once, in
the backend process, regardless of how many frontend clients connect.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

EMOTION_EMOJI = {
    "neutral": "😐", "calm": "🙂", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}


def plot_probabilities(probabilities: dict):
    items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    labels = [f"{EMOTION_EMOJI.get(k, '')} {k}" for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(
        go.Bar(x=values, y=labels, orientation="h",
               marker=dict(color=values, colorscale="Sunsetdark"))
    )
    fig.update_layout(xaxis_title="Confidence", xaxis=dict(range=[0, 1]),
                       height=380, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def plot_attention(attention_weights: list, duration: float = 3.0):
    fig, ax = plt.subplots(figsize=(9, 2.2))
    times = np.linspace(0, duration, num=len(attention_weights))
    ax.plot(times, attention_weights, color="crimson", linewidth=2)
    ax.fill_between(times, attention_weights, alpha=0.25, color="crimson")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("attention weight")
    ax.set_title("Where the model focused when predicting")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="centered")
    st.title("🎤 Speech Emotion Recognition")
    st.caption(
        "Decoupled architecture: this UI calls a FastAPI backend over HTTP "
        "rather than loading the model in-process — the model loads once, "
        "in the API, no matter how many clients connect."
    )

    # Backend health check, surfaced up front rather than failing silently on upload
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if not health.get("model_loaded"):
            st.warning("Backend is up but the model hasn't finished loading yet. Try again in a moment.")
    except requests.exceptions.RequestException:
        st.error(
            f"Can't reach the backend at `{API_URL}`. Is it running? "
            f"(`uvicorn backend.main:app` or `docker compose up`)"
        )
        st.stop()

    uploaded = st.file_uploader(
        "Upload a short speech clip (wav / mp3 / ogg / flac / m4a)",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
    )

    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        st.audio(audio_bytes)

        with st.spinner("Analyzing..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, audio_bytes, uploaded.type or "audio/wav")},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Prediction request failed: {e}")
                st.stop()

        top = result["predicted_emotion"]
        conf = result["confidence"]
        st.markdown(f"### {EMOTION_EMOJI.get(top, '')} Predicted: **{top.capitalize()}** ({conf*100:.1f}% confidence)")

        st.plotly_chart(plot_probabilities(result["probabilities"]), use_container_width=True)

        with st.expander("Why did the model predict this? (attention over time)"):
            st.pyplot(plot_attention(result["attention_weights"]))
    else:
        st.info("Upload an audio clip to get started.")


if __name__ == "__main__":
    main()
