"""
Shared visual identity for the Streamlit apps (app.py and frontend/streamlit_app.py).

The core design idea: instead of giving each of the 8 emotions an arbitrary
color, every emotion is colored by AROUSAL, the axis the model actually
separates well (see the error analysis in README.md). High-arousal emotions
(happy, angry, fearful, disgust, surprised) get the warm "ember" accent;
low-arousal emotions (neutral, calm, sad) get the cool "current" accent.
This makes the model's real strengths and confusions visible in the UI
itself, rather than being purely decorative.
"""

INK = "#0C0F14"       # background
PANEL = "#171B23"     # cards / surfaces
PANEL_BORDER = "#262C39"
PAPER = "#E9EAEE"     # primary text
MUTED = "#8890A0"     # secondary text / captions
EMBER = "#FF6A4D"     # warm accent, high arousal
CURRENT = "#35C7C2"   # cool accent, low arousal

HIGH_AROUSAL = {"happy", "angry", "fearful", "disgust", "surprised"}
LOW_AROUSAL = {"neutral", "calm", "sad"}

EMOTION_EMOJI = {
    "neutral": "😐", "calm": "🙂", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}


def emotion_color(emotion: str) -> str:
    """Ember for high-arousal emotions, current (teal) for low-arousal ones."""
    return EMBER if emotion in HIGH_AROUSAL else CURRENT


THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {INK};
    color: {PAPER};
}}

/* Hero */
.ser-hero-title {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.6rem;
    letter-spacing: -0.01em;
    color: {PAPER};
    margin-bottom: 0.1rem;
}}
.ser-hero-sub {{
    font-family: 'Inter', sans-serif;
    color: {MUTED};
    font-size: 1rem;
    max-width: 640px;
    line-height: 1.5;
}}

/* Waveform divider */
.ser-waveform {{
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 28px;
    margin: 14px 0 22px 0;
}}
.ser-waveform span {{
    display: inline-block;
    width: 3px;
    background: linear-gradient(180deg, {EMBER}, {CURRENT});
    border-radius: 2px;
    opacity: 0.85;
}}

/* Upload zone styling */
[data-testid="stFileUploaderDropzone"] {{
    background-color: {PANEL} !important;
    border: 1px dashed {PANEL_BORDER} !important;
    border-radius: 12px !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {{
    color: {MUTED} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Prediction badge */
.ser-prediction {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-left: 4px solid var(--ser-accent, {EMBER});
    border-radius: 10px;
    padding: 14px 20px;
    margin: 10px 0 18px 0;
}}
.ser-prediction-label {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.4rem;
    color: {PAPER};
}}
.ser-prediction-conf {{
    font-family: 'JetBrains Mono', monospace;
    color: {MUTED};
    font-size: 0.85rem;
}}

/* Always-visible section box (replaces expanders) */
.ser-section {{
    background: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
    padding: 16px 20px;
    margin: 14px 0;
}}
.ser-section-title {{
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: {PAPER};
    margin-bottom: 8px;
}}
.ser-section-text {{
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: {MUTED};
    line-height: 1.6;
}}

/* Captions / limitations text */
.ser-caption {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {MUTED};
    line-height: 1.6;
}}

/* Streamlit expander header */
[data-testid="stExpander"] {{
    background-color: {PANEL} !important;
    border: 1px solid {PANEL_BORDER} !important;
    border-radius: 10px !important;
}}
</style>
"""


def waveform_divider_html(n_bars: int = 48) -> str:
    """A static waveform-style divider, ember-to-teal gradient, varied bar heights."""
    import random

    random.seed(7)  # stable across reruns
    bars = "".join(
        f'<span style="height:{random.randint(6, 28)}px;"></span>' for _ in range(n_bars)
    )
    return f'<div class="ser-waveform">{bars}</div>'


def section_box_html(title: str, body_html: str) -> str:
    """An always-visible box (used instead of collapsible expanders)."""
    return f"""
    <div class="ser-section">
        <div class="ser-section-title">{title}</div>
        <div class="ser-section-text">{body_html}</div>
    </div>
    """


def prediction_badge_html(emotion: str, confidence: float) -> str:
    color = emotion_color(emotion)
    emoji = EMOTION_EMOJI.get(emotion, "")
    return f"""
    <div class="ser-prediction" style="--ser-accent: {color};">
        <span style="font-size:1.6rem;">{emoji}</span>
        <div>
            <div class="ser-prediction-label">{emotion.capitalize()}</div>
            <div class="ser-prediction-conf">{confidence*100:.0f}% confident</div>
        </div>
    </div>
    """
