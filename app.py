"""
app.py
======
BioSound AI – Main Streamlit Application
=========================================
Living Being Sound Recognition & Classification System

Run with:
    streamlit run app.py
"""

import os
import sys
import io
import time
import tempfile
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SUPPORTED_CLASSES, CLASS_EMOJI, CLASS_COLOR, CLASS_LABELS,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, TOP_K_PREDICTIONS,
    DEFAULT_MODEL,
)
from src.utils.helpers import (
    format_confidence, format_duration, get_confidence_badge,
    audio_info_to_markdown, bytes_to_tempfile, cleanup_tempfile,
    validate_file_extension,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BioSound AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – dark theme with cards
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #0a0e17; }
[data-testid="stSidebar"]          { background: #111827; border-right: 1px solid #1f2937; }
h1,h2,h3,h4                        { color: #f0f4ff; }
p, li, label                        { color: #c9d1e0; }

/* ── Cards ──────────────────────────────────────────────── */
.biosound-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.biosound-card-accent {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1321 100%);
    border: 1px solid #2563eb44;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Result Banner ──────────────────────────────────────── */
.result-banner {
    text-align: center;
    padding: 2rem 1rem;
    border-radius: 16px;
    margin-bottom: 1rem;
}
.result-emoji  { font-size: 5rem; line-height: 1.2; }
.result-class  { font-size: 2.4rem; font-weight: 800; letter-spacing: 0.06em; }
.result-conf   { font-size: 1.4rem; opacity: 0.85; margin-top: 0.3rem; }
.badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}

/* ── Progress bars ──────────────────────────────────────── */
.pred-row { display:flex; align-items:center; margin:0.4rem 0; gap:0.6rem; }
.pred-label { width: 90px; font-size:0.9rem; color:#c9d1e0; text-align:right; }
.pred-bar-bg { flex:1; background:#1f2937; border-radius:6px; height:20px; overflow:hidden; }
.pred-bar    { height:100%; border-radius:6px; transition: width 0.6s ease; }
.pred-pct    { width: 52px; font-size:0.9rem; color:#e0e7ff; text-align:left; }

/* ── Metrics table ──────────────────────────────────────── */
.metric-row { display:flex; justify-content:space-between; padding:0.45rem 0;
              border-bottom:1px solid #1f2937; align-items:center; }
.metric-name { color:#94a3b8; font-size:0.95rem; }
.metric-val  { color:#60a5fa; font-weight:700; font-size:1rem; }
.metric-val-highlight { color:#34d399; font-weight:700; font-size:1rem; }

/* ── Section headers ────────────────────────────────────── */
.section-header {
    color:#7c9fe0;
    font-size:0.75rem;
    font-weight:700;
    letter-spacing:0.12em;
    text-transform:uppercase;
    margin-bottom:0.6rem;
    padding-bottom:0.3rem;
    border-bottom:1px solid #1f2937;
}

/* ── Status dots ────────────────────────────────────────── */
.dot-green  { color:#4ade80; font-size:0.9rem; }
.dot-orange { color:#fb923c; font-size:0.9rem; }
.dot-red    { color:#f87171; font-size:0.9rem; }

/* ── Streamlit overrides ────────────────────────────────── */
div[data-testid="stMetricValue"] { color: #60a5fa; }
.stButton > button {
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
    color: white; border:none; border-radius:8px;
    padding:0.5rem 1.5rem; font-weight:600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity:0.85; }
.stTabs [data-baseweb="tab"] { color:#94a3b8; font-size:0.9rem; }
.stTabs [aria-selected="true"] { color:#60a5fa !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "result":        None,
        "model_mode":    DEFAULT_MODEL,
        "processing":    False,
        "error_msg":     None,
        "show_perf":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ---------------------------------------------------------------------------
# Lazy Classifier Loader with Caching
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Loading PyTorch AI model into RAM...")
def get_cached_classifier(mode: str):
    """Cache the classifier and pre-load model weights so inference is instant."""
    from src.prediction.classifier import BioSoundClassifier
    return BioSoundClassifier(mode=mode, load_immediately=True)


# Warm up default model on app load
try:
    classifier_instance = get_cached_classifier(st.session_state.model_mode)
except Exception as e:
    st.error(f"Failed to load AI model: {e}")


# ---------------------------------------------------------------------------
# Audio Processing Helper
# ---------------------------------------------------------------------------
def process_audio(source, filename: str = "audio"):
    """
    Run the full inference pipeline on a source (bytes or path).
    Updates st.session_state.result or st.session_state.error_msg.
    """
    from src.audio.inference import run_pipeline

    st.session_state.processing = True
    st.session_state.error_msg  = None
    st.session_state.result     = None

    clf = get_cached_classifier(st.session_state.model_mode)
    tmp_path = None

    try:
        if isinstance(source, bytes):
            ext = os.path.splitext(filename)[1] or ".wav"
            tmp_path = bytes_to_tempfile(source, suffix=ext)
            result   = run_pipeline(tmp_path, classifier=clf)
        else:
            result = run_pipeline(source, classifier=clf)

        result["filename"] = filename
        st.session_state.result = result

    except ValueError as e:
        st.session_state.error_msg = str(e)
    except Exception as e:
        st.session_state.error_msg = f"An unexpected error occurred: {e}"
    finally:
        cleanup_tempfile(tmp_path)
        st.session_state.processing = False


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_header():
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem;">
        <div style="font-size:3.5rem; line-height:1.1;">🧠</div>
        <h1 style="font-size:2.4rem; font-weight:900; margin:0.2rem 0;
                   background:linear-gradient(90deg,#60a5fa,#a78bfa);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            BioSound AI
        </h1>
        <p style="color:#7c9fe0; font-size:1.05rem; margin:0;">
            Multiclass Living Being Sound Recognition System
        </p>
        <p style="color:#4b5563; font-size:0.8rem; margin-top:0.4rem;">
            Classifies sounds among: Human · Dog · Cat · Bird · Cow · Horse · Chicken · Sheep
        </p>
    </div>
    <hr style="border-color:#1f2937; margin:1rem 0 0.5rem;">
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        # Model selector
        model_choice = st.selectbox(
            "🤖 AI Model",
            options=["ast", "custom"],
            index=0 if st.session_state.model_mode == "ast" else 1,
            format_func=lambda x: "Audio Spectrogram Transformer (Pre-trained PyTorch)" if x == "ast" else "Custom BioSoundCNN",
            help="AST model works out-of-the-box. Custom CNN requires training first."
        )
        if model_choice != st.session_state.model_mode:
            st.session_state.model_mode = model_choice
            st.session_state.result     = None
            st.rerun()

        if model_choice == "ast":
            st.info("✅ Pre-trained Audio Transformer active — PyTorch powered.")
        else:
            from config import CUSTOM_MODEL_PATH
            if os.path.exists(CUSTOM_MODEL_PATH):
                st.success("✅ Custom model found and ready.")
            else:
                st.warning("⚠️ Custom model not trained yet.\nRun: `python src/model/train.py`")

        st.markdown("---")
        st.markdown("### 📋 Supported Classes")
        for cls in SUPPORTED_CLASSES:
            emoji = CLASS_EMOJI.get(cls, "•")
            st.markdown(f"{emoji} {cls.capitalize()}")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **BioSound AI** uses deep learning to classify
        the source of an audio recording among
        living beings.

        - **Confidence** = model's certainty for *this* clip
        - **Accuracy** = model performance on *test dataset*
        """)


def render_result_banner(result: dict):
    cls    = result["predicted_class"]
    conf   = result["confidence"]
    emoji  = CLASS_EMOJI.get(cls, "🔊")
    color  = CLASS_COLOR.get(cls, "#2196F3")
    label  = cls.upper()
    badge_color, badge_label = get_confidence_badge(conf)
    conf_str = format_confidence(conf)

    if conf >= CONFIDENCE_HIGH:
        dot = '<span class="dot-green">🟢</span>'
    elif conf >= CONFIDENCE_MEDIUM:
        dot = '<span class="dot-orange">🟠</span>'
    else:
        dot = '<span class="dot-red">🔴</span>'

    st.markdown(f"""
    <div class="result-banner" style="background:linear-gradient(135deg,{color}22,{color}11);
         border:2px solid {color}55;">
        <div class="result-emoji">{emoji}</div>
        <div class="result-class" style="color:{color};">{dot} {label}</div>
        <div class="result-conf">Confidence: <b>{conf_str}</b></div>
        <span class="badge" style="background:{badge_color}33;color:{badge_color};
              border:1px solid {badge_color}66;">{badge_label} CONFIDENCE</span>
    </div>
    """, unsafe_allow_html=True)


def render_top_predictions(result: dict):
    st.markdown('<div class="section-header">📊 Top Predictions</div>', unsafe_allow_html=True)
    top_k = result.get("top_k", [])
    for cls_name, prob in top_k:
        color   = CLASS_COLOR.get(cls_name, "#607D8B")
        emoji   = CLASS_EMOJI.get(cls_name, "•")
        pct     = prob * 100
        bar_w   = min(pct, 100)
        pct_str = f"{pct:.1f}%"
        st.markdown(f"""
        <div class="pred-row">
            <div class="pred-label">{emoji} {cls_name.capitalize()}</div>
            <div class="pred-bar-bg">
                <div class="pred-bar" style="width:{bar_w}%;background:{color};"></div>
            </div>
            <div class="pred-pct">{pct_str}</div>
        </div>
        """, unsafe_allow_html=True)


def render_audio_info(result: dict):
    info = result.get("audio_info", {})
    st.markdown('<div class="section-header">🎵 Audio Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration",    format_duration(info.get("duration_s", 0)))
    c2.metric("Sample Rate", f"{info.get('sample_rate', 0):,} Hz")
    c3.metric("Channels",    info.get("channels", 1))
    c1.metric("Peak",        f"{info.get('peak', 0):.3f}")
    c2.metric("RMS Level",   f"{info.get('rms', 0):.3f}")
    c3.metric("Samples",     f"{info.get('n_samples', 0):,}")


def render_visualizations(result: dict):
    tab1, tab2 = st.tabs(["🌊 Waveform", "🌈 Mel-Spectrogram"])
    with tab1:
        if result.get("waveform_fig"):
            st.pyplot(result["waveform_fig"], use_container_width=True)
    with tab2:
        if result.get("mel_fig"):
            st.pyplot(result["mel_fig"], use_container_width=True)
            st.caption("Mel-Spectrogram: frequency (y-axis) vs. time (x-axis), intensity in dB (color).")


def render_model_performance():
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True)

    mode = st.session_state.model_mode

    if mode == "ast":
        st.info("""
        **AST Mode (Pre-trained Transformer)** – Performance metrics are not computed in this mode.

        Audio Spectrogram Transformer is a pre-trained AudioSet classifier.
        To view custom test metrics for your dataset, train the Custom BioSoundCNN:

        ```bash
        python src/model/train.py     # train first
        python src/model/evaluate.py  # then evaluate
        ```
        """)
        return

    # Custom model mode — try to load cached metrics
    try:
        from src.model.evaluate import load_cached_metrics, plot_confusion_matrix
    except ImportError:
        st.error("Could not import evaluate module.")
        return

    metrics = load_cached_metrics()
    if metrics is None:
        st.warning("""
        No evaluation metrics found. Run the evaluation script first:
        ```bash
        python src/model/evaluate.py
        ```
        """)
        return

    # --- Overall metrics ---
    overall_acc = metrics.get("overall_accuracy", 0)
    human_m     = metrics.get("human_metrics", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="biosound-card">', unsafe_allow_html=True)
        st.markdown("**Overall Performance**")

        def metric_row(name, val, highlight=False):
            cls = "metric-val-highlight" if highlight else "metric-val"
            st.markdown(f"""
            <div class="metric-row">
                <span class="metric-name">{name}</span>
                <span class="{cls}">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        metric_row("Overall Accuracy", format_confidence(overall_acc), highlight=True)
        per_class = metrics.get("per_class", {})
        for cls in SUPPORTED_CLASSES:
            if cls in per_class:
                metric_row(
                    f"{CLASS_EMOJI.get(cls,'')} {cls.capitalize()} Accuracy",
                    format_confidence(per_class[cls].get("accuracy", 0))
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="biosound-card">', unsafe_allow_html=True)
        st.markdown("**🧑 Human Class Metrics**")
        metric_row("Human Accuracy",  format_confidence(human_m.get("accuracy", 0)),  True)
        metric_row("Human Precision", format_confidence(human_m.get("precision", 0)), True)
        metric_row("Human Recall",    format_confidence(human_m.get("recall", 0)),    True)
        metric_row("Human F1-Score",  format_confidence(human_m.get("f1", 0)),        True)
        n_test = metrics.get("n_test_samples", "N/A")
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-name">Test Samples</span>
            <span class="metric-val">{n_test}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Confusion Matrix ---
    st.markdown("**Confusion Matrix**")
    try:
        fig = plot_confusion_matrix(metrics)
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render confusion matrix: {e}")


# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------

def main():
    render_header()
    render_sidebar()

    # ── Input Section ──────────────────────────────────────────────────────
    st.markdown('<div class="biosound-card-accent">', unsafe_allow_html=True)
    st.markdown("### 🎙️ Sound Input")
    st.markdown("Choose a method to provide audio:")

    tab_mic, tab_upload = st.tabs(["🎤 Live Recording", "📂 Upload Audio File"])

    # ── Tab 1: Live Microphone ─────────────────────────────────────────────
    with tab_mic:
        st.markdown("""
        Click the button below to **record audio from your microphone**.
        Speak, make an animal sound, or play audio near the mic.
        """)
        audio_input = st.audio_input(
            "🎤 Record Sound",
            help="Click to start/stop recording."
        )
        if audio_input is not None:
            st.audio(audio_input, format="audio/wav")
            if st.button("🔍 Analyse Recording", use_container_width=True, key="btn_mic"):
                audio_bytes = audio_input.getvalue()
                with st.spinner("⚡ Running AI classification..."):
                    process_audio(audio_bytes, filename="recorded_audio.wav")
                st.rerun()

    # ── Tab 2: Upload File ─────────────────────────────────────────────────
    with tab_upload:
        st.markdown("Upload a **.wav**, **.mp3**, **.flac**, or **.m4a** audio file.")
        uploaded = st.file_uploader(
            "Choose audio file",
            type=["wav", "mp3", "flac", "m4a", "ogg"],
            help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
        )
        if uploaded is not None:
            st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")
            if st.button("🔍 Analyse File", use_container_width=True, key="btn_upload"):
                audio_bytes = uploaded.read()
                with st.spinner("⚡ Running AI classification..."):
                    process_audio(audio_bytes, filename=uploaded.name)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Error Display ──────────────────────────────────────────────────────
    if st.session_state.error_msg:
        st.error(f"❌ **Error:** {st.session_state.error_msg}")

    # ── Results ────────────────────────────────────────────────────────────
    result = st.session_state.result
    if result is not None:
        st.markdown("---")
        st.markdown("## 🎯 AI Prediction Result")

        st.markdown("""
        <div style="color:#4b6080;font-size:0.8rem;margin-bottom:1rem;">
        ✅ Load Audio → ✅ Noise Reduction → ✅ Normalization
        → ✅ Mel-Spectrogram → ✅ AI Model → ✅ Classification
        </div>
        """, unsafe_allow_html=True)

        render_result_banner(result)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown('<div class="biosound-card">', unsafe_allow_html=True)
            render_top_predictions(result)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="biosound-card">', unsafe_allow_html=True)
            render_audio_info(result)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="biosound-card">', unsafe_allow_html=True)
        render_visualizations(result)
        st.markdown("</div>", unsafe_allow_html=True)

    render_model_performance()

    with st.expander("ℹ️ About BioSound AI"):
        st.markdown("""
        ## BioSound AI – Living Being Sound Recognition

        BioSound AI is a **Multiclass Bioacoustic Sound Classification System** that identifies
        the type of living being that produced an audio input.
        """)

if __name__ == "__main__":
    main()
