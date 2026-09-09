"""
src/utils/helpers.py
====================
Utility functions: logging, file I/O, error formatting, temp file management.
"""

import os
import sys
import io
import tempfile
import logging
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str = "biosound", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a named logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s – %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

VALID_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def validate_file_extension(filename: str) -> bool:
    """Return True if the file extension is a supported audio format."""
    ext = Path(filename).suffix.lower()
    return ext in VALID_AUDIO_EXTENSIONS


def bytes_to_tempfile(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """
    Write audio bytes to a temporary file and return the file path.
    The caller is responsible for deleting the file when done.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


def cleanup_tempfile(path: str):
    """Delete a temporary file if it exists."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_confidence(prob: float) -> str:
    """Format a probability as a percentage string, e.g. '96.8%'."""
    return f"{prob * 100:.1f}%"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs    = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def get_confidence_badge(prob: float) -> tuple:
    """
    Return (color, label) based on confidence level.

    Returns
    -------
    (color_hex: str, label: str)
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM

    if prob >= CONFIDENCE_HIGH:
        return "#4CAF50", "HIGH"       # green
    elif prob >= CONFIDENCE_MEDIUM:
        return "#FF9800", "MEDIUM"    # orange
    else:
        return "#F44336", "LOW"       # red


# ---------------------------------------------------------------------------
# Audio info formatting
# ---------------------------------------------------------------------------

def audio_info_to_markdown(info: dict) -> str:
    """Format audio_info dict into a markdown table string."""
    rows = [
        ("Filename",    info.get("filename",    "N/A")),
        ("Duration",    format_duration(info.get("duration_s", 0))),
        ("Sample Rate", f"{info.get('sample_rate', 'N/A')} Hz"),
        ("Channels",    str(info.get("channels",    1))),
        ("Peak",        f"{info.get('peak', 0):.4f}"),
        ("RMS",         f"{info.get('rms',  0):.4f}"),
    ]
    header = "| Property | Value |\n|---|---|\n"
    body   = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return header + body


# ---------------------------------------------------------------------------
# Prediction formatting
# ---------------------------------------------------------------------------

def prediction_to_summary(result: dict) -> str:
    """Format a prediction result dict into a printable summary string."""
    cls  = result.get("predicted_class", "unknown").upper()
    conf = format_confidence(result.get("confidence", 0))
    top  = result.get("top_k", [])
    lines = [
        f"Detected : {cls}",
        f"Confidence: {conf}",
        "",
        "Top Predictions:",
    ]
    for name, prob in top:
        bar = "█" * int(prob * 20)
        lines.append(f"  {name.capitalize():<10} {bar:<20} {prob*100:.1f}%")
    return "\n".join(lines)
