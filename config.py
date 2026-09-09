# =============================================================================
# config.py - BioSound AI Central Configuration
# =============================================================================
# All tuneable constants live here. Edit this file to add new classes,
# change model paths, or adjust audio preprocessing parameters.

import os

# ---------------------------------------------------------------------------
# SUPPORTED LIVING-BEING CLASSES
# Add new class names here to extend the system.
# ---------------------------------------------------------------------------
SUPPORTED_CLASSES = [
    "human",
    "dog",
    "cat",
    "bird",
    "cow",
    "horse",
    "chicken",
    "sheep",
]

# Class display labels (capitalised, user-facing)
CLASS_LABELS = {cls: cls.capitalize() for cls in SUPPORTED_CLASSES}

# Emoji per class for the result dashboard
CLASS_EMOJI = {
    "human":   "🧑",
    "dog":     "🐕",
    "cat":     "🐈",
    "bird":    "🐦",
    "cow":     "🐄",
    "horse":   "🐴",
    "chicken": "🐔",
    "sheep":   "🐑",
}

# Badge color per class
CLASS_COLOR = {
    "human":   "#2196F3",
    "dog":     "#FF9800",
    "cat":     "#9C27B0",
    "bird":    "#4CAF50",
    "cow":     "#795548",
    "horse":   "#F44336",
    "chicken": "#FFC107",
    "sheep":   "#607D8B",
}

# ---------------------------------------------------------------------------
# AUDIO PREPROCESSING
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16_000      # Hz - required by AST; good for CNNs
SEGMENT_DURATION   = 3.0         # seconds - clip length for inference
MIN_AUDIO_DURATION = 0.5         # seconds - shorter clips are rejected
MAX_AUDIO_DURATION = 30.0        # seconds - clips longer than this are trimmed
SILENCE_THRESHOLD  = 0.01        # RMS below this -> considered silent

# ---------------------------------------------------------------------------
# MEL-SPECTROGRAM PARAMETERS
# ---------------------------------------------------------------------------
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048
FMIN        = 20.0
FMAX        = 8_000.0

# ---------------------------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------------------------
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR         = os.path.join(BASE_DIR, "models", "sound_classifier")
CUSTOM_MODEL_PATH  = os.path.join(MODELS_DIR, "biosound_cnn.pth")
LABEL_MAP_PATH     = os.path.join(MODELS_DIR, "label_map.json")
METRICS_CACHE_PATH = os.path.join(MODELS_DIR, "metrics_cache.json")

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
DATASET_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_DIR      = os.path.join(BASE_DIR, "data", "test")

TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10   # Early stopping patience (epochs)
NUM_WORKERS   = 0    # DataLoader workers (0 = main process, safe on Windows)

# ---------------------------------------------------------------------------
# INFERENCE SETTINGS
# ---------------------------------------------------------------------------
CONFIDENCE_HIGH   = 0.70  # >= 70% -> green badge
CONFIDENCE_MEDIUM = 0.45  # >= 45% -> orange badge; below -> red badge
TOP_K_PREDICTIONS = 5     # Number of top predictions to display

# ---------------------------------------------------------------------------
# MODEL SELECTION
# "ast"    -> use Audio Spectrogram Transformer (PyTorch / HuggingFace, no training required)
# "custom" -> use trained BioSoundCNN (requires running train.py first)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "ast"

# AST Model ID on Hugging Face Hub
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

# ---------------------------------------------------------------------------
# AUDIOSET CLASS MAPPING
# Maps AudioSet class names -> our supported classes.
# ---------------------------------------------------------------------------
AUDIOSET_CLASS_MAP = {
    # Human
    "Speech": "human",
    "Male speech, man speaking": "human",
    "Female speech, woman speaking": "human",
    "Child speech, kid speaking": "human",
    "Conversation": "human",
    "Narration, monologue": "human",
    "Babbling": "human",
    "Shout": "human",
    "Yell": "human",
    "Screaming": "human",
    "Whispering": "human",
    "Laughter": "human",
    "Crying, sobbing": "human",
    "Singing": "human",
    "Humming": "human",
    "Groan": "human",
    "Grunt": "human",
    "Cough": "human",
    "Sneeze": "human",
    "Breathing": "human",
    "Snoring": "human",
    "Gasp": "human",
    "Hiccup": "human",
    # Dog
    "Dog": "dog",
    "Bark": "dog",
    "Yip": "dog",
    "Howl": "dog",
    "Bow-wow": "dog",
    "Growling": "dog",
    "Whimper (dog)": "dog",
    # Cat
    "Cat": "cat",
    "Purr": "cat",
    "Meow": "cat",
    "Hiss": "cat",
    "Caterwaul": "cat",
    # Bird
    "Bird": "bird",
    "Bird vocalization, bird call, bird song": "bird",
    "Chirp, tweet": "bird",
    "Squawk": "bird",
    "Pigeon, dove": "bird",
    "Duck": "bird",
    "Crow": "bird",
    "Owl": "bird",
    "Turkey": "bird",
    "Goose": "bird",
    "Parrot": "bird",
    "Cuckoo": "bird",
    "Canary": "bird",
    # Cow
    "Cattle, bovine": "cow",
    "Moo": "cow",
    # Horse
    "Horse": "horse",
    "Neigh, whinny": "horse",
    # Chicken
    "Chicken, rooster": "chicken",
    "Cluck": "chicken",
    "Crowing, cock-a-doodle-doo": "chicken",
    "Rooster": "chicken",
    # Sheep
    "Sheep": "sheep",
    "Bleat": "sheep",
}
# Backward compatibility for YAMNet mapping name
YAMNET_CLASS_MAP = AUDIOSET_CLASS_MAP
