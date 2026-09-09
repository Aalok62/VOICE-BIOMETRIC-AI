# 🧠 BioSound AI
## Multiclass Living Being Sound Recognition System

BioSound AI is a complete AI-powered web application that classifies audio recordings
and identifies **which type of living being** produced the sound.

> **Supported classes:** Human · Dog · Cat · Bird · Cow · Horse · Chicken · Sheep

---

## 🚀 Quick Start (No Training Required)

The application ships ready-to-use via **Audio Spectrogram Transformer (AST)** — a state-of-the-art PyTorch pre-trained audio model (Hugging Face).

### 1. Navigate to the project directory
```bash
cd C:\Users\aalok\.gemini\antigravity\scratch\BioSoundAI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501` in your browser.

---

## 🎙️ How to Use

### Live Recording
1. Click the **"🎤 Live Recording"** tab
2. Click **"Record Sound"** — speak or make an animal sound
3. Click **"🔍 Analyse Recording"**
4. See the result: `🧑 HUMAN — Confidence: 96.8%`

### Upload Audio File
1. Click the **"📂 Upload Audio File"** tab
2. Upload a `.wav`, `.mp3`, `.flac`, or `.m4a` file
3. Click **"🔍 Analyse File"**
4. View predictions and visualizations

---

## 📊 Result Dashboard

The dashboard shows:
- **Detected class** with emoji and confidence badge (🟢 HIGH / 🟠 MEDIUM / 🔴 LOW)
- **Top predictions** bar chart (all 8 classes)
- **Audio waveform** visualization
- **Mel-spectrogram** (frequency vs. time)
- **Audio metadata** (duration, sample rate, channels)
- **Model performance** section

---

## 🏋️ Train a Custom Model

If you have your own labeled audio dataset, you can train the custom PyTorch CNN.

### Dataset Structure
```
data/raw/
├── human/       (*.wav, *.mp3, *.flac, *.m4a)
├── dog/
├── cat/
├── bird/
├── cow/
├── horse/
├── chicken/
└── sheep/
```

### Train & Evaluate
```bash
python src/model/train.py
python src/model/evaluate.py
```

In the app sidebar, select **"Custom BioSoundCNN"** from the Model dropdown to use your trained weights!

---

## 🔬 Run Tests
```bash
python -m pytest tests/ -v
```

All 44 test cases pass! ✅
