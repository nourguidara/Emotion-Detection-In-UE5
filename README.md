# 🎭 Emotion-Driven Character Animation using Speech Emotion Recognition and Unreal Engine 5

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Unreal Engine](https://img.shields.io/badge/Unreal%20Engine-5-black)
![AI](https://img.shields.io/badge/AI-Speech%20Emotion%20Recognition-orange)

## 📌 Project Overview

This project presents an **AI-powered Speech Emotion Recognition (SER) system** integrated with **Unreal Engine 5** to create an emotion-aware virtual character.

The system analyzes either an audio recording or spoken speech, predicts the speaker's emotional state, and dynamically drives character animations and visual effects inside Unreal Engine.


---

## 🎯 Objectives

- Detect emotions from speech using Deep Learning.
- Convert audio into emotional predictions.
- Communicate emotion results from Python backend to Unreal Engine 5.
- Dynamically animate a virtual character based on detected emotions.
- Create immersive visual feedback using facial expressions, character color changes, and environmental lighting.

---

## 🏗️ System Architecture

<img width="930" height="602" alt="image" src="https://github.com/user-attachments/assets/c380aeb9-6db6-4359-8757-7663fb2662ff" />


---

## ✨ Main Features

### 🎤 Emotion Detection

The backend predicts emotions from `.wav` audio files.

Supported emotions include:

- 😀 Happy
- 😢 Sad
- 😠 Angry
- 😐 Neutral

---

### 🎮 Unreal Engine Integration

The Unreal Engine client communicates with the AI backend through REST API calls.

Implemented features:

- Real-time emotion retrieval
- Backend status monitoring
- Emotion display panel
- Automatic character updates

---

### 🎭 Character Reactions

The virtual character reacts according to the predicted emotion through:

#### Prototype 1

- Character body color changes
- Environmental lighting color changes
- Backend integration
- Emotion visualization

#### Prototype 2

- Jaw synchronization (lip-sync)
- Facial animation framework
- Backend integration improvements
- Advanced emotion handling architecture
- Support for future facial expression expansion

---

## 🧠 AI Model Pipeline

### 1. Audio Preprocessing

The audio file is processed and transformed into features suitable for machine learning.

Examples of extracted features:

- MFCCs
- Spectral Features
- Temporal Features

### 2. Emotion Prediction

The trained Speech Emotion Recognition model predicts the most probable emotion.

### 3. API Response

The backend returns:

```json
{
    "predicted_emotion": "happy",
    "confidence": 0.93
}
```

### 4. Unreal Engine Response

Unreal Engine receives the prediction and updates:

- Character appearance
- Lighting environment
- Facial animations
- Lip synchronization

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/emotion-driven-character-animation.git
cd emotion-driven-character-animation
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Backend

```bash
python app.py
```

or

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

## 📡 API Example

### Predict Emotion

**POST**

```text
/api/predict
```

Upload a `.wav` file and receive:

```json
{
    "emotion": "angry",
    "confidence": 0.88
}
```

---

## 🎥 Demo Video

The complete demonstration video is available in:

```text
Demo/Demo_Video.mp4
```

The video showcases:

- Backend execution
- Emotion prediction
- Unreal Engine integration
- Prototype 1 visual effects
- Prototype 2 lip synchronization
- Character emotional reactions

---

## 📊 Presentation

The final project presentation is included in:

[EmotionDetectionPresentation_compressed.pdf](https://github.com/user-attachments/files/28760889/EmotionDetectionPresentation_compressed.pdf)

It contains:

- Project motivation
- Methodology
- System architecture
- Implementation details
- Results
- Demonstration screenshots

---

## 👩‍💻 Authors

- **Nour Guidara**

## 📜 License

This project was developed in collaboration with Lanterns Studio Company
.
