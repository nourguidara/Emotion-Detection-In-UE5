# Emotion Detection Backend

This backend is part of the ISS project. It processes audio files, detects emotions using an AI model, and returns results to be used in Unreal Engine.

---

## Project Structure

```
BACKEND/
│
├── app.py                 # FastAPI entry point
├── inference.py           # Emotion prediction logic
├── preprocess.py          # Audio preprocessing
├── requirements.txt       # Dependencies
│
├── checkpoints/           # Model files (.pth) - NOT included
├── test_audio/            # Sample audio files
├── uploaded_audio/        # Uploaded audio from API
└── debug/                 # Debug files
```

---

## Requirements

- Python 3.11

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Files

Model files are not included (too large).

You must manually add them in:

```
checkpoints/
```

Example:

```
checkpoints/wavlm_emotion_ravdess_final.pth
```

---

## Run the Backend

From the backend folder:

```bash
python -m uvicorn app:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

---

## Test the API

Open:

```
http://127.0.0.1:8000/docs
```

Use the FastAPI interface to upload and test audio files.

---

## Output Example

```json
{
  "emotion": "angry",
  "confidence": 0.87
}
```

---

## Notes

- Use `.wav` audio files only  
- Recommended sample rate: 16000 Hz  
- Low-confidence predictions may default to neutral  
- Designed to communicate with Unreal Engine via HTTP requests  
