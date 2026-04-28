import os
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn

from transformers import WavLMModel, Wav2Vec2FeatureExtractor


INFERENCE_CONFIG = {
    "model_name": "microsoft/wavlm-base-plus",
    "pooling": "mean",
    "sample_rate": 16000,
    "max_duration": 10.0,
    "emotions": ["happy", "sad", "angry", "neutral"],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_path": os.path.join("checkpoints", "wavlm_emotion_ravdess_final.pth"),
    "confidence_threshold": 0.65,
}


class WavLMForEmotionClassification(nn.Module):
    def __init__(self, model_name: str, num_emotions: int, pooling: str = "mean"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.pooling = pooling

        hidden_size = self.wavlm.config.hidden_size
        classifier_input_size = hidden_size * 2 if pooling == "both" else hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wavlm(input_values).last_hidden_state

        if self.pooling == "mean":
            pooled = torch.mean(hidden_states, dim=1)
        elif self.pooling == "std":
            pooled = torch.std(hidden_states, dim=1)
        else:
            pooled = torch.cat(
                [
                    torch.mean(hidden_states, dim=1),
                    torch.std(hidden_states, dim=1),
                ],
                dim=1,
            )

        return self.classifier(pooled)


_model = None
_feature_extractor = None


def load_emotion_model():
    global _model, _feature_extractor

    if _model is not None and _feature_extractor is not None:
        return _model, _feature_extractor

    device = INFERENCE_CONFIG["device"]
    emotions = INFERENCE_CONFIG["emotions"]
    checkpoint_path = INFERENCE_CONFIG["checkpoint_path"]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        INFERENCE_CONFIG["model_name"]
    )

    model = WavLMForEmotionClassification(
        model_name=INFERENCE_CONFIG["model_name"],
        num_emotions=len(emotions),
        pooling=INFERENCE_CONFIG["pooling"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model = model
    return _model, _feature_extractor


def predict_emotion(audio_path: str) -> dict:
    model, feature_extractor = load_emotion_model()

    sample_rate = INFERENCE_CONFIG["sample_rate"]
    max_length = int(INFERENCE_CONFIG["max_duration"] * sample_rate)
    emotions = INFERENCE_CONFIG["emotions"]
    device = INFERENCE_CONFIG["device"]

    try:
        audio, sr = sf.read(audio_path)
    except Exception:
        audio, sr = librosa.load(audio_path, sr=None)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)))

    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        logits = model(input_values)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    predicted_emotion = emotions[pred_idx]
    confidence = float(probs[pred_idx])

    print("\n=== DEBUG ===")
    print("Probabilities:", probs)
    print("Predicted index:", pred_idx)
    print("Emotion mapping:", INFERENCE_CONFIG["emotions"])
    print("Chosen emotion:", predicted_emotion)
    print("Confidence:", confidence)
    print("=============\n")

    final_emotion = (
        predicted_emotion
        if confidence >= INFERENCE_CONFIG["confidence_threshold"]
        else "neutral"
    )

    return {
        "predicted_emotion": predicted_emotion,
        "final_emotion": final_emotion,
        "confidence": confidence,
        "all_probabilities": {e: float(p) for e, p in zip(emotions, probs)},
    }


def predict_emotions_for_chunks(chunk_paths: list[str], chunk_timeline: list[dict]) -> list[dict]:
    """
    Returns Unreal-friendly chunks.
    Each chunk has start_time and end_time so Unreal can trigger facial animation
    at the correct moment in the audio timeline.
    """
    results = []

    for path, timeline in zip(chunk_paths, chunk_timeline):
        pred = predict_emotion(path)

        start_time = round(float(timeline["start_s"]), 2)
        end_time = round(float(timeline["end_s"]), 2)
        confidence = round(float(pred["confidence"]), 4)

        results.append({
            "chunk_index": int(timeline.get("chunk_index", len(results) + 1)),
            "start_time": start_time,
            "end_time": end_time,
            "duration": round(end_time - start_time, 2),
            "emotion": pred["final_emotion"],
            "raw_emotion": pred["predicted_emotion"],
            "confidence": confidence,
            "scores": pred["all_probabilities"],
            "chunk_path": path,
        })

    return results

# to test the inference logic locally, you can run this file directly with a test audio path
# run this command; pyton inference.py
#uncomment the following block 

# if __name__ == "__main__":
#     test_audio = "test_audio/test.wav"
#     result = predict_emotion(test_audio)
#     print(result)
