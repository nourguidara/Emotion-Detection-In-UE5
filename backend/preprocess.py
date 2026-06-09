# backend/preprocess.py
import os
import time
import glob
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import soundfile as sf

import torch
import torchaudio


CONFIG: Dict[str, Any] = {
    "supported_ext": {".wav"},
    "target_sr": 16000,
    "force_mono": True,

    "min_duration_s": 1.0,

    # Chunking
    "max_chunk_s": 6.0,
    "merge_last_chunk_if_shorter_than_s": 2.0,
    "cut_search_ms": 600,   # search for a quieter cut near target

    # Denoise
    "do_denoise": True,
    "denoise_method": "spectral_gate",   # or "highpass"
    "highpass_hz": 60,

    # Speech-aware chunking (SOFT)
    "do_speech_segment": True,
    "frame_ms": 30,
    "hop_ms": 10,
    "merge_gap_ms": 700,      # merge nearby speech parts
    "pad_ms": 300,            # preserve context around speech
    "energy_threshold_db": 12.0,  # softer than before

    # Chunk post-processing
    "postprocess_chunks": True,
    "fade_ms": 8,
    "peak_target": 0.95,
    "min_peak_for_norm": 0.02,

    # Disable aggressive dropping for now
    "disable_chunk_filter": True,

    # Debug
    "debug_store": True,
    "debug_dir": "./debug",
    "max_debug_saves": 300,
}

os.makedirs(CONFIG["debug_dir"], exist_ok=True)


ERROR_NO_FILE = "NO_FILE"
ERROR_UNSUPPORTED_EXT = "UNSUPPORTED_EXT"
ERROR_AUDIO_LOAD = "AUDIO_LOAD_FAILED"
ERROR_AUDIO_TOO_SHORT = "AUDIO_TOO_SHORT"
ERROR_TOO_LITTLE_SPEECH = "TOO_LITTLE_SPEECH"


@dataclass
class Result:
    ok: bool
    message: str
    error_code: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    @staticmethod
    def success(message: str, payload: Optional[Dict[str, Any]] = None) -> "Result":
        return Result(ok=True, message=message, payload=payload)

    @staticmethod
    def fail(message: str, error_code: str, payload: Optional[Dict[str, Any]] = None) -> "Result":
        return Result(ok=False, message=message, error_code=error_code, payload=payload)


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _duration_s(y: np.ndarray, sr: int) -> float:
    return float(len(y) / sr) if sr else 0.0


def _cleanup_old_debug_files(debug_dir: str, max_keep: int) -> None:
    files = sorted(glob.glob(os.path.join(debug_dir, "*.wav")), key=os.path.getmtime, reverse=True)
    for f in files[max_keep:]:
        try:
            os.remove(f)
        except Exception:
            pass


def _resample_np(y: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if sr == target_sr:
        return y, sr
    t = torch.from_numpy(y).to(torch.float32)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    y2 = resampler(t.unsqueeze(0)).squeeze(0).cpu().numpy()
    return y2, target_sr


def _highpass_np(y: np.ndarray, sr: int, hz: int) -> np.ndarray:
    t = torch.from_numpy(y).to(torch.float32)
    y2 = torchaudio.functional.highpass_biquad(t, sr, cutoff_freq=float(hz))
    return y2.cpu().numpy()


def _spectral_gate_np(y: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(y).to(torch.float32)
    n_fft = 1024
    hop = 256
    win = torch.hann_window(n_fft)

    stft = torch.stft(
        t,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        return_complex=True,
    )
    mag = stft.abs()
    phase = torch.angle(stft)

    noise = torch.quantile(mag, 0.10, dim=1, keepdim=True)
    thr = noise * 2.0

    mask = (mag - thr) / (mag + 1e-8)
    mask = torch.clamp(mask, 0.0, 1.0)

    mag_d = mag * mask
    stft_d = mag_d * torch.exp(1j * phase)
    y_out = torch.istft(
        stft_d,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        length=t.numel(),
    )
    return y_out.cpu().numpy()


# -----------------------------
# Chunk post-processing
# -----------------------------
def _fade(y: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
    n = int(sr * fade_ms / 1000)
    if n <= 0 or len(y) < 2 * n:
        return y
    y2 = y.copy()
    y2[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    y2[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return y2


def _peak_normalize(y: np.ndarray, peak_target: float, min_peak_for_norm: float) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-9 or peak < float(min_peak_for_norm):
        return y
    gain = float(peak_target) / peak
    return np.clip(y * gain, -1.0, 1.0).astype(np.float32)


def postprocess_chunk(y: np.ndarray, sr: int, cfg: Dict[str, Any]) -> np.ndarray:
    y2 = (y - float(np.mean(y))).astype(np.float32)
    y2 = _fade(y2, sr, int(cfg.get("fade_ms", 8)))
    y2 = _peak_normalize(
        y2,
        peak_target=float(cfg.get("peak_target", 0.95)),
        min_peak_for_norm=float(cfg.get("min_peak_for_norm", 0.02)),
    )
    return y2


# -----------------------------
# VAD helpers
# -----------------------------
def _frame_audio(y: np.ndarray, sr: int, frame_ms: int, hop_ms: int):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    if frame_len <= 0 or hop_len <= 0 or len(y) < frame_len:
        return None, None, None

    num_frames = 1 + (len(y) - frame_len) // hop_len
    frames = np.stack([y[i * hop_len: i * hop_len + frame_len] for i in range(num_frames)])
    starts = np.array([i * hop_len for i in range(num_frames)], dtype=int)
    ends = starts + frame_len
    return frames, starts, ends


def _db_rms(frames: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rms = np.sqrt(np.mean(frames**2, axis=1) + eps)
    return 20.0 * np.log10(rms + eps)


def _mask_to_segments(mask: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    in_seg = False
    s0 = 0
    for i, voiced in enumerate(mask):
        if voiced and not in_seg:
            in_seg = True
            s0 = int(starts[i])
        elif (not voiced) and in_seg:
            in_seg = False
            segs.append((s0, int(ends[i - 1])))
    if in_seg:
        segs.append((s0, int(ends[-1])))
    return segs


def _merge_segments(segs: List[Tuple[int, int]], sr: int, merge_gap_ms: int) -> List[Tuple[int, int]]:
    if not segs:
        return segs
    gap = int(sr * merge_gap_ms / 1000)
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _pad_segments(segs: List[Tuple[int, int]], sr: int, total_len: int, pad_ms: int) -> List[Tuple[int, int]]:
    pad = int(sr * pad_ms / 1000)
    return [(max(0, s - pad), min(total_len, e + pad)) for s, e in segs]


def speech_segments_energy_vad(y: np.ndarray, sr: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    frames, starts, ends = _frame_audio(y, sr, cfg["frame_ms"], cfg["hop_ms"])
    if frames is None:
        return {"speech_segments": [], "note": "audio too short for framing"}

    db = _db_rms(frames)
    noise_floor = float(np.percentile(db, 10))
    thr = noise_floor + float(cfg["energy_threshold_db"])

    segs = _mask_to_segments(db >= thr, starts, ends)
    segs = _merge_segments(segs, sr, cfg["merge_gap_ms"])
    segs = _pad_segments(segs, sr, len(y), cfg["pad_ms"])

    if not segs:
        return {
            "speech_segments": [],
            "note": "no speech detected",
            "noise_floor_db": noise_floor,
            "threshold_db": thr,
        }

    return {
        "speech_segments": segs,
        "noise_floor_db": noise_floor,
        "threshold_db": thr,
    }


# -----------------------------
# Smart splitting inside long speech segments
# -----------------------------
def _best_cut_point_by_energy(y: np.ndarray, sr: int, center: int, search_ms: int) -> int:
    half = int(sr * search_ms / 1000)
    left = max(0, center - half)
    right = min(len(y), center + half)

    win = int(sr * 0.03)
    hop = int(sr * 0.01)
    if right - left < win or win <= 0 or hop <= 0:
        return center

    energies = []
    positions = []

    for p in range(left, right - win, hop):
        frame = y[p:p + win]
        energies.append(float(np.mean(frame * frame)))
        positions.append(p + win // 2)

    if not energies:
        return center

    return int(positions[int(np.argmin(energies))])


def _split_long_segment_soft(
    y: np.ndarray,
    sr: int,
    s: int,
    e: int,
    max_chunk_samples: int,
    search_ms: int,
) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    cur_s = s

    while (e - cur_s) > max_chunk_samples:
        target = cur_s + max_chunk_samples
        cut = _best_cut_point_by_energy(y, sr, center=target, search_ms=search_ms)

        # safety: avoid absurd early cut
        if cut <= cur_s + int(0.5 * max_chunk_samples):
            cut = target

        out.append((cur_s, cut))
        cur_s = cut

    out.append((cur_s, e))
    return out


def build_chunks_from_segments(
    y: np.ndarray,
    segs: List[Tuple[int, int]],
    sr: int,
    cfg: Dict[str, Any],
) -> List[List[Tuple[int, int]]]:
    max_chunk_samples = int(sr * float(cfg["max_chunk_s"]))
    cut_search_ms = int(cfg.get("cut_search_ms", 600))
    merge_last_shorter_than = int(sr * float(cfg.get("merge_last_chunk_if_shorter_than_s", 0.0)))

    expanded: List[Tuple[int, int]] = []
    for s, e in segs:
        if (e - s) > max_chunk_samples:
            expanded.extend(_split_long_segment_soft(y, sr, s, e, max_chunk_samples, cut_search_ms))
        else:
            expanded.append((s, e))

    # merge tiny last chunk into previous
    if len(expanded) >= 2 and merge_last_shorter_than > 0:
        last_s, last_e = expanded[-1]
        if (last_e - last_s) < merge_last_shorter_than:
            prev_s, prev_e = expanded[-2]
            expanded[-2] = (prev_s, last_e)
            expanded.pop()

    chunks: List[List[Tuple[int, int]]] = [[(s, e)] for s, e in expanded]
    return chunks


def concat_segments(y: np.ndarray, segs: List[Tuple[int, int]]) -> np.ndarray:
    parts = [y[s:e] for s, e in segs if e > s]
    return np.concatenate(parts) if parts else y


def save_wav_chunks(
    y: np.ndarray,
    sr: int,
    chunks: List[List[Tuple[int, int]]],
    out_dir: str,
    base_name: str,
    cfg: Dict[str, Any],
) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")

    out_paths: List[str] = []
    durations: List[float] = []
    chunk_timeline: List[Dict[str, float]] = []

    for i, ch in enumerate(chunks, start=1):
        y_chunk = concat_segments(y, ch)

        if cfg.get("postprocess_chunks", True):
            y_chunk = postprocess_chunk(y_chunk, sr, cfg)

        start_sample = ch[0][0]
        end_sample = ch[-1][1]

        start_s = float(start_sample / sr)
        end_s = float(end_sample / sr)
        duration_s = float(len(y_chunk) / sr)

        out_path = os.path.join(out_dir, f"{base_name}_chunk{i:02d}_{ts}.wav")
        sf.write(out_path, y_chunk, sr)

        out_paths.append(out_path)
        durations.append(duration_s)
        chunk_timeline.append(
            {
                "chunk_index": i,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": duration_s,
            }
        )

    return out_paths, durations, chunk_timeline


# =============================
# Main preprocessing
# =============================
def backend_preprocess(path: str) -> Result:
    actions: List[str] = []

    if not path or not os.path.exists(path):
        return Result.fail("File not found.", ERROR_NO_FILE, payload={"path": path})

    if _ext(path) not in CONFIG["supported_ext"]:
        return Result.fail(
            "Unsupported audio extension (WAV only for this project).",
            ERROR_UNSUPPORTED_EXT,
            payload={"ext": _ext(path), "supported": sorted(list(CONFIG["supported_ext"]))},
        )

    try:
        y2d, sr = sf.read(path, dtype="float32", always_2d=True)
        actions.append("load_audio:soundfile")
    except Exception as e:
        return Result.fail("Failed to load audio.", ERROR_AUDIO_LOAD, payload={"exception": str(e)})

    if CONFIG.get("force_mono", True):
        y = y2d.mean(axis=1) if y2d.shape[1] > 1 else y2d[:, 0]
        if y2d.shape[1] > 1:
            actions.append("stereo_to_mono")
    else:
        y = y2d[:, 0]

    original_duration_s = _duration_s(y, sr)

    if original_duration_s < float(CONFIG["min_duration_s"]):
        return Result.fail(
            "Audio too short.",
            ERROR_AUDIO_TOO_SHORT,
            payload={"duration_s": original_duration_s, "min_duration_s": CONFIG["min_duration_s"]},
        )

    target_sr = int(CONFIG.get("target_sr", 16000))
    if sr != target_sr:
        y, sr = _resample_np(y, sr, target_sr)
        actions.append(f"resample:{target_sr}")

    if CONFIG.get("do_denoise", True):
        if CONFIG.get("denoise_method") == "highpass":
            y = _highpass_np(y, sr, int(CONFIG.get("highpass_hz", 60)))
            actions.append(f"denoise:highpass@{int(CONFIG.get('highpass_hz', 60))}Hz")
        else:
            y = _spectral_gate_np(y)
            actions.append("denoise:spectral_gate")

    speech_meta = None
    segs: List[Tuple[int, int]] = []

    if CONFIG.get("do_speech_segment", True):
        speech_meta = speech_segments_energy_vad(y, sr, CONFIG)
        segs = speech_meta.get("speech_segments", [])
        actions.append("speech_segment:energy_vad")

    # fallback: if VAD finds nothing, use full audio
    if not segs:
        segs = [(0, len(y))]
        if speech_meta is None:
            speech_meta = {"speech_segments": segs, "note": "segmentation disabled; used full audio"}
        else:
            speech_meta["note"] = (speech_meta.get("note", "") + " | fallback: used full audio").strip()

    chunks = build_chunks_from_segments(y, segs, sr, CONFIG)
    actions.append(f"chunking:speech_aware_max_chunk_s={float(CONFIG['max_chunk_s'])}")

    # no aggressive filtering for now
    dropped = {"too_short": 0, "no_speech": 0}
    actions.append("chunk_filter:disabled")

    if not chunks:
        return Result.fail(
            "No valid chunks produced.",
            ERROR_TOO_LITTLE_SPEECH,
            payload={"speech_meta": speech_meta, "dropped": dropped},
        )

    debug_wav_paths: List[str] = []
    chunk_durations: List[float] = []
    chunk_timeline: List[Dict[str, float]] = []

    if CONFIG.get("debug_store", True):
        base = os.path.splitext(os.path.basename(path))[0]
        debug_wav_paths, chunk_durations, chunk_timeline = save_wav_chunks(
            y, sr, chunks, CONFIG["debug_dir"], base, CONFIG
        )
        actions.append(f"save_chunks:{len(debug_wav_paths)}")
        _cleanup_old_debug_files(CONFIG["debug_dir"], int(CONFIG.get("max_debug_saves", 300)))

    payload: Dict[str, Any] = {
        "actions": actions,
        "original_duration_s": float(original_duration_s),
        "final_sr": int(sr),
        "num_chunks": int(len(chunks)),
        "chunk_durations_s": chunk_durations,
        "chunk_timeline": chunk_timeline,
        "debug_wav_paths": debug_wav_paths,
        "speech_meta": speech_meta,
        "dropped_chunks": dropped,
        "used_config": {
            "max_chunk_s": float(CONFIG["max_chunk_s"]),
            "merge_last_chunk_if_shorter_than_s": float(CONFIG["merge_last_chunk_if_shorter_than_s"]),
            "cut_search_ms": int(CONFIG["cut_search_ms"]),
            "do_speech_segment": bool(CONFIG["do_speech_segment"]),
            "energy_threshold_db": float(CONFIG["energy_threshold_db"]),
            "merge_gap_ms": int(CONFIG["merge_gap_ms"]),
            "pad_ms": int(CONFIG["pad_ms"]),
            "do_denoise": bool(CONFIG["do_denoise"]),
            "postprocess_chunks": bool(CONFIG["postprocess_chunks"]),
        },
    }

    return Result.success(
        "Preprocessing completed successfully (soft speech-aware chunking).",
        payload=payload,
    )
