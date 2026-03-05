"""
04_audio_workflow.py
--------------------
Demonstrates the multimodal audio annotation workflow:

  1. Extracts audio track from a demo video
  2. Runs a simple speech/silence detector to identify speech segments
  3. Detects likely language (placeholder — shows where Whisper/ASR plugs in)
  4. Prints a structured annotation plan showing what annotators would see
  5. Shows how audio labels link back to video timestamps in Encord

For production, swap the placeholder ASR with:
  - OpenAI Whisper (open source, multilingual)
  - AssemblyAI / Deepgram / Azure Speech (API)

Usage:
    export ENCORD_SSH_KEY_PATH=~/.ssh/id_ed25519
    python scripts/04_audio_workflow.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "demo_data"

# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video using ffmpeg. Returns True on success."""
    if audio_path.exists():
        print(f"  ↳ Audio already extracted: {audio_path.name}")
        return True

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn",                     # no video
                "-acodec", "pcm_s16le",    # raw PCM
                "-ar", "16000",            # 16kHz (Whisper standard)
                "-ac", "1",               # mono
                str(audio_path),
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0:
            print(f"  ✓ Extracted audio: {audio_path.name}")
            return True
        else:
            print(f"  ✗ ffmpeg error: {result.stderr.decode()[:200]}")
            return False
    except FileNotFoundError:
        print("  ✗ ffmpeg not found. Install with: brew install ffmpeg")
        return False
    except Exception as e:
        print(f"  ✗ Audio extraction failed: {e}")
        return False


def read_wav_simple(wav_path: Path) -> tuple:
    """Read a WAV file without scipy dependency. Returns (samples, sample_rate)."""
    import wave
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sample_rate


# ---------------------------------------------------------------------------
# Simple speech/silence detector (energy-based VAD)
# ---------------------------------------------------------------------------

def detect_speech_segments(
    wav_path: Path,
    frame_ms: int = 30,
    energy_threshold_percentile: float = 35.0,
    min_speech_duration_s: float = 0.5,
) -> list:
    """
    Very simple energy-based voice activity detection.
    Returns list of (start_s, end_s, energy_db) dicts.

    In production replace with Whisper or a proper VAD model.
    """
    try:
        samples, sr = read_wav_simple(wav_path)
    except Exception as e:
        print(f"  ✗ Could not read audio: {e}")
        return []

    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(samples) // frame_len

    energies = []
    for i in range(n_frames):
        frame = samples[i * frame_len : (i + 1) * frame_len]
        rms = np.sqrt(np.mean(frame ** 2)) + 1e-10
        energies.append(20 * np.log10(rms))  # dBFS

    threshold = np.percentile(energies, energy_threshold_percentile)
    is_speech = [e > threshold for e in energies]

    # Merge into segments
    segments = []
    in_speech = False
    seg_start = 0

    for i, speech in enumerate(is_speech):
        t = i * frame_ms / 1000
        if speech and not in_speech:
            seg_start = t
            in_speech = True
        elif not speech and in_speech:
            duration = t - seg_start
            if duration >= min_speech_duration_s:
                avg_energy = np.mean(energies[int(seg_start * 1000 / frame_ms) : i])
                segments.append({
                    "start_s": round(seg_start, 2),
                    "end_s": round(t, 2),
                    "duration_s": round(duration, 2),
                    "energy_db": round(float(avg_energy), 1),
                })
            in_speech = False

    return segments


# ---------------------------------------------------------------------------
# ASR placeholder (Whisper integration point)
# ---------------------------------------------------------------------------

def transcribe_segment(segment: dict, wav_path: Path) -> dict:
    """
    Placeholder for ASR transcription.
    In production: pipe audio segment through Whisper or cloud ASR.
    """
    # Simulate realistic outputs for demo purposes
    demo_transcriptions = [
        {"text": "Hi, I have a delivery for this address.", "language": "english", "confidence": 0.96},
        {"text": "Muchas gracias, aquí lo dejo.", "language": "spanish", "confidence": 0.92},
        {"text": "Delivery confirmation scan complete.", "language": "english", "confidence": 0.98},
        {"text": "[doorbell rings]", "language": "en", "confidence": 1.0},
        {"text": "Your order is here, have a great day!", "language": "english", "confidence": 0.95},
    ]
    import random
    random.seed(int(segment["start_s"] * 100))
    result = random.choice(demo_transcriptions)
    return {**segment, **result}


# ---------------------------------------------------------------------------
# Format annotation plan for demo output
# ---------------------------------------------------------------------------

def print_audio_annotation_plan(video_name: str, segments: list, transcribed: list):
    print(f"\n{'='*65}")
    print(f"  AUDIO ANNOTATION PLAN: {video_name}")
    print(f"{'='*65}")
    print(f"  Speech segments detected: {len(segments)}")
    print()

    if not segments:
        print("  No speech detected (ambient/background audio only).")
        print("  → Tag as: Audio Scene = outdoor_urban_noise | Speech Presence = no_speech")
        return

    for i, seg in enumerate(transcribed, 1):
        lang = seg.get("language", "unknown")
        text = seg.get("text", "[unintelligible]")
        conf = seg.get("confidence", 0)
        duration = seg["duration_s"]

        print(f"  Segment {i}: {seg['start_s']}s → {seg['end_s']}s  ({duration}s)")
        print(f"    Language detected : {lang.upper()}")
        print(f"    ASR transcript    : \"{text}\"")
        print(f"    Confidence        : {conf:.0%}")
        print(f"    Annotation tasks  :")
        print(f"      → Speaker role  : [annotator labels]")
        print(f"      → Transcription : verify/correct ASR output")
        if lang != "english":
            print(f"      → Translation   : English translation required")
        print()

    print("  ENCORD PLATFORM LINKS:")
    print("  → Video + waveform synchronized view (Wavesurfer)")
    print("  → Audio segments highlighted on timeline")
    print("  → Cross-modal: speech timestamps linked to video frames")
    print("  → Export format: WebVTT / SRT / JSON with frame-level sync")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== ENCORD MULTIMODAL AUDIO ANNOTATION DEMO ===\n")

    audio_dir = DATA_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    video_files = list(DATA_DIR.glob("*.mp4"))
    if not video_files:
        print("No video files found in demo_data/.")
        print("Run python scripts/02_upload_demo_data.py first.")
        return

    all_results = []

    for vf in sorted(video_files):
        print(f"\nProcessing: {vf.name}")
        wav_path = audio_dir / (vf.stem + ".wav")

        # Step 1: Extract audio
        ok = extract_audio(vf, wav_path)
        if not ok:
            print(f"  Skipping audio analysis for {vf.name}")
            continue

        # Step 2: Detect speech segments
        print("  Running voice activity detection...")
        segments = detect_speech_segments(wav_path)
        print(f"  Found {len(segments)} speech segment(s)")

        # Step 3: ASR (placeholder)
        print("  Running ASR transcription (placeholder)...")
        transcribed = [transcribe_segment(s, wav_path) for s in segments]

        # Step 4: Print annotation plan
        print_audio_annotation_plan(vf.name, segments, transcribed)

        all_results.append({
            "video": vf.name,
            "segments": transcribed,
        })

    # Save results
    output_path = PROJECT_DIR / "audio_annotation_plan.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Annotation plan saved to: {output_path}")

    print("\n--- HOW THIS SCALES (DEMO TALKING POINTS) ---")
    print("1. Whisper (open source) handles 40+ languages out of the box")
    print("2. ASR pre-labeling cuts transcription cost by ~70%")
    print("3. Human annotators verify + correct, not transcribe from scratch")
    print("4. Synchronized video+audio view catches mis-aligned transcriptions")
    print("5. Niche language routing: flag segments for specialist annotators")
    print("6. Export directly to HuggingFace, JSONL, or lab-specific formats")


if __name__ == "__main__":
    main()
