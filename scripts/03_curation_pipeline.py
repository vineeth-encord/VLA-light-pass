"""
03_curation_pipeline.py
-----------------------
Demonstrates the data curation and quality filtering pipeline using the
Encord Python SDK. Shows:

  1. Listing all data units in the VLA project
  2. Computing a simple quality score per clip (brightness, blur, motion)
  3. Tagging clips with quality flags
  4. Printing a curation summary — "raw → curated" story

This is the DEMO SCRIPT for Act 1: Indexing & Curation.

Usage:
    export ENCORD_SSH_KEY_PATH=~/.ssh/id_ed25519
    python scripts/03_curation_pipeline.py
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from encord import EncordUserClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SSH_KEY_PATH = os.environ.get("ENCORD_SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_ed25519"))
PROJECT_DIR = Path(__file__).parent.parent
HASHES_FILE = PROJECT_DIR / "project_hashes.json"
DATA_DIR = PROJECT_DIR / "demo_data"

# Curation thresholds — tune these to tell the story you want
QUALITY_THRESHOLDS = {
    "min_brightness": 40,       # 0–255, below = too dark
    "max_brightness": 220,      # above = overexposed
    "min_sharpness": 80,        # Laplacian variance; below = blurry
    "min_motion_score": 5,      # frames with zero motion are probably static/useless
    "max_motion_score": 300,    # frames with extreme motion are too shaky
}

# ---------------------------------------------------------------------------
# Video quality analysis
# ---------------------------------------------------------------------------

def analyze_video(video_path: Path) -> dict:
    """
    Compute quality metrics for a video file.
    Returns a dict with per-video quality stats.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "Could not open video", "quality_score": 0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_s = total_frames / fps if fps > 0 else 0

    # Sample up to 30 frames evenly across the video
    sample_count = min(30, total_frames)
    sample_indices = set(
        int(i * total_frames / sample_count) for i in range(sample_count)
    )

    brightness_vals = []
    sharpness_vals = []
    motion_vals = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in sample_indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Brightness: mean pixel value
            brightness_vals.append(float(gray.mean()))

            # Sharpness: Laplacian variance (higher = sharper)
            sharpness_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            # Motion: mean absolute diff from previous sampled frame
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_vals.append(float(diff.mean()))
            prev_gray = gray

        frame_idx += 1

    cap.release()

    if not brightness_vals:
        return {"error": "No frames sampled", "quality_score": 0}

    avg_brightness = np.mean(brightness_vals)
    avg_sharpness = np.mean(sharpness_vals)
    avg_motion = np.mean(motion_vals) if motion_vals else 0
    p5_sharpness = np.percentile(sharpness_vals, 5)  # worst frames

    # Quality flags
    flags = []
    t = QUALITY_THRESHOLDS

    if avg_brightness < t["min_brightness"]:
        flags.append("LOW_LIGHT")
    if avg_brightness > t["max_brightness"]:
        flags.append("OVEREXPOSED")
    if p5_sharpness < t["min_sharpness"]:
        flags.append("BLURRY")
    if avg_motion < t["min_motion_score"]:
        flags.append("STATIC_FRAMES")
    if avg_motion > t["max_motion_score"]:
        flags.append("EXCESSIVE_SHAKE")

    # Composite quality score 0–100
    brightness_score = 100 - abs(avg_brightness - 128) / 128 * 100
    sharpness_score = min(100, p5_sharpness / 200 * 100)
    motion_score = min(100, max(0, (avg_motion - t["min_motion_score"]) /
                                   (t["max_motion_score"] - t["min_motion_score"]) * 100))
    quality_score = round(
        0.35 * brightness_score + 0.45 * sharpness_score + 0.20 * motion_score
    )

    return {
        "total_frames": total_frames,
        "duration_s": round(duration_s, 1),
        "fps": round(fps, 1),
        "avg_brightness": round(avg_brightness, 1),
        "avg_sharpness": round(avg_sharpness, 1),
        "avg_motion": round(avg_motion, 1),
        "quality_flags": flags,
        "quality_score": quality_score,
        "keep": len(flags) == 0 and quality_score >= 50,
    }


# ---------------------------------------------------------------------------
# Curation report
# ---------------------------------------------------------------------------

def print_curation_report(results: list):
    """Print a formatted curation summary for the demo."""
    total = len(results)
    kept = sum(1 for r in results if r.get("keep", False))
    rejected = total - kept

    print("\n" + "="*60)
    print("  DATA CURATION REPORT")
    print("="*60)
    print(f"  Total clips analyzed:   {total}")
    print(f"  Passed quality filter:  {kept}  ({kept/total*100:.0f}%)")
    print(f"  Filtered out:           {rejected}  ({rejected/total*100:.0f}%)")
    print()

    for r in results:
        status = "✓ KEEP" if r.get("keep") else "✗ FILTER"
        flags = ", ".join(r.get("quality_flags", [])) or "none"
        print(f"  {status}  [{r['quality_score']:3d}/100]  {r['name']}")
        print(f"            Duration: {r.get('duration_s', '?')}s | "
              f"Sharpness: {r.get('avg_sharpness', '?')} | "
              f"Flags: {flags}")

    print()
    print("  KEY MESSAGE FOR DEMO:")
    print(f"  → Only {kept} of {total} raw clips meet quality thresholds.")
    print(f"  → Without curation, {rejected/total*100:.0f}% of annotation spend is wasted.")
    print("  → Encord Active scales this to 100K+ hour datasets automatically.")
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HASHES_FILE.exists():
        print(f"ERROR: {HASHES_FILE} not found.")
        print("Run python scripts/01_setup_encord_project.py first.")
        return

    print("=== ENCORD DATA CURATION PIPELINE DEMO ===\n")
    print("Analyzing local demo clips for quality metrics...\n")

    # Analyze local video files
    video_files = list(DATA_DIR.glob("*.mp4"))
    if not video_files:
        print("No video files found in demo_data/.")
        print("Run python scripts/02_upload_demo_data.py first.")
        return

    results = []
    for vf in sorted(video_files):
        print(f"Analyzing: {vf.name}")
        metrics = analyze_video(vf)
        metrics["name"] = vf.name
        results.append(metrics)
        time.sleep(0.1)

    # Print curation report
    print_curation_report(results)

    # Save results
    output_path = PROJECT_DIR / "curation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

    # Encord platform tagging (optional — requires valid project)
    print("\n--- Encord Platform Integration ---")
    print("In production, these quality scores would be:")
    print("  1. Pushed to Encord Active as dataset-level metrics")
    print("  2. Used to auto-filter the project's dataset view")
    print("  3. Made available as search/filter criteria in the UI")
    print("  4. Exportable as JSON for downstream pipeline decisions")
    print("\nNext step: run  python scripts/04_audio_workflow.py")


if __name__ == "__main__":
    main()
