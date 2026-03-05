"""
02_upload_demo_data.py
----------------------
Stages local hand-tracking sample clips for the demo and uploads them
to the Encord datasets created in script 01.

Source clips:
  ~/Downloads/HandtrackingSamples/  — egocentric hand manipulation videos

Usage:
    export ENCORD_SSH_KEY_PATH=~/.ssh/id_ed25519
    python scripts/02_upload_demo_data.py
"""

import json
import os
import shutil
from pathlib import Path

from encord import EncordUserClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SSH_KEY_PATH = os.environ.get("ENCORD_SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_ed25519"))
PROJECT_DIR = Path(__file__).parent.parent
HASHES_FILE = PROJECT_DIR / "project_hashes.json"
DATA_DIR = PROJECT_DIR / "demo_data"
DATA_DIR.mkdir(exist_ok=True)

LOCAL_CLIPS_DIR = Path.home() / "Downloads" / "HandtrackingSamples"

# Hand tracking sample clips — egocentric hand manipulation footage.
# dataset_key: "vla"        → VLA Egocentric (action/object/affordance annotation)
#              "world_model" → World Model (top-view scene understanding)
#              "audio"       → Multimodal audio + transcription
DEMO_CLIPS = [
    # --- VLA Egocentric dataset (first-person hand manipulation) ---
    {"name": "Aayan_4.mp4",        "description": "Egocentric hand manipulation — Aayan session 4",        "dataset_key": "vla"},
    {"name": "Aayan_5.mp4",        "description": "Egocentric hand manipulation — Aayan session 5",        "dataset_key": "vla"},
    {"name": "Karan_5.mp4",        "description": "Egocentric hand manipulation — Karan session 5",        "dataset_key": "vla"},
    {"name": "Prathamesh_1.mp4",   "description": "Egocentric hand manipulation — Prathamesh session 1",   "dataset_key": "vla"},
    {"name": "Prathamesh_2.mp4",   "description": "Egocentric hand manipulation — Prathamesh session 2",   "dataset_key": "vla"},
    {"name": "Sameer_2.mp4",       "description": "Egocentric hand manipulation — Sameer session 2",       "dataset_key": "vla"},
    {"name": "Sameer_3.mp4",       "description": "Egocentric hand manipulation — Sameer session 3",       "dataset_key": "vla"},
    {"name": "Sameer_5.mp4",       "description": "Egocentric hand manipulation — Sameer session 5",       "dataset_key": "vla"},
    {"name": "Sarfaraz_1.mp4",     "description": "Egocentric hand manipulation — Sarfaraz session 1",     "dataset_key": "vla"},
    {"name": "Sarfaraz_3.mp4",     "description": "Egocentric hand manipulation — Sarfaraz session 3",     "dataset_key": "vla"},
    {"name": "Varsha_H4.mp4",      "description": "Egocentric hand manipulation — Varsha session 4",       "dataset_key": "vla"},
    {"name": "Varsha_H5.mp4",      "description": "Egocentric hand manipulation — Varsha session 5",       "dataset_key": "vla"},
    {"name": "Vinayak_2.mp4",      "description": "Egocentric hand manipulation — Vinayak session 2",      "dataset_key": "vla"},
    {"name": "Vinayak_3.mp4",      "description": "Egocentric hand manipulation — Vinayak session 3",      "dataset_key": "vla"},
    # --- World Model dataset (top-view / scene understanding) ---
    {"name": "Aayan_TopView_SpoonTransfer.mp4",          "description": "Top-view spoon transfer — scene understanding",            "dataset_key": "world_model"},
    {"name": "Aayan_TopView_WaterTransfer.mp4",          "description": "Top-view water transfer — scene understanding",            "dataset_key": "world_model"},
    {"name": "Sameer_TopView_pastatransfer.mp4",         "description": "Top-view pasta transfer — scene understanding",            "dataset_key": "world_model"},
    {"name": "Sameer_TpView_SpoonZipbagTransfer.mp4",    "description": "Top-view spoon + zip-bag transfer — scene understanding",  "dataset_key": "world_model"},
    {"name": "arranging_shoes_and_slipper_bedroom_up_14_003.mov", "description": "Object arrangement from above — scene understanding", "dataset_key": "world_model"},
    # --- Audio dataset (multimodal annotation) ---
    {"name": "Vinayak_1.mp4",      "description": "Hand manipulation with audio — multimodal annotation",  "dataset_key": "audio"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stage_clip(source_dir: Path, dest_path: Path, name: str) -> bool:
    """Copy a clip from the local source directory to demo_data/. Returns True on success."""
    if dest_path.exists():
        print(f"  ↳ Already staged: {name}")
        return True

    src = source_dir / name
    if not src.exists():
        print(f"  ✗ Source not found: {src}")
        return False

    print(f"  Staging {name}...")
    try:
        shutil.copy2(src, dest_path)
        size_mb = dest_path.stat().st_size / 1e6
        print(f"  ✓ Staged ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed to stage {name}: {e}")
        return False


def upload_to_encord(client: EncordUserClient, dataset_hash: str, file_path: Path) -> bool:
    """Upload a local video file to an Encord dataset."""
    print(f"  Uploading {file_path.name} to Encord...")
    try:
        dataset = client.get_dataset(dataset_hash)
        dataset.upload_video(
            file_path=str(file_path),
            title=file_path.name,
        )
        print(f"  ✓ Uploaded: {file_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HASHES_FILE.exists():
        print(f"ERROR: {HASHES_FILE} not found.")
        print("Run python scripts/01_setup_encord_project.py first.")
        return

    with open(HASHES_FILE) as f:
        hashes = json.load(f)

    print("Connecting to Encord...")
    client = EncordUserClient.create_with_ssh_private_key(
        ssh_private_key_path=SSH_KEY_PATH
    )
    print("  ✓ Connected\n")

    upload_results = []

    if not LOCAL_CLIPS_DIR.exists():
        print(f"ERROR: Source clips directory not found: {LOCAL_CLIPS_DIR}")
        print("Expected hand tracking samples at ~/Downloads/HandtrackingSamples/")
        return

    for clip in DEMO_CLIPS:
        print(f"\n{'─'*50}")
        print(f"Clip: {clip['name']}")
        print(f"  {clip['description']}")

        dest = DATA_DIR / clip["name"]
        staged = stage_clip(LOCAL_CLIPS_DIR, dest, clip["name"])

        if staged and dest.exists():
            dataset_hash = hashes[clip["dataset_key"]]["dataset_hash"]
            success = upload_to_encord(client, dataset_hash, dest)
            upload_results.append({"name": clip["name"], "success": success})
        else:
            print(f"  ⚠ Skipping upload — file not available.")
            upload_results.append({"name": clip["name"], "success": False})

    # Summary
    succeeded = sum(1 for r in upload_results if r["success"])
    print(f"\n{'='*50}")
    print(f"Upload complete: {succeeded}/{len(upload_results)} clips uploaded.")
    print("="*50)
    print("\nNext step: run  python scripts/03_curation_pipeline.py")


if __name__ == "__main__":
    main()
