
## Quick Start

```bash
pip install -r requirements.txt
export ENCORD_SSH_KEY_PATH=~/.ssh/id_ed25519

# 1. Create ontologies + projects on Encord platform
python scripts/01_setup_encord_project.py

# 2. Download public demo clips + upload to Encord
python scripts/02_upload_demo_data.py

# 3. Run curation pipeline demo (Act 1)
python scripts/03_curation_pipeline.py

# 4. Run audio annotation workflow demo (Act 3)
python scripts/04_audio_workflow.py

# SAM2 tracking notebook (Act 2) — open in Jupyter
jupyter notebook notebooks/sam2_video_tracking_demo.ipynb
```

## What's Here

```
ontologies/
  vla_ontology.json              # VLA Egocentric — Dasher delivery annotation spec
  world_model_ontology.json      # World model — scene understanding spec
  audio_transcription_ontology.json  # Multimodal audio + transcription spec

scripts/
  01_setup_encord_project.py    # Creates 3 projects on Encord platform
  02_upload_demo_data.py        # Downloads CC0 demo clips + uploads to Encord
  03_curation_pipeline.py       # Quality filtering pipeline (Act 1 demo)
  04_audio_workflow.py          # Audio annotation workflow (Act 3 demo)

notebooks/
  sam2_video_tracking_demo.ipynb  # SAM2 video tracking + ROI calc (Act 2 demo)
```

## Demo Script (30 min)

### Act 1 — Indexing & Curation (10 min)
Run `03_curation_pipeline.py`. Show quality score per clip. Message: *"Raw data ≠ valuable data."*

### Act 2 — VLA & World Model Annotation (15 min)
Open `sam2_video_tracking_demo.ipynb`. Show one click → full video tracking.
Open Encord platform → show VLA ontology with actions/objects/affordances.
Open Encord platform → show world model ontology with depth/scene/motion layers.

### Act 3 — Multimodal Audio (5 min)
Run `04_audio_workflow.py`. Show synchronized video+audio annotation.
Message: *"40+ languages, niche dialects, synchronized to video frame timestamps."*

## Requirements

- Python 3.10+
- Encord account + SSH key (for scripts 01–04)
- GPU recommended for SAM2 notebook (CPU works but slower)
- ffmpeg (for audio extraction): `brew install ffmpeg`
