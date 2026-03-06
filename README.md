
## VLA-light-pass — Robotics Egocentric Annotation

Encord Task Agent that uses Qwen2.5-VL-7B (via vLLM) to pre-annotate
humanoid robot egocentric manipulation videos with Ego4D-style
verb+noun action decomposition.

## Quick Start

```bash
pip install -r requirements.txt
export ENCORD_SSH_KEY_PATH=~/.ssh/id_ed25519

# 1. Create ontologies + projects on Encord platform
python scripts/01_setup_encord_project.py

# 2. Upload robot clips to Encord
python scripts/02_upload_demo_data.py

# 3. Run curation pipeline (quality filtering)
python scripts/03_curation_pipeline.py

# 4. Run VLA annotation agent (on GH200 with vLLM)
export AGENT_PROJECT_HASH=<your-project-hash>
python scripts/05_vla_agent.py
```

## What's Here

```
ontologies/
  vla_ontology.json                # VLA Egocentric — robot manipulation annotation spec
  world_model_ontology.json        # World model — scene understanding spec
  audio_transcription_ontology.json  # Multimodal audio + transcription spec

scripts/
  01_setup_encord_project.py      # Creates 3 projects on Encord platform
  02_upload_demo_data.py          # Uploads robot clips to Encord
  03_curation_pipeline.py         # Quality filtering pipeline
  04_audio_workflow.py            # Audio annotation workflow
  05_vla_agent.py                 # Main VLA annotation agent (runs on GH200)

scripts/deploy/
  setup_gh200.sh                  # GH200 setup script
  test_vllm.py                    # vLLM smoke test
```

## Ontology

The VLA ontology uses Ego4D-style verb+noun decomposition for rich action labeling:

**Objects (4):** End Effector, Manipulated Object, Target Surface, Obstacle
**Classifications (5):** Manipulation Phase, Scene Context, Video Quality, Task Outcome, Motion Mode

Each Manipulated Object carries a `manipulation_verb` attribute (21 verbs: reach, grasp, lift, carry, place, etc.)
and an `object_category` attribute (17 categories: cup_mug, bottle, door_handle, etc.), enabling
compositional queries like "all frames where the robot is GRASPING a CUP in the KITCHEN".

## Deployment

- **GH200:** 192.222.50.122
- **vLLM service:** `systemd vllm.service`, endpoint `http://127.0.0.1:8000/v1`
- **Python env:** `source ~/agent-env/bin/activate`

## Requirements

- Python 3.10+
- Encord account + SSH key
- GH200 with vLLM for agent inference
- ffmpeg (for audio extraction): `brew install ffmpeg`
