"""
05_vla_agent.py
---------------
Encord Task Agent that uses Qwen2.5-VL-7B (served by vLLM on the GH200) to
pre-annotate egocentric delivery videos with the VLA Egocentric ontology.

Inference backend
-----------------
  vLLM serves Qwen2.5-VL-7B-Instruct as an OpenAI-compatible REST API on
  localhost:8000. The agent sends base64-encoded JPEG frames to it and parses
  the JSON response into Encord labels. The model never loads into agent memory.

Encord workflow setup
---------------------
  Add an Agent node named "Agent" to the VLA project workflow in the Encord UI.
  Create two outgoing pathways: "annotated" and "error".

Usage
-----
  # 1. Start vLLM (if not already running via systemd)
  #    sudo systemctl start vllm
  #
  # 2. Run the agent
  export ENCORD_SSH_KEY_PATH=~/.ssh/id_ed25519
  python scripts/05_vla_agent.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Iterator, Optional

from PIL import Image

# ---------------------------------------------------------------------------
# Encord imports
# ---------------------------------------------------------------------------
from encord.objects import OntologyStructure
from encord.objects.classification_instance import ClassificationInstance
from encord.objects.common import ChecklistAttribute, RadioAttribute
from encord.objects.coordinates import BoundingBoxCoordinates, PointCoordinate, PolygonCoordinates
from encord.objects.frames import Range
from encord.objects.ontology_object_instance import ObjectInstance
from encord.storage import StorageItem
from encord_agents.tasks import Runner
from encord_agents.core.data_model import Frame
from encord_agents.tasks.dependencies import dep_storage_item, dep_video_iterator
from encord_agents.core.dependencies import Depends
from encord_agents.tasks.models import TaskAgentReturnStruct

from encord.objects.ontology_labels_impl import LabelRowV2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SSH_KEY_PATH = os.environ.get(
    "ENCORD_SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_ed25519")
)
HASHES_PATH = Path(__file__).parent.parent / "project_hashes.json"

# Encord workflow stage / pathway names (must match the UI)
AGENT_STAGE_NAME  = "Qwen-VLA"
PATHWAY_ANNOTATED = "annotated"
PATHWAY_ERROR     = "error"

# vLLM server — runs on the same GH200, served by systemd
VLLM_BASE_URL  = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_API_KEY   = os.environ.get("VLLM_API_KEY", "token-encord")   # vLLM ignores value
VLLM_MODEL     = os.environ.get("VLLM_MODEL", "qwen-vl")          # --served-model-name
VLLM_MAX_TOKENS = 768                  # multi-object frames need ~500 tokens; 768 is safe

# Resize frames to this max side length before encoding. Keeps image token count
# well below max_model_len (4096). At 640px Qwen2.5-VL uses ~300-500 visual tokens.
MAX_IMAGE_SIDE = 640

# Frame sampling: run inference every Nth frame (delivery videos change slowly)
INFERENCE_STRIDE = 10

# Max frames sent to vLLM in parallel — matches vLLM --max-num-seqs 64 with headroom
INFERENCE_CONCURRENCY = 16

# ---------------------------------------------------------------------------
# Ontology constants (must match vla_ontology.json exactly)
# ---------------------------------------------------------------------------

ACTION_PHASE_OPTIONS = [
    "navigating_to_location", "approaching_entrance", "interacting_with_access_control",
    "carrying_item", "placing_item", "handing_item_to_person", "scanning_confirming",
    "returning_to_vehicle", "waiting", "other",
]
SCENE_ENV_OPTIONS = [
    "outdoor_street", "outdoor_parking", "outdoor_path", "building_lobby",
    "building_hallway", "building_elevator", "apartment_corridor",
    "residential_front", "commercial_entrance", "other",
]
VIDEO_QUALITY_OPTIONS = [
    "camera_shake", "motion_blur", "occlusion_heavy", "low_light",
    "overexposed", "partial_frame", "clean",
]
TASK_COMPLETION_OPTIONS = [
    "successful_delivery", "left_at_door", "handed_to_person",
    "failed_no_access", "failed_no_recipient", "returned_to_vehicle",
]
OBJECT_LABELS = [
    "Active Hand", "Manipulated Object", "Target Location",
    "Person / Recipient", "Obstacle / Hazard",
]

# ---------------------------------------------------------------------------
# Inference data structures
# ---------------------------------------------------------------------------


@dataclass
class ObjectPred:
    """A predicted object for one frame."""
    label: str
    bbox: Optional[tuple[float, float, float, float]] = None  # (x, y, w, h) normalised
    polygon: Optional[list[tuple[float, float]]] = None
    attributes: dict[str, str | list[str]] = field(default_factory=dict)


@dataclass
class FramePrediction:
    """All predictions for a single video frame."""
    frame_idx: int
    objects: list[ObjectPred] = field(default_factory=list)
    classifications: dict[str, str | list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OpenAI client (singleton pointing at local vLLM)
# ---------------------------------------------------------------------------

def _get_openai_client():
    """Return a cached OpenAI client pointed at the local vLLM server."""
    from openai import OpenAI
    # Re-use a single client for the lifetime of the process
    if not hasattr(_get_openai_client, "_client"):
        _get_openai_client._client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
        )
    return _get_openai_client._client


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert video annotator for egocentric last-mile delivery footage. "
    "Analyse the provided frame and respond ONLY with a valid JSON object — no markdown, "
    "no extra text.\n\n"
    "Schema:\n"
    "{\n"
    '  "action_phase": "<one of ACTION_PHASE_OPTIONS>",\n'
    '  "scene_environment": "<one of SCENE_ENV_OPTIONS>",\n'
    '  "video_quality": ["<zero or more from VIDEO_QUALITY_OPTIONS>"],\n'
    '  "objects": [\n'
    "    {\n"
    '      "label": "<one of OBJECT_LABELS>",\n'
    '      "bbox": [x, y, w, h],  // normalised 0-1, top-left origin; null if not visible\n'
    '      "attributes": {}\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    f"ACTION_PHASE_OPTIONS: {json.dumps(ACTION_PHASE_OPTIONS)}\n"
    f"SCENE_ENV_OPTIONS: {json.dumps(SCENE_ENV_OPTIONS)}\n"
    f"VIDEO_QUALITY_OPTIONS: {json.dumps(VIDEO_QUALITY_OPTIONS)}\n"
    f"OBJECT_LABELS: {json.dumps(OBJECT_LABELS)}\n\n"
    "Object attribute schemas:\n"
    '  "Active Hand": hand_side (left|right|both), '
    "hand_state (empty|holding_rigid|holding_flexible|in_contact|reaching), "
    "grasp_type (power_grasp|pinch_grasp|hook_grasp|none)\n"
    '  "Manipulated Object": object_category (package_bag|package_box|package_envelope|'
    "food_container|food_bag|beverage|door_handle|doorbell|keypad|mailbox|phone_device|other), "
    "affordances [list of: graspable|pushable|openable|pressable|liftable|fragile|heavy], "
    "interaction_phase (approaching|grasping|carrying|placing|releasing|none)\n"
    '  "Target Location": location_type (front_door|doorstep|mailbox|lobby_desk|locker|'
    "hand_to_person|doorman|other), "
    "accessibility (direct_access|requires_keypad|requires_doorbell|requires_buzzer|gated)\n"
    '  "Person / Recipient": person_role (recipient|bystander|doorman|unknown), '
    "attention_state (looking_at_agent|aware_of_agent|unaware|approaching)\n"
    '  "Obstacle / Hazard": hazard_type (vehicle|cyclist|pedestrian|step_curb|'
    "wet_surface|narrow_passage|dog_animal|other), "
    "severity (minor|moderate|blocking)\n\n"
    "Only include objects that are clearly visible. Omit any object that is not present."
)


# ---------------------------------------------------------------------------
# Inference — calls vLLM via OpenAI-compatible API
# ---------------------------------------------------------------------------


def run_vla_inference(frame_obj: Frame) -> FramePrediction:
    """
    Send a video frame to the vLLM-served Qwen2.5-VL-7B model and return
    structured predictions for the Encord VLA ontology.

    Args:
        frame_obj: Frame dataclass with .frame (int index) and .content (numpy array)

    Returns:
        FramePrediction with classifications and object detections
    """
    client = _get_openai_client()
    # Resize frame to MAX_IMAGE_SIDE before encoding so Qwen2.5-VL visual token
    # count stays within max_model_len (4096). Full-res frames can exceed 16k tokens.
    img = Image.fromarray(frame_obj.content)
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    }

    response = client.chat.completions.create(
        model=VLLM_MODEL,
        max_tokens=VLLM_MAX_TOKENS,
        temperature=0.0,   # deterministic
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    image_content,
                    {
                        "type": "text",
                        "text": "Analyse this frame and return the JSON annotation.",
                    },
                ],
            },
        ],
    )

    raw_text = response.choices[0].message.content or ""
    return _parse_response(raw_text, frame_obj.frame)


def _parse_response(text: str, frame_idx: int) -> FramePrediction:
    """Parse the model's JSON response into a FramePrediction. Fails gracefully."""
    pred = FramePrediction(frame_idx=frame_idx)

    # Strip markdown code fences if the model wraps the JSON
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        print(f"  [warn] Frame {frame_idx}: no JSON in response — skipping")
        return pred

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        print(f"  [warn] Frame {frame_idx}: JSON parse error ({exc}) — skipping")
        return pred

    # Frame classifications
    if data.get("action_phase") in ACTION_PHASE_OPTIONS:
        pred.classifications["Action Phase"] = data["action_phase"]
    if data.get("scene_environment") in SCENE_ENV_OPTIONS:
        pred.classifications["Scene Environment"] = data["scene_environment"]
    valid_quality = [f for f in data.get("video_quality", []) if f in VIDEO_QUALITY_OPTIONS]
    if valid_quality:
        pred.classifications["Video Quality"] = valid_quality

    # Object detections
    for obj in data.get("objects", []):
        label = obj.get("label")
        if label not in OBJECT_LABELS:
            continue
        bbox = None
        raw_bbox = obj.get("bbox")
        if raw_bbox and len(raw_bbox) == 4:
            x, y, w, h = (float(v) for v in raw_bbox)
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0 - x, w))
            h = max(0.0, min(1.0 - y, h))
            if w > 0 and h > 0:
                bbox = (x, y, w, h)
        pred.objects.append(ObjectPred(
            label=label,
            bbox=bbox,
            attributes=obj.get("attributes", {}),
        ))

    return pred


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def _find_ontology_object(structure: OntologyStructure, name: str):
    for obj in structure.objects:
        if obj.name == name:
            return obj
    return None


def _find_ontology_classification(structure: OntologyStructure, cls_name: str):
    for cls in structure.classifications:
        if cls.attributes and cls.attributes[0].name == cls_name:
            return cls
    return None


def _set_attr_answer(instance, attribute_name: str, answer, frame_idx: int = 0) -> None:
    """Resolve a raw answer to the typed Option object and set it on the instance."""
    for attr in instance.ontology_item.attributes:
        if attr.name == attribute_name:
            try:
                if isinstance(attr, RadioAttribute):
                    resolved = next(
                        (opt for opt in attr.options if opt.title == answer), None
                    )
                    if resolved is None:
                        return
                elif isinstance(attr, ChecklistAttribute):
                    answer_set = set(answer) if isinstance(answer, list) else {answer}
                    resolved = [opt for opt in attr.options if opt.title in answer_set]
                    if not resolved:
                        return
                else:
                    resolved = answer  # TextAttribute
                try:
                    instance.set_answer(resolved, attribute=attr, frames=frame_idx)
                except Exception:
                    # Static attributes don't accept frames — retry without
                    instance.set_answer(resolved, attribute=attr, overwrite=True)
            except Exception as exc:
                print(f"  [warn] set_answer({attribute_name!r}): {exc}")
            return


def write_predictions_to_label_row(label_row, predictions: list[FramePrediction]) -> None:
    """
    Convert FramePredictions → Encord label objects and write them into label_row.
    label_row must already be initialised before calling this.
    """
    ontology = label_row.ontology_structure

    for pred in predictions:
        # Objects
        for obj_pred in pred.objects:
            if obj_pred.bbox is None and obj_pred.polygon is None:
                continue
            onto_obj = _find_ontology_object(ontology, obj_pred.label)
            if onto_obj is None:
                print(f"  [warn] Object '{obj_pred.label}' not in ontology — skipping")
                continue
            instance = ObjectInstance(onto_obj)
            frame_range = Range(pred.frame_idx, pred.frame_idx + INFERENCE_STRIDE - 1)
            if obj_pred.bbox is not None:
                x, y, w, h = obj_pred.bbox
                instance.set_for_frames(
                    coordinates=BoundingBoxCoordinates(
                        top_left_x=x, top_left_y=y, width=w, height=h
                    ),
                    frames=frame_range,
                )
            elif obj_pred.polygon is not None:
                instance.set_for_frames(
                    coordinates=PolygonCoordinates(
                        values=[PointCoordinate(x=px, y=py) for px, py in obj_pred.polygon]
                    ),
                    frames=frame_range,
                )
            for attr_name, answer in obj_pred.attributes.items():
                _set_attr_answer(instance, attr_name, answer, pred.frame_idx)
            label_row.add_object_instance(instance)

    # -------------------------------------------------------------------
    # Frame classifications — group by (cls_name, answer_key) so we
    # create ONE ClassificationInstance per unique answer and set all
    # matching frames on it (SDK rejects overlapping frame ranges).
    # -------------------------------------------------------------------
    # Collect: { (cls_name, answer_key) : [frame_idx, ...] }
    cls_groups: dict[tuple[str, str], list[int]] = {}
    for pred in predictions:
        for cls_name, answer in pred.classifications.items():
            # Use a hashable key — for checklists, freeze the sorted list
            if isinstance(answer, list):
                key = (cls_name, tuple(sorted(answer)))
            else:
                key = (cls_name, answer)
            cls_groups.setdefault(key, []).append(pred.frame_idx)

    print(f"  Classification groups: {len(cls_groups)} "
          f"({', '.join(f'{k[0]}={k[1]!r} [{len(v)}f]' for k, v in cls_groups.items())})")

    for (cls_name, answer_key), frame_indices in cls_groups.items():
        onto_cls = _find_ontology_classification(ontology, cls_name)
        if onto_cls is None:
            print(f"  [warn] Classification '{cls_name}' not in ontology — skipping")
            continue
        attribute = onto_cls.attributes[0]
        # Resolve string → typed Option
        answer_raw = list(answer_key) if isinstance(answer_key, tuple) else answer_key
        try:
            if isinstance(attribute, RadioAttribute):
                resolved = next(
                    (opt for opt in attribute.options if opt.title == answer_raw), None
                )
                if resolved is None:
                    continue
            elif isinstance(attribute, ChecklistAttribute):
                answer_set = set(answer_raw) if isinstance(answer_raw, list) else {answer_raw}
                resolved = [opt for opt in attribute.options if opt.title in answer_set]
                if not resolved:
                    continue
            else:
                resolved = answer_raw
        except Exception as exc:
            print(f"  [warn] option lookup for {cls_name!r}: {exc}")
            continue
        cls_instance = ClassificationInstance(onto_cls)
        cls_instance.set_for_frames(Range(min(frame_indices), max(frame_indices)))
        try:
            cls_instance.set_answer(resolved, attribute)
        except Exception as exc:
            print(f"  [warn] set_answer({cls_name!r}): {exc}")
        try:
            label_row.add_classification_instance(cls_instance)
        except Exception as exc:
            print(f"  [warn] add_classification({cls_name!r}): {exc}")


# ---------------------------------------------------------------------------
# Task Agent
# ---------------------------------------------------------------------------


def main() -> None:
    with open(HASHES_PATH) as f:
        hashes = json.load(f)
    # Allow overriding the project hash via env var (e.g. for testing projects)
    project_hash = os.environ.get("AGENT_PROJECT_HASH") or hashes["vla"]["project_hash"]

    # encord-agents reads the SSH key from ENCORD_SSH_KEY_FILE env var
    os.environ["ENCORD_SSH_KEY_FILE"] = SSH_KEY_PATH

    print(f"Connecting to Encord  (key: {SSH_KEY_PATH})")
    print(f"vLLM endpoint         {VLLM_BASE_URL}  model={VLLM_MODEL}")

    runner = Runner(project_hash=project_hash)
    print(f"Project hash          {project_hash}")
    print(f"Agent stage           '{AGENT_STAGE_NAME}'\n")

    @runner.stage(stage=AGENT_STAGE_NAME)
    def vla_annotate(
        label_row: LabelRowV2,
        video_iterator: Annotated[Iterator[Frame], Depends(dep_video_iterator)],
        storage_item:   Annotated[StorageItem, Depends(dep_storage_item)],
    ) -> TaskAgentReturnStruct:
        title = label_row.data_title
        print(f"Processing: {title}")

        label_row.initialise_labels(overwrite=True)

        # Purge any previously-saved annotations so we always annotate from a clean slate.
        # Required because on retry the server may have partial labels from a prior run,
        # and adding new instances to the same frames raises LabelRowError (overlapping).
        for inst in list(label_row.get_object_instances()):
            label_row.remove_object(inst)
        for inst in list(label_row.get_classification_instances()):
            label_row.remove_classification(inst)

        # Sample frames then run inference in parallel
        sampled = [f for f in video_iterator if f.frame % INFERENCE_STRIDE == 0]
        print(f"  Sampled {len(sampled)} frames (stride={INFERENCE_STRIDE}), "
              f"running {INFERENCE_CONCURRENCY} in parallel…")

        predictions: list[FramePrediction] = []
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=INFERENCE_CONCURRENCY) as pool:
            futures = {pool.submit(run_vla_inference, f): f for f in sampled}
            for future in as_completed(futures):
                frame_obj = futures[future]
                try:
                    predictions.append(future.result())
                except Exception as exc:
                    errors.append(f"frame {frame_obj.frame}: {exc}")
                    print(f"  [error] frame {frame_obj.frame}: {exc}")

        if errors:
            print(f"  {len(errors)} frame(s) failed — routing to '{PATHWAY_ERROR}'")
            return TaskAgentReturnStruct(pathway=PATHWAY_ERROR, label_row=label_row)

        print(f"  {len(predictions)} frames annotated")
        write_predictions_to_label_row(label_row, predictions)
        label_row.save()
        print(f"  Labels saved → '{PATHWAY_ANNOTATED}'")
        return TaskAgentReturnStruct(pathway=PATHWAY_ANNOTATED, label_row=label_row)

    runner.run()


if __name__ == "__main__":
    main()
