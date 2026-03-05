"""
01_setup_encord_project.py
--------------------------
Creates three Encord ontologies and projects on the platform from the JSON spec files:
  1. VLA Egocentric — Dasher Delivery
  2. World Model — Egocentric Scene Understanding
  3. Multimodal Audio — Voice & Environmental Sound

Usage:
    export ENCORD_SSH_KEY_PATH=~/.ssh/id_ed25519
    python scripts/01_setup_encord_project.py
"""

import json
import os
import sys
from pathlib import Path

from encord import EncordUserClient
from encord.objects import OntologyStructure, Shape
from encord.objects.attributes import (
    ChecklistAttribute,
    RadioAttribute,
    TextAttribute,
)
from encord.orm.dataset import StorageLocation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SSH_KEY_PATH = os.environ.get("ENCORD_SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_ed25519"))
ONTOLOGY_DIR = Path(__file__).parent.parent / "ontologies"

SHAPE_MAP = {
    "BOUNDING_BOX": Shape.BOUNDING_BOX,
    "POLYGON": Shape.POLYGON,
    "POLYLINE": Shape.POLYLINE,
    "KEYPOINT": Shape.POINT,
    "BITMASK": Shape.BITMASK,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_attributes(parent, attributes: list):
    """Recursively add attributes to an object or classification node."""
    for attr_spec in attributes:
        attr_type = attr_spec["type"]
        attr_name = attr_spec["name"]
        required = attr_spec.get("required", False)

        if attr_type == "radio":
            attr = parent.add_attribute(RadioAttribute, attr_name, required=required)
            for option_label in attr_spec.get("options", []):
                attr.add_option(option_label)

        elif attr_type == "checklist":
            attr = parent.add_attribute(ChecklistAttribute, attr_name, required=required)
            for option_label in attr_spec.get("options", []):
                attr.add_option(option_label)

        elif attr_type == "text":
            parent.add_attribute(TextAttribute, attr_name, required=required)


def build_ontology_structure(spec: dict) -> OntologyStructure:
    """Build an OntologyStructure from our JSON spec format."""
    structure = OntologyStructure()

    # ---- Objects ----
    for obj_spec in spec.get("objects", []):
        shape = SHAPE_MAP[obj_spec["shape"]]
        obj = structure.add_object(obj_spec["name"], shape=shape)
        _apply_attributes(obj, obj_spec.get("attributes", []))

    # ---- Frame classifications ----
    for cls_spec in spec.get("frame_classifications", []):
        # add_classification() takes no name — uid is auto-generated as int.
        # The classification name comes from the attribute added to it.
        cls = structure.add_classification()
        required = cls_spec.get("required", False)

        if cls_spec["type"] == "radio":
            attr = cls.add_attribute(RadioAttribute, cls_spec["name"], required=required)
            for option_label in cls_spec.get("options", []):
                attr.add_option(option_label)

        elif cls_spec["type"] == "checklist":
            attr = cls.add_attribute(ChecklistAttribute, cls_spec["name"], required=required)
            for option_label in cls_spec.get("options", []):
                attr.add_option(option_label)

    return structure


def create_ontology_from_file(client: EncordUserClient, json_path: Path) -> str:
    """Load a JSON spec, build the ontology on Encord, return ontology hash."""
    with open(json_path) as f:
        spec = json.load(f)

    title = spec["title"]
    print(f"\n→ Creating ontology: {title}")
    structure = build_ontology_structure(spec)
    try:
        ontology = client.create_ontology(title=title, structure=structure)
        ontology_hash = ontology.ontology_hash
    except (ValueError, AttributeError):
        # SDK fails to parse the API response (known issue with classification IDs),
        # but the ontology WAS created — find it by title.
        print("  (SDK parse issue — locating created ontology by title...)")
        matched = [o for o in client.get_ontologies(title) if o.title == title]
        if not matched:
            raise RuntimeError(f"Ontology '{title}' not found after creation attempt")
        ontology_hash = matched[0].ontology_hash
    print(f"  ✓ Created — hash: {ontology_hash}")
    return ontology_hash


def create_demo_dataset(client: EncordUserClient, title: str) -> str:
    """Create an Encord Storage dataset, return dataset hash."""
    print(f"\n→ Creating dataset: {title}")
    response = client.create_dataset(
        dataset_title=title,
        dataset_type=StorageLocation.CORD_STORAGE,
    )
    dataset_hash = response.dataset_hash
    print(f"  ✓ Created — hash: {dataset_hash}")
    return dataset_hash


def create_project(
    client: EncordUserClient,
    title: str,
    dataset_hashes: list,
    ontology_hash: str,
) -> str:
    """Create an Encord project linked to a dataset and ontology."""
    print(f"\n→ Creating project: {title}")
    project = client.create_project(
        project_title=title,
        dataset_hashes=dataset_hashes,
        ontology_hash=ontology_hash,
    )
    # SDK returns a string (project_hash) directly in some versions
    project_hash = project if isinstance(project, str) else project.project_hash
    print(f"  ✓ Created — hash: {project_hash}")
    return project_hash


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Connecting to Encord...")
    client = EncordUserClient.create_with_ssh_private_key(
        ssh_private_key_path=SSH_KEY_PATH
    )
    print("  ✓ Connected\n")

    results = {}

    # ---- 1. VLA Project ----
    vla_ontology_hash = create_ontology_from_file(
        client, ONTOLOGY_DIR / "vla_ontology.json"
    )
    vla_dataset_hash = create_demo_dataset(client, "DoorDash Demo — VLA Egocentric")
    vla_project_hash = create_project(
        client,
        title="DoorDash Demo — VLA Egocentric Delivery",
        dataset_hashes=[vla_dataset_hash],
        ontology_hash=vla_ontology_hash,
    )
    results["vla"] = {
        "ontology_hash": vla_ontology_hash,
        "dataset_hash": vla_dataset_hash,
        "project_hash": vla_project_hash,
    }

    # ---- 2. World Model Project ----
    wm_ontology_hash = create_ontology_from_file(
        client, ONTOLOGY_DIR / "world_model_ontology.json"
    )
    wm_dataset_hash = create_demo_dataset(client, "DoorDash Demo — World Model")
    wm_project_hash = create_project(
        client,
        title="DoorDash Demo — World Model Scene Understanding",
        dataset_hashes=[wm_dataset_hash],
        ontology_hash=wm_ontology_hash,
    )
    results["world_model"] = {
        "ontology_hash": wm_ontology_hash,
        "dataset_hash": wm_dataset_hash,
        "project_hash": wm_project_hash,
    }

    # ---- 3. Audio Project ----
    audio_ontology_hash = create_ontology_from_file(
        client, ONTOLOGY_DIR / "audio_transcription_ontology.json"
    )
    audio_dataset_hash = create_demo_dataset(client, "DoorDash Demo — Audio Transcription")
    audio_project_hash = create_project(
        client,
        title="DoorDash Demo — Multimodal Audio Annotation",
        dataset_hashes=[audio_dataset_hash],
        ontology_hash=audio_ontology_hash,
    )
    results["audio"] = {
        "ontology_hash": audio_ontology_hash,
        "dataset_hash": audio_dataset_hash,
        "project_hash": audio_project_hash,
    }

    # ---- Save hashes for later scripts ----
    output_path = Path(__file__).parent.parent / "project_hashes.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("All projects created successfully.")
    print(f"Hashes saved to: {output_path}")
    print("="*60)
    print("\nNext step: run  python scripts/02_upload_demo_data.py")


if __name__ == "__main__":
    main()
