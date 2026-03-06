"""
migrate_ontology.py
-------------------
Migrates the existing VLA ontology on Encord from the delivery schema to the
robotics manipulation schema. Adds new objects/classifications and archives
the old ones.

Usage:
    export ENCORD_SSH_KEY_PATH=~/.ssh/encord_key
    python scripts/migrate_ontology.py
"""

import json
import os
from pathlib import Path

from encord import EncordUserClient
from encord.objects import OntologyStructure, Shape
from encord.objects.attributes import ChecklistAttribute, RadioAttribute

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SSH_KEY_PATH = os.environ.get("ENCORD_SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_ed25519"))
HASHES_PATH = Path(__file__).parent.parent / "project_hashes.json"
ONTOLOGY_JSON = Path(__file__).parent.parent / "ontologies" / "vla_ontology.json"

# You can also pass the project hash directly via env var
PROJECT_HASH = os.environ.get("AGENT_PROJECT_HASH")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_attributes(parent, attributes: list):
    """Add attributes (radio/checklist) to an object or classification node."""
    for attr_spec in attributes:
        attr_type = attr_spec["type"]
        attr_name = attr_spec["name"]
        required = attr_spec.get("required", False)

        if attr_type == "radio":
            attr = parent.add_attribute(RadioAttribute, attr_name, required=required)
            for opt in attr_spec.get("options", []):
                attr.add_option(opt)
        elif attr_type == "checklist":
            attr = parent.add_attribute(ChecklistAttribute, attr_name, required=required)
            for opt in attr_spec.get("options", []):
                attr.add_option(opt)


SHAPE_MAP = {
    "BOUNDING_BOX": Shape.BOUNDING_BOX,
    "POLYGON": Shape.POLYGON,
    "POLYLINE": Shape.POLYLINE,
    "KEYPOINT": Shape.POINT,
    "BITMASK": Shape.BITMASK,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load project hashes
    with open(HASHES_PATH) as f:
        hashes = json.load(f)

    ontology_hash = hashes["vla"]["ontology_hash"]
    project_hash = PROJECT_HASH or hashes["vla"]["project_hash"]

    # Load new ontology spec
    with open(ONTOLOGY_JSON) as f:
        new_spec = json.load(f)

    print(f"Connecting to Encord (key: {SSH_KEY_PATH})")
    client = EncordUserClient.create_with_ssh_private_key(ssh_private_key_path=SSH_KEY_PATH)
    print("  ✓ Connected\n")

    # Get existing ontology
    ontology = client.get_ontology(ontology_hash)
    structure = ontology.structure
    print(f"Ontology: {ontology.title}  ({ontology_hash})")

    # --- Catalog existing items ---
    existing_objects = {obj.name for obj in structure.objects}
    existing_classifications = set()
    for cls in structure.classifications:
        if cls.attributes:
            existing_classifications.add(cls.attributes[0].name)

    print(f"  Existing objects: {existing_objects}")
    print(f"  Existing classifications: {existing_classifications}")

    # --- Add new objects ---
    new_object_names = {obj["name"] for obj in new_spec.get("objects", [])}
    for obj_spec in new_spec.get("objects", []):
        if obj_spec["name"] in existing_objects:
            print(f"  ⊘ Object '{obj_spec['name']}' already exists — skipping")
            continue
        shape = SHAPE_MAP[obj_spec["shape"]]
        obj = structure.add_object(obj_spec["name"], shape=shape)
        _add_attributes(obj, obj_spec.get("attributes", []))
        print(f"  + Added object: {obj_spec['name']} ({obj_spec['shape']})")

    # --- Add new classifications ---
    new_cls_names = {cls["name"] for cls in new_spec.get("frame_classifications", [])}
    for cls_spec in new_spec.get("frame_classifications", []):
        if cls_spec["name"] in existing_classifications:
            print(f"  ⊘ Classification '{cls_spec['name']}' already exists — skipping")
            continue
        cls = structure.add_classification()
        required = cls_spec.get("required", False)

        if cls_spec["type"] == "radio":
            attr = cls.add_attribute(RadioAttribute, cls_spec["name"], required=required)
            for opt in cls_spec.get("options", []):
                attr.add_option(opt)
        elif cls_spec["type"] == "checklist":
            attr = cls.add_attribute(ChecklistAttribute, cls_spec["name"], required=required)
            for opt in cls_spec.get("options", []):
                attr.add_option(opt)

        print(f"  + Added classification: {cls_spec['name']} ({cls_spec['type']})")

    # --- Save updated ontology ---
    print("\nSaving ontology...")
    ontology.save()
    print("  ✓ Ontology saved\n")

    # --- Report old items that should be archived ---
    old_objects = existing_objects - new_object_names
    old_classifications = existing_classifications - new_cls_names
    if old_objects or old_classifications:
        print("Old items to archive manually in Encord UI:")
        for name in old_objects:
            print(f"  ✗ Object: {name}")
        for name in old_classifications:
            print(f"  ✗ Classification: {name}")
        print("\n  (Cannot permanently delete while ontology is attached to a project.")
        print("   Archive them in the Encord UI → Ontology editor → ⋮ menu → Archive)")
    else:
        print("No old items to clean up.")

    print("\n✅  Migration complete.")
    print(f"   Project: {project_hash}")
    print(f"   Ontology: {ontology_hash}")


if __name__ == "__main__":
    main()
