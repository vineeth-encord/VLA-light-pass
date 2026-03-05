"""
add_storage_to_project.py
--------------------------
Links existing Encord storage files (already uploaded to workspace) into
the VLA project dataset — no re-upload needed.

Usage:
    export ENCORD_SSH_KEY_PATH=~/.ssh/encord_key
    python scripts/add_storage_to_project.py
"""

import json
import os
from pathlib import Path

from encord import EncordUserClient
from encord.storage import StorageFolder

SSH_KEY_PATH = os.environ.get("ENCORD_SSH_KEY_PATH", str(Path.home() / ".ssh" / "encord_key"))
HASHES_FILE = Path(__file__).parent.parent / "project_hashes.json"

# The workspace folder UUID from the URL:
# https://app.encord.com/data/workspace/files/<FOLDER_UUID>/index/explorer
WORKSPACE_FOLDER_UUID = "aea55136-2916-4fed-9103-33a9858ee9a5"


def main():
    with open(HASHES_FILE) as f:
        hashes = json.load(f)

    vla_dataset_hash = hashes["vla"]["dataset_hash"]

    print("Connecting to Encord...")
    client = EncordUserClient.create_with_ssh_private_key(ssh_private_key_path=SSH_KEY_PATH)
    print("  ✓ Connected\n")

    # List items in the workspace folder
    print(f"→ Listing items in workspace folder {WORKSPACE_FOLDER_UUID}...")
    folder = client.get_storage_folder(WORKSPACE_FOLDER_UUID)
    items = list(folder.list_items())

    if not items:
        print("  No items found in folder. Check the folder UUID.")
        return

    print(f"  Found {len(items)} items:")
    for item in items:
        print(f"    - {item.name} ({item.uuid})")

    # Create a new non-folder-synced dataset and link the storage items to it
    from encord.orm.dataset import StorageLocation
    print(f"\n→ Creating new dataset (non-folder-synced)...")
    ds_response = client.create_dataset(
        dataset_title="VLA Delivery Videos",
        dataset_type=StorageLocation.CORD_STORAGE,
        create_backing_folder=False,
    )
    new_dataset_hash = ds_response.dataset_hash
    print(f"  ✓ Created dataset — hash: {new_dataset_hash}")

    print(f"\n→ Linking {len(items)} storage items to dataset...")
    dataset = client.get_dataset(new_dataset_hash)
    item_uuids = [item.uuid for item in items]
    dataset.link_items(item_uuids)
    print(f"  ✓ Linked {len(item_uuids)} items")

    # Add the new dataset to the existing VLA project
    vla_project_hash = hashes["vla"]["project_hash"]
    print(f"\n→ Adding dataset to VLA project {vla_project_hash}...")
    project = client.get_project(vla_project_hash)
    project.add_datasets([new_dataset_hash])
    print(f"  ✓ Done!")

    # Save the new dataset hash back to project_hashes.json
    hashes["vla"]["dataset_hash_2"] = new_dataset_hash
    with open(HASHES_FILE, "w") as f:
        json.dump(hashes, f, indent=2)

    print("\nVideos are now in the VLA project.")
    print(f"Open the label editor: https://app.encord.com/projects/{vla_project_hash}/labels")


if __name__ == "__main__":
    main()
