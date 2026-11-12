"""Checkpoint resolver's validation, path sharing, and integrity behavior.

Tiny synthetic artifacts exercise cache semantics without bundled model bytes.
Reads: package registry, temporary override metadata, temporary artifact files.
"""

import hashlib
import json
from importlib import resources

import pytest

from efficientat_infer.checkpoints import CheckpointStore


def registry():
    return json.loads(
        resources.files("efficientat_infer")
        .joinpath("config/checkpoints.json")
        .read_text()
    )


def test_explicit_verified_artifact_and_read_only_cache_info(tmp_path):
    data = registry()
    payload = b"real-size-and-sha-check"
    artifact = data["models"]["mn04"]
    artifact.update(filename="tiny.pt", size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    config = tmp_path / "config.json"
    config.write_text(json.dumps(data))
    explicit = tmp_path / "manual"
    explicit.mkdir()
    path = explicit / "tiny.pt"
    path.write_bytes(payload)
    store = CheckpointStore("mn04", checkpoint_dir=explicit, checkpoint_config=config, offline=True)
    before = store.cache_info()
    assert before["path"] == str(path)
    assert before["integrity"] == "not_checked"
    assert store.resolve() == path
    path.write_bytes(b"wrong-size")
    with pytest.raises(ValueError, match="size mismatch"):
        store.resolve()


def test_cache_info_does_not_create_paths(tmp_path):
    root = tmp_path / "no-cache"
    store = CheckpointStore("mn10", cache_dir=root, offline=True)
    assert store.cache_info()["exists"] is False
    assert not root.exists()
    with pytest.raises(FileNotFoundError, match="offline"):
        store.resolve()


def test_invalid_override_rejected_before_path_use(tmp_path):
    data = registry()
    data["models"]["mn10"]["filename"] = "../escape.pt"
    config = tmp_path / "bad.json"
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsafe"):
        CheckpointStore("mn10", checkpoint_config=config)
    data["models"] = ["mn04", "mn10"]
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Registry must define"):
        CheckpointStore("mn10", checkpoint_config=config)
