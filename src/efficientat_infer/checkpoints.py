"""Own pinned EfficientAT artifact configuration and verified acquisition.

Managed weights download only on explicit load. Cache inspection and model
discovery never read model bytes or create directories.
Reads: package checkpoint registry, bundled AudioSet labels, local cache files.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from datetime import date
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from filelock import FileLock


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _registry(config_path=None):
    try:
        text = (
            Path(config_path).read_text(encoding="utf-8")
            if config_path is not None
            else resources.files("efficientat_infer")
            .joinpath("config/checkpoints.json")
            .read_text(encoding="utf-8")
        )
        data = json.loads(text)
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read EfficientAT checkpoint registry") from exc
    required = {
        "release", "source_repo", "source_revision", "updated_at", "code_license",
        "checkpoint_license", "labels", "models",
    }
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError("Invalid EfficientAT checkpoint registry")
    for key in required - {"labels", "models"}:
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"Invalid registry field: {key}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", data["release"]):
        raise ValueError("Unsafe registry release path")
    if not re.fullmatch(r"[0-9a-f]{40}", data["source_revision"]):
        raise ValueError("Invalid source revision")
    try:
        date.fromisoformat(data["updated_at"])
    except ValueError as exc:
        raise ValueError("Invalid registry update date") from exc
    labels = data["labels"]
    if not isinstance(labels, dict) or not all(
        isinstance(labels.get(k), str) and labels[k]
        for k in ("namespace", "sha256", "source")
    ):
        raise ValueError("Invalid AudioSet label metadata")
    if not re.fullmatch(r"[0-9a-f]{64}", labels["sha256"]):
        raise ValueError("Invalid AudioSet label SHA256")
    if not isinstance(data["models"], dict) or set(data["models"]) != {"mn04", "mn10"}:
        raise ValueError("Registry must define mn04 and mn10")
    for model_id, artifact in data["models"].items():
        if not isinstance(artifact, dict) or not {"filename", "url", "size", "sha256"}.issubset(artifact):
            raise ValueError(f"Invalid {model_id} artifact")
        filename = artifact["filename"]
        if not isinstance(filename, str) or not filename or Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError(f"Unsafe {model_id} filename")
        if not isinstance(artifact["size"], int) or isinstance(artifact["size"], bool) or artifact["size"] <= 0:
            raise ValueError(f"Invalid {model_id} size")
        if not isinstance(artifact["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
            raise ValueError(f"Invalid {model_id} SHA256")
        url = artifact["url"]
        if not isinstance(url, str) or urlparse(url).scheme != "https" or not urlparse(url).hostname:
            raise ValueError(f"Invalid {model_id} HTTPS URL")
    return copy.deepcopy(data)


def audioset_labels(registry=None):
    """Return the bundled 527 class names and MIDs without loading a model."""
    registry = _registry() if registry is None else registry
    raw = resources.files("efficientat_infer").joinpath("config/class_labels_indices.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != registry["labels"]["sha256"]:
        raise ValueError("Bundled AudioSet labels differ from registry identity")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    if len(rows) != 527 or [int(row["index"]) for row in rows] != list(range(527)):
        raise ValueError("Invalid AudioSet label indices")
    mids = tuple(row["mid"] for row in rows)
    if len(set(mids)) != 527:
        raise ValueError("Duplicate AudioSet MID")
    return tuple(row["display_name"] for row in rows), mids


class CheckpointStore:
    """One package-owned resolver for loading and cache inspection."""

    def __init__(self, model_id, *, cache_dir=None, checkpoint_dir=None, offline=False, checkpoint_config=None):
        self.registry = _registry(checkpoint_config)
        if model_id not in self.registry["models"]:
            raise ValueError("model_id must be mn04 or mn10")
        self.model_id = model_id
        root = cache_dir if cache_dir is not None else os.environ.get(
            "EFFICIENTAT_INFER_CACHE", "~/.cache/efficientat-infer"
        )
        self.cache_dir = Path(root).expanduser().absolute() / self.registry["release"]
        self.checkpoint_dir = (
            Path(checkpoint_dir).expanduser().absolute() if checkpoint_dir is not None else None
        )
        self.offline = bool(offline)

    @property
    def artifact(self):
        return self.registry["models"][self.model_id]

    def _path(self):
        base = self.checkpoint_dir if self.checkpoint_dir is not None else self.cache_dir
        return base / self.artifact["filename"]

    def _verify(self, path):
        if path.stat().st_size != self.artifact["size"]:
            raise ValueError(f"Checkpoint size mismatch: {path}")
        if _sha256(path) != self.artifact["sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {path}")

    def _download(self, path):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as target:
                temporary = Path(target.name)
                size = 0
                with urlopen(self.artifact["url"], timeout=60) as response:
                    if urlparse(response.geturl()).scheme != "https":
                        raise ValueError("Checkpoint download redirected away from HTTPS")
                    while block := response.read(1 << 20):
                        size += len(block)
                        if size > self.artifact["size"]:
                            raise ValueError("Checkpoint download exceeds expected size")
                        target.write(block)
                target.flush()
                os.fsync(target.fileno())
            self._verify(temporary)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def resolve(self):
        """Return a verified file, downloading an absent managed file if online."""
        path = self._path()
        if path.exists():
            self._verify(path)
        elif self.checkpoint_dir is not None:
            raise FileNotFoundError(f"Missing explicit checkpoint file: {path}")
        elif self.offline:
            raise FileNotFoundError(f"Checkpoint unavailable offline: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(path) + ".lock"):
                if path.exists():
                    self._verify(path)
                else:
                    self._download(path)
        return path

    def cache_info(self):
        """Inspect the same resolved path without a write, download, or hash."""
        path = self._path()
        exists = path.is_file()
        return {
            "model_id": self.model_id,
            "path": str(path),
            "exists": exists,
            "size": path.stat().st_size if exists else None,
            "expected_size": self.artifact["size"],
            "expected_sha256": self.artifact["sha256"],
            "integrity": "not_checked",
            "source": "explicit" if self.checkpoint_dir is not None else "cache",
            "offline": self.offline,
        }

    def provenance(self):
        """Expected identity; only load() verifies the actual file."""
        return {
            "model_id": self.model_id,
            "checkpoint_id": f"{self.registry['source_repo']}@{self.registry['release']}:{self.model_id}",
            "source_revision": self.registry["source_revision"],
            "checkpoint_sha256": self.artifact["sha256"],
            "url": self.artifact["url"],
            "code_license": self.registry["code_license"],
            "checkpoint_license": self.registry["checkpoint_license"],
        }
