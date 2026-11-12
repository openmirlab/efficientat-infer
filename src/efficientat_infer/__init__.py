"""Standalone EfficientAT AudioSet inference package."""

from .__about__ import __version__
from .checkpoints import audioset_labels
from .results import EfficientATResult, output_semantics
from .session import EfficientATSession, tag_audio

__all__ = [
    "__version__", "EfficientATSession", "EfficientATResult", "tag_audio",
    "audioset_labels", "output_semantics",
]
