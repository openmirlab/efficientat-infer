"""Native EfficientAT result and weight-free output interpretation contract.

Values describe the supplied clip, including explicit right padding. The
embedding is an unnormalized pooled backbone feature, not a calibrated score.
Reads: AudioSet labels and checkpoint identity supplied by the session.
"""

from dataclasses import dataclass

import numpy as np


def output_semantics(model_id):
    if model_id not in {"mn04", "mn10"}:
        raise ValueError("model_id must be mn04 or mn10")
    return {
        "contract_version": 1,
        "model_id": model_id,
        "scope": "whole supplied input after explicit padding, if any",
        "score_vocabulary": "Google AudioSet 527-class MID, class_labels_indices.csv order",
        "score_transform": "elementwise sigmoid of the native clip logits",
        "score_calibration": "not calibrated confidence; classes are independent",
        "embedding": {
            "source": "adaptive average pool of final MobileNet feature map",
            "dimension": 384 if model_id == "mn04" else 960,
            "normalization": "none",
            "scope": "whole model input including explicit padding",
        },
        "analysis": {
            "sample_rate_hz": 32000,
            "channels": 1,
            "mono": "arithmetic channel mean before resampling",
            "padding": "optional right zeros at original sample rate before resampling",
            "sample_counts": "input_samples and padding_samples are source-rate counts; normalized_samples, model_padding_samples and model_samples are 32000 Hz counts",
            "windowing": "none; caller supplies each desired window",
        },
    }


@dataclass(frozen=True)
class EfficientATResult:
    model_id: str
    scores: np.ndarray
    embedding: np.ndarray
    labels: tuple[str, ...]
    mids: tuple[str, ...]
    audio: dict
    checkpoint: dict

    @property
    def semantics(self):
        return output_semantics(self.model_id)

    @property
    def ranked_tags(self):
        order = np.argsort(-self.scores, kind="stable")
        return tuple(
            {"mid": self.mids[int(i)], "label": self.labels[int(i)], "score": float(self.scores[i])}
            for i in order
        )

    def to_dict(self, *, include_embedding=True):
        return {
            "model_id": self.model_id,
            "scores": self.scores.tolist(),
            "labels": list(self.labels),
            "mids": list(self.mids),
            "embedding": self.embedding.tolist() if include_embedding else None,
            "embedding_omitted": not include_embedding,
            "audio": dict(self.audio),
            "checkpoint": dict(self.checkpoint),
            "semantics": self.semantics,
        }
