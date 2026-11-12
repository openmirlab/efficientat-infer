"""Own EfficientAT resident model lifecycle and ready-only inference.

mn04 and mn10 share one package but each session loads one model. There is no
global model manager and repeated infer calls reuse the resident weights.
Reads: checkpoint store, audio normalizer, MobileNet backbone and mel frontend.
"""

import threading

import numpy as np
import torch

from ._impl.audio import normalize
from ._impl.frontend import EfficientATMel
from ._impl.model import get_model
from .checkpoints import CheckpointStore, audioset_labels
from .results import EfficientATResult


def _device(request):
    if request == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if request == "cpu":
        return torch.device("cpu")
    if isinstance(request, str) and (request == "cuda" or (request.startswith("cuda:") and request[5:].isdigit())):
        index = 0 if request == "cuda" else int(request[5:])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device unavailable: {request}")
        return torch.device(f"cuda:{index}")
    raise ValueError("device must be auto, cpu, cuda, or cuda:N; mps is unsupported")


class EfficientATSession:
    """Explicit load, reusable infer, reloadable release, terminal close."""

    def __init__(
        self, model_id="mn10", *, device="auto", cache_dir=None,
        checkpoint_dir=None, offline=False, checkpoint_config=None,
    ):
        self.model_id = model_id
        self._device_request = device
        self._store = CheckpointStore(
            model_id, cache_dir=cache_dir, checkpoint_dir=checkpoint_dir,
            offline=offline, checkpoint_config=checkpoint_config,
        )
        self._lock = threading.RLock()
        self._status = "new"
        self._device = None
        self._model = None
        self._frontend = None
        self._labels = None
        self._mids = None
        self._checkpoint = None

    @property
    def status(self):
        return self._status

    @property
    def device(self):
        return str(self._device) if self._device is not None else None

    def cache_info(self):
        return self._store.cache_info()

    def loaded_checkpoint_info(self):
        with self._lock:
            if self._status != "ready":
                raise RuntimeError("Call load() before loaded_checkpoint_info()")
            return {**self._checkpoint, "device": str(self._device)}

    def load(self):
        with self._lock:
            if self._status == "closed":
                raise RuntimeError("Cannot load a closed EfficientATSession")
            if self._status == "ready":
                return self
            self._status = "loading"
            try:
                device = _device(self._device_request)
                labels, mids = audioset_labels(self._store.registry)
                path = self._store.resolve()
                width = 0.4 if self.model_id == "mn04" else 1.0
                model = get_model(width_mult=width, pretrained_name=None)
                state = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                model = model.to(device).eval()
                frontend = EfficientATMel().to(device).eval()
                self._device = device
                self._model = model
                self._frontend = frontend
                self._labels = labels
                self._mids = mids
                self._checkpoint = self._store.provenance()
                self._status = "ready"
                return self
            except BaseException:
                self._model = self._frontend = self._device = None
                self._labels = self._mids = self._checkpoint = None
                self._status = "failed"
                raise

    def infer(self, audio, *, sample_rate=None, pad_to_seconds=None):
        with self._lock:
            if self._status != "ready":
                raise RuntimeError("Call load() before infer()")
            self._status = "busy"
            try:
                waveform, audio_info = normalize(
                    audio, sample_rate, self._device, pad_to_seconds=pad_to_seconds
                )
                with torch.inference_mode():
                    mel = self._frontend(waveform)
                    logits, features = self._model(mel.unsqueeze(0))
                    scores = torch.sigmoid(logits.float())[0].cpu().numpy().copy()
                    embedding = features[0].float().cpu().numpy().copy()
                if scores.shape != (527,) or not np.isfinite(scores).all():
                    raise RuntimeError("EfficientAT returned invalid AudioSet scores")
                scores.setflags(write=False)
                embedding.setflags(write=False)
                return EfficientATResult(
                    model_id=self.model_id, scores=scores, embedding=embedding,
                    labels=self._labels, mids=self._mids, audio=audio_info,
                    checkpoint=dict(self._checkpoint),
                )
            finally:
                self._status = "ready"

    def release(self):
        with self._lock:
            if self._status == "closed":
                return
            self._model = self._frontend = self._device = None
            self._labels = self._mids = self._checkpoint = None
            self._status = "released"

    def close(self):
        with self._lock:
            if self._status != "closed":
                self.release()
                self._status = "closed"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def tag_audio(audio, *, model_id="mn10", sample_rate=None, pad_to_seconds=None, **session_options):
    """Convenience one-shot path; use a session to avoid reloading on every call."""
    with EfficientATSession(model_id=model_id, **session_options) as session:
        session.load()
        return session.infer(audio, sample_rate=sample_rate, pad_to_seconds=pad_to_seconds)
