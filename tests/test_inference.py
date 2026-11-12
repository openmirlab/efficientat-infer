"""Golden frontend parity and public result semantics checks.

The fixture was recorded from unchanged upstream EfficientAT before extraction.
Reads: committed upstream_cpu.npz/json and the installed package source.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from efficientat_infer import EfficientATSession, audioset_labels, output_semantics
from efficientat_infer._impl.audio import normalize
from efficientat_infer._impl.frontend import EfficientATMel

FIXTURES = Path(__file__).parent / "fixtures"


def test_import_contract_without_weights():
    labels, mids = audioset_labels()
    assert len(labels) == len(mids) == len(set(mids)) == 527
    assert mids[0] == "/m/09x0r"
    assert output_semantics("mn04")["embedding"]["dimension"] == 384
    assert output_semantics("mn10")["embedding"]["dimension"] == 960
    session = EfficientATSession("mn10", device="cpu", offline=True)
    assert session.status == "new"
    assert session.cache_info()["integrity"] == "not_checked"
    with pytest.raises(RuntimeError, match="load"):
        session.infer(np.zeros(32000, dtype="float32"), sample_rate=32000)


def test_frontend_matches_upstream_golden():
    meta = json.loads((FIXTURES / "upstream_cpu.json").read_text())
    if torch.__version__ != meta["torch_version"]:
        pytest.skip("bit-exact upstream fixture requires its recorded Torch build")
    torch.set_num_threads(meta["torch_num_threads"])
    with np.load(FIXTURES / "upstream_cpu.npz", allow_pickle=False) as fixture:
        waveform = torch.from_numpy(fixture["waveform"])[None, :]
        with torch.inference_mode():
            actual = EfficientATMel()(waveform).numpy()
        assert np.array_equal(actual, fixture["mel"])
        assert not np.allclose(actual, 0)
        assert not np.any(fixture["waveform"][-32000 * 3 :])


def test_audio_padding_and_channel_order():
    samples = np.ones((2, 16000), dtype="float32")
    samples[1] = -1
    waveform, info = normalize(samples, 32000, torch.device("cpu"), pad_to_seconds=1)
    assert waveform.shape == (1, 32000)
    assert torch.count_nonzero(waveform) == 0
    assert info["input_samples"] == 16000
    assert info["padding_samples"] == 16000
    assert info["model_samples"] == info["normalized_samples"] + info["model_padding_samples"]
    _, resampled = normalize(
        np.zeros((2, 441), dtype=np.float32), 44100, torch.device("cpu"), pad_to_seconds=0.02
    )
    assert resampled["padding_samples"] == 441
    assert (resampled["normalized_samples"], resampled["model_padding_samples"]) == (320, 320)
    with pytest.raises(ValueError, match="shorter"):
        normalize(samples, 32000, torch.device("cpu"), pad_to_seconds=0.25)
    with pytest.raises(ValueError, match="floating-point"):
        normalize(np.zeros(32000, dtype=np.int16), 32000, torch.device("cpu"))


@pytest.mark.parametrize("model_id", ["mn04", "mn10"])
def test_author_checkpoint_matches_upstream_golden(model_id, monkeypatch):
    weights = os.environ.get("EFFICIENTAT_TEST_WEIGHTS_DIR")
    if weights is None:
        pytest.skip("set EFFICIENTAT_TEST_WEIGHTS_DIR for author-checkpoint parity")
    meta = json.loads((FIXTURES / "upstream_cpu.json").read_text())
    if torch.__version__ != meta["torch_version"]:
        pytest.skip("bit-exact upstream fixture requires its recorded Torch build")
    torch.set_num_threads(meta["torch_num_threads"])
    with np.load(FIXTURES / "upstream_cpu.npz", allow_pickle=False) as fixture:
        with EfficientATSession(model_id, device="cpu", checkpoint_dir=weights, offline=True) as session:
            session.load()
            assert session.loaded_checkpoint_info()["checkpoint_sha256"] == meta["checkpoint_sha256"][model_id]
            first = session.infer(fixture["waveform"], sample_rate=32000)
            # A second infer must use the resident model, not re-open the checkpoint.
            monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("reloaded checkpoint"))
            second = session.infer(fixture["waveform"], sample_rate=32000)
            expected = torch.sigmoid(torch.from_numpy(fixture[f"{model_id}_logits"]))[0].numpy()
            assert np.array_equal(first.scores, expected)
            assert np.array_equal(first.embedding, fixture[f"{model_id}_features"][0])
            assert np.array_equal(first.scores, second.scores)
            assert not first.scores.flags.writeable
            assert not first.embedding.flags.writeable
            assert first.mids == audioset_labels()[1]
            assert first.to_dict(include_embedding=False)["embedding_omitted"] is True
            session.release()
            assert session.status == "released"
            with pytest.raises(RuntimeError, match="load"):
                session.loaded_checkpoint_info()
