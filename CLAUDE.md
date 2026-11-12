# efficientat-infer maintainer guide

This standalone package extracts the mn04/mn10 AudioSet inference path from
Florian Schmid's EfficientAT repository at
`a425fdce92572e602a1d5634799bd9f1f2efa806`. The model and research are
by Florian Schmid, Khaled Koutini and Gerhard Widmer. Author release v0.0.1
supplies the checkpoints; this package never bundles them. Code is MIT,
retained pure Torch TorchAudio frontend math is BSD-2-Clause, the TorchVision
compatibility wrapper follows BSD-3-Clause, and checkpoint
licensing is not separately stated in the release.

## Scope and status

- `mn04` and `mn10` only, both fixed 527-class AudioSet classifiers.
- Runtime surface: model, frontend, labels, normalization, verified downloader,
  cache, CLI, result semantics and an independent resident session.
- No training, evaluation, datasets, experiment CLI, Dynamic MobileNet, shared
  runtime or hidden 10-second segmentation in the shipped package.
- Generated upstream fixture is committed. Current local parity on recorded
  Torch 2.7.1+cu126 CPU with two threads: frontend, both logits and pooled
  features exactly equal. Two real-audio end-to-end A/B runs (44.1 kHz mono,
  48 kHz stereo) also have zero maximum difference at every compared stage.
  `torch.view_as_real(stft(return_complex=True))` was independently checked
  bit-equal to upstream's deprecated real-return STFT on the captured input.
- Fresh managed downloads verified both author checkpoint hashes. Offline
  checkpoint-enabled suite: 8/8 passed. Wheel from sdist built; installed
  wheel loaded labels and checkpoint, ran API/CLI, and imported no TorchVision
  or TorchAudio binary runtime. CUDA execution produced finite 527 scores for
  both models. Python 3.10/3.12/3.13 CI and fresh published-name pip install
  remain release gates, not claims made by this local build.
- This is a local new package; GitHub/PyPI publication and org-hosted weights
  have not happened.
- `openmirlab-skills` was searched for existing EfficientAT guidance; none
  exists while this package is local and unpublished.

## File navigation

Every load-bearing file starts with a purpose, rationale and `Reads:` header.
`session.py` owns lifecycle and inference. `checkpoints.py` owns the resolver
shared by `load()` and `cache_info()`. `_impl/audio.py` owns input policy.
`_impl/frontend.py`, `_impl/kaldi.py` and `_impl/resample.py` own the model
frontend. `_impl/model.py`, `_impl/block_types.py`, `_impl/conv.py`,
`_impl/attention_pooling.py` and `_impl/utils.py` own the checkpoint-faithful MN.
`results.py` owns score/embedding meaning; `cli.py` only calls the public API.

## Verification

From this checkout with a CPU/CUDA Torch environment:

```bash
python -m pip install '.[dev]'
python -m pytest -q
EFFICIENTAT_TEST_WEIGHTS_DIR=/path/to/author/weights python -m pytest -q
python -m build
```

Regenerate the original fixture *only* from a clean, pinned upstream checkout:

```bash
python tools/capture_upstream.py \
  --upstream /path/to/EfficientAT-a425fdc \
  --weights /path/to/author/weights \
  --output tests/fixtures/upstream_cpu.npz
```

The fixture's JSON records the exact Torch build. Do not treat an exact float
digest from another build as an accuracy regression. Preserve silence and
nontrivial model outputs in any replacement fixture. Check checkpoint hashes
and strict state load, then compare mel, all logits, features and scores.
For changes to input handling, compare real audio end to end against the
upstream path and record tolerance relative to output scale.

Before a release: verify author attribution and current URLs, run the full
suite with both checkpoints, build wheel from sdist, install that wheel in a
clean environment, verify imports and `config/` data, and confirm a real
`pip install` of the published package after publishing. Update this file,
README and CHANGELOG together for any public behavior or install change.
