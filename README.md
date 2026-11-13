# efficientat-infer

Standalone inference for the **mn04** and **mn10** EfficientAT MobileNet AudioSet models. One call returns all 527 native scores in Google AudioSet MID order and the model's pooled backbone embedding.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

## Why this exists

[EfficientAT](https://github.com/fschmid56/EfficientAT) is research code for compact audio taggers. Its original inference path expects the research checkout layout and imports TorchVision and TorchAudio binary packages. This package extracts the two tested MobileNet models into an installable, inference-only session with verified checkpoint downloads and an explicit input/result contract. It does not change the trained models.

## Acknowledgments

- [Florian Schmid, Khaled Koutini and Gerhard Widmer](https://arxiv.org/abs/2211.04772) created the EfficientAT research and models. All model credit belongs to them.
- [Florian Schmid's EfficientAT repository](https://github.com/fschmid56/EfficientAT) supplies the original model code, under MIT, and the [v0.0.1 author release](https://github.com/fschmid56/EfficientAT/releases/tag/v0.0.1) supplies both checkpoints.
- [Google AudioSet](https://research.google.com/audioset/) supplies the 527-class MID vocabulary.
- [TorchAudio](https://github.com/pytorch/audio) supplies the original Kaldi mel-bank and sinc-resampling algorithms; their retained pure Torch dependency closure is under [BSD-2-Clause](docs/TORCHAUDIO_LICENSE).
- [TorchVision](https://github.com/pytorch/vision) supplied the MobileNet convolution wrapper contract retained by a small pure Torch compatibility module, under [BSD-3-Clause](docs/TORCHVISION_LICENSE).

## Citation

From the [paper's arXiv BibTeX record](https://arxiv.org/bibtex/2211.04772):

```bibtex
@misc{schmid2023efficientlargescaleaudiotagging,
  title={Efficient Large-scale Audio Tagging via Transformer-to-CNN Knowledge Distillation},
  author={Florian Schmid and Khaled Koutini and Gerhard Widmer},
  year={2023},
  eprint={2211.04772},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2211.04772},
}
```

## Features

- `mn10` (default) and `mn04` use the author AudioSet checkpoints at v0.0.1.
- Scores, labels and MIDs preserve the official AudioSet CSV order. Scores are independent sigmoid outputs, **not calibrated confidence**.
- The additional embedding is the unnormalized adaptive average of the last MobileNet feature map: 960 values for mn10 or 384 for mn04. It is not a contrastive text embedding.
- Paths, NumPy arrays and Torch tensors; CPU or CUDA; reusable resident model; automatic verified download and explicit offline folder.
- No implicit segmentation. Each `infer()` call scores the complete supplied audio. A caller doing 10-second window evaluation supplies each window separately and combines its results.

## Scope

Training, evaluation frameworks, datasets, model fine-tuning, Dynamic MobileNet variants, frame-level event timing and a global model manager are outside this package. The package does not bundle checkpoints. The supported inference scope is mn04 and mn10 clip-level AudioSet tagging and native pooled embeddings.

## Install

Python 3.10+ and Torch 2.0+. Install the appropriate CPU or CUDA Torch build for your platform first, then install current source from public GitHub:

```bash
python -m pip install 'efficientat-infer @ git+https://github.com/openmirlab/efficientat-infer.git'
```

There is no PyPI release. In a local development checkout, `python -m pip install .` also works. TorchVision and the TorchAudio binary runtime are not required.

## Quick start

```python
from efficientat_infer import EfficientATSession

with EfficientATSession("mn10", device="auto") as session:
    session.load()  # downloads and verifies the author checkpoint on first use
    result = session.infer("audio.wav", pad_to_seconds=10)
    print(result.ranked_tags[:5])
    print(result.scores.shape, result.embedding.shape)  # (527,), (960,)
```

`pad_to_seconds=10` right-pads a short clip before resampling, as in the evaluation that motivated this package. It raises if the supplied audio is longer than ten seconds. Omit it to score the whole unpadded clip. For a 30-second recording, the caller owns the three 10-second windows and any max-across-windows rule. The convenience function `tag_audio(...)` loads and closes a model per call; keep a session open for bulk scanning.

```python
from efficientat_infer import tag_audio

result = tag_audio("audio.wav", model_id="mn04", device="cpu")
```

## Inputs and result meaning

File inputs use SoundFile metadata and accept common formats supported by libsndfile. Waveform inputs require `sample_rate=`, floating-point samples conventionally scaled to `[-1, 1]`, and shape `[samples]` or `[channels, samples]`; a SoundFile `[samples, channels]` NumPy array must be transposed. Channels are averaged to mono, optional zeros are appended at the original rate, then a checkpoint-faithful Torch sinc resampler converts to 32 kHz. Integer PCM arrays, empty and non-finite audio raise instead of being silently mis-scaled. A short unpadded input that cannot form the mel frontend raises instead of being silently changed.

`result.scores[i]` corresponds to `result.mids[i]` and `result.labels[i]`. Service adapters should align classes by MID even though this package already uses canonical AudioSet CSV order. `result.audio` records `input_samples` and `padding_samples` at the original rate; `normalized_samples`, `model_padding_samples` and `model_samples` are counts at the 32 kHz model rate, with `normalized_samples + model_padding_samples == model_samples`. `result.checkpoint` records the verified model identity and SHA-256; `loaded_checkpoint_info()` adds the resident device. `result.semantics` and weight-free `output_semantics("mn10")` describe the native outputs. `result.to_dict()` serializes all 527 scores and metadata; pass `include_embedding=False` to omit the larger feature array explicitly.

Scores and embeddings own read-only CPU NumPy storage, so `release()` or `close()` does not invalidate earlier results.

The mel frontend uses a 25 ms Hann window, 10 ms hop, 128 Kaldi mel bins, a 32 kHz input and the original pre-emphasis and normalization. The output scores and embedding describe the entire model input, including explicit right padding. There are no timestamps or localized event estimates.

`load()` is idempotent and holds one model in memory. `infer()` requires a ready session; `release()` frees its resident model and permits reload; `close()` is terminal. `status`, `cache_info()` and `loaded_checkpoint_info()` expose lifecycle/cache and the verified resident checkpoint. `device="auto"` chooses CUDA 0 when available, else CPU; an explicit unavailable CUDA request fails. MPS and MLX are unsupported.

## Checkpoints and offline use

The package registry at [config/checkpoints.json](src/efficientat_infer/config/checkpoints.json) pins both author release assets and hashes. Missing managed files download to `~/.cache/efficientat-infer/v0.0.1/` on `load()`; use `EFFICIENTAT_INFER_CACHE` or `cache_dir=` to change the root. Downloads are atomic and SHA-256 checked. Existing corruption raises. `cache_info()` does not hash, download or write, and never claims verified integrity.

For an air-gapped installation, download the files yourself into one folder:

| Model | Author URL | Size | SHA-256 |
| --- | --- | ---: | --- |
| mn04 | [mn04_as_mAP_432.pt](https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt) | 4,066,513 | `899a8c6217063f6941793102a2780b3dd788ab11fce468d9cf5f0577c453321c` |
| mn10 | [mn10_as_mAP_471.pt](https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt) | 19,708,753 | `0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e` |

```python
with EfficientATSession("mn10", checkpoint_dir="/path/to/weights", offline=True) as session:
    session.load()
    result = session.infer("audio.wav")
```

`checkpoint_config=` accepts a validated JSON registry with alternate URLs, filenames and hashes for compatible mn04/mn10 architecture checkpoints. The upstream code is MIT; the release does not state a separate checkpoint license. Record that distinction before any public redistribution of weights.

## CLI

```bash
efficientat-infer audio.wav --model mn10 --device cpu --pad-to-seconds 10 --output result.json
efficientat-infer audio.wav --model mn04 --checkpoint-dir /path/to/weights --offline
```

The JSON contains all 527 scores, MIDs, labels, the embedding and result metadata.

## What this project will NEVER bundle

Model checkpoints, private audio, datasets, training/evaluation code or research experiment logs. The test suite includes only generated audio and compact original-output fixtures.

## Development

```bash
python -m pip install '.[dev]'
python -m pytest -q
EFFICIENTAT_TEST_WEIGHTS_DIR=/path/to/author/weights python -m pytest -q
python -m build
```

The committed fixture was captured by `tools/capture_upstream.py` from upstream revision `a425fdce92572e602a1d5634799bd9f1f2efa806` with Torch 2.7.1+cu126 on CPU and two Torch threads. It includes a nontrivial 10-second signal with three seconds of genuine silence. On that recorded build, the extracted frontend, mn04/mn10 logits and pooled features match upstream exactly. End-to-end A/B also matched exactly on real 44.1 kHz mono and 48 kHz stereo audio with ten-second right padding; reproduce with `tools/verify_upstream_parity.py`. Both author URLs were downloaded into a fresh managed cache and verified against the pinned hashes; the built wheel loaded both models without importing TorchVision or TorchAudio. A different Torch build skips bit-exact fixture assertions; checkpoint integration must be checked on its own platform.

## License

The extracted model code is MIT. The retained TorchAudio math is BSD-2-Clause and the TorchVision compatibility wrapper is BSD-3-Clause; see [NOTICE](NOTICE) and the licenses under `docs/`. AudioSet label/ontology terms are separate. This is an independent openmirlab integration; model research and training belong to the cited authors.

## Support

Report package issues in [openmirlab/efficientat-infer issues](https://github.com/openmirlab/efficientat-infer/issues).
