# Changelog

## Unreleased

- Make the GitHub repository public (2026-11-12); not on PyPI and no weights are
  rehosted. Document installation from GitHub.
- Align README sections with OpenMIRLab style and document the public GitHub
  install and issue tracker.

## 0.1.0 — 2026-11-12

- Add standalone mn04 and mn10 AudioSet inference sessions, native 527-class
  scores, AudioSet MIDs, and pooled model embeddings.
- Pin author release checkpoints with SHA-256 verification and explicit cache,
  manual-folder, and offline paths.
- Retain original CPU inference outputs exactly for the recorded Torch build
  while removing TorchVision and TorchAudio binary runtime dependencies.
- Use complex STFT plus `view_as_real` to retain the original spectrum values
  without the deprecated `return_complex=False` call.
- Return read-only CPU score and embedding arrays that remain valid after
  session release.
