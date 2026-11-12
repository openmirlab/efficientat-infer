# Changelog

## 0.1.0 — 2026-11-12

- Add standalone mn04 and mn10 AudioSet inference sessions, native 527-class
  scores, AudioSet MIDs, and pooled model embeddings.
- Pin author release checkpoints with SHA-256 verification and explicit cache,
  manual-folder, and offline paths.
- Retain original CPU inference outputs exactly for the recorded Torch build
  while removing TorchVision and TorchAudio binary runtime dependencies.
- Use complex STFT plus `view_as_real` to retain the original spectrum values
  without the deprecated `return_complex=False` call.
