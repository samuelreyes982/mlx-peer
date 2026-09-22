# Contributing to MLX Peer

This is a research prototype. Useful contributions include tracing numerical differences, improving reproducibility, or testing explicit device/runtime combinations.

## Reproduce before comparing

Record the checkpoint revision, quantization, Python and Swift MLX versions, devices, prompt, context length, and memory starting conditions. Separate model loading, prefill, decoding, and transport time. A coherent answer alone does not prove numerical parity or capacity gain.

Use the portable checks in the README for protocol/data changes. On Apple Silicon, run the full Python suite with the pinned runtime. Swift and device changes need the relevant worker builds and reference-fixture comparisons described in `ios/README.md`.

Preserve original acceptance thresholds and retain failed measurements. Do not loosen a failed numerical tolerance to make a result pass. Keep exploratory runs labeled accordingly.

## Keep publication small and reproducible

Commit source, small synthetic examples, and sanitized measurement summaries. Do not commit model weights, generated fixtures, device IDs, wire tokens, signing profiles, private keys, `.venv`, or Xcode build products. Keep upstream license notices with adapted code.

A pull request should explain the observed problem, its fix, the commands used for validation, and remaining limitations. Use a separate issue to propose larger architecture changes.
