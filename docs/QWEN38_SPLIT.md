# Qwen3.8-27B: first Mac + iPhone generation

On September 15, 2026, the 18 GiB M3 Pro Mac and a physical iPhone 16 completed one short **Qwen3.8-27B 4-bit** generation over USB. The phone retained and executed layers 0–11; the Mac constructed layers 12–63, embeddings, final normalization, and output head. Vision weights were omitted. The Mac did not load the phone layers during the split run.

**This is an exploratory working prototype, not a completed capacity validation.** The 12-layer prefill comparison still misses its original numerical tolerance, and the successful run used substantial Mac swap. The structured evidence is [QWEN38_SPLIT_RESULT.json](QWEN38_SPLIT_RESULT.json).

## Observed result

Prompt: “In one short sentence, explain why the sky appears blue.”

> The sky appears blue because shorter wavelengths of sunlight, such as blue, are scattered more effectively by the atmosphere than longer wavelengths.

| Measurement | Observed value |
| --- | --- |
| Mac assigned weight storage | 12.564 GB / 11.701 GiB |
| iPhone assigned weight storage | 2.569 GB / 2.393 GiB |
| Mac loading after phone connection | 5.71 seconds |
| First token after generation started | 26.91 seconds |
| Decode rate after first token | 3.93 tokens/second |
| Entire generation | 33.28 seconds, 26 tokens including EOS |
| Mac peak MLX allocation | 12.675 GB |
| iPhone peak MLX allocation | 2.596 GB |
| Minimum sampled iPhone process headroom | 883 MB |
| Peak additional system swap on Mac | 3.85 GiB |

These are a single short debug-build run. The phone was provisioned and loaded before Mac timing. The actual prompt was 24 tokens; an 8,192-token ceiling was configured but not exercised. No 15-minute stability, thermal, long-context, or quality benchmark has been completed.

## Controls and changes

The initial split attempt crossed the predeclared 4 GiB additional-system-swap guard during loading, with zero observed tokens. Applying MLX's device-recommended wired-memory limit during **loading as well as generation** enabled the completed attempt. The working limit was not raised beyond the device recommendation. The 180-second deadline, 4 GiB swap guard, and 384 MiB minimum sampled phone headroom remained in place.

A subsequent Mac-only control used the same shard files, coordinator, MLX core, prompt, generation settings, and early-wiring policy; both stages ran locally. It loaded all text weights, then crossed the swap guard before any token was logged: 4.74 GiB additional system swap at the triggering sample, 9.73 seconds into the monitored process. Its supervisor encountered a process-group termination error and did not write its normal final report. A later process-list check found no remaining benchmark worker. The separately labeled `recovered-result.json` preserves that limitation. The recorder now signals only its own child and persists the triggering sample before cleanup; a direct child-termination check passed.

The completed split and control began with different OS paging states and uncontrolled file caches. System swap includes other applications. The earlier [MLX-LM diagnostic](MAC_BASELINE.md) used MLX 0.32.2 and wired memory during generation only; it is historical context, not the equivalent current control. None of these guard stops is an observed fatal out-of-memory error or proof that the Mac can never run this model.

## Numerical validation remains open

Four real checkpoint layers passed Python-versus-Swift prefill and cached-continuation checks. The 12-layer test uses actual checkpoint embeddings for fixed 5-token, 1-token, and 3-token sequences. Its original criteria were `rtol=0.03, atol=0.03`; they were not relaxed after failure.

The first implementation mixed Python MLX 0.32.2 and the Swift package's embedded MLX 0.31.1. Tracing found differences beginning at quantized matrix multiplication. Pinning Python to 0.31.1 removed differences in all traced operations in the first three recurrent layers. In the next full-attention layer, one BF16 attention-output value still differed by 0.00390625. The origin of this remaining library/build-level difference has not been fully resolved.

With aligned versions, the 12-layer phone result has relative RMS errors of 0.70%, 0.94%, and 0.64%. The single-token and cached multi-token checks pass; the initial prefill fails the per-element criterion. Therefore the full run required the recorder's explicit `--exploratory` mode. Producing a sensible sentence does not override that failed check.

The full upstream-versus-partitioned small synthetic hybrid model, reset behavior, and failure handling pass in Python. The complete suite has 105 tests and 25 subtests; three Swift validation tests pass. Both Mac and signed iOS builds succeeded.

## Implementation and reproduction

The checkpoint is `mlx-community/Qwen3.8-27B-4bit`, installed revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`. Its `qwen3_5` architecture uses affine 4-bit weights, group size 64, BF16 activations, Gated DeltaNet recurrent state, and full-attention KV caches. Python MLX and the Swift embedded core are now both pinned to 0.31.1; MLX-LM is 0.31.1 and MLX Swift is 0.31.6. No additional 27B model download was needed.

`src/mlx_peer/hybrid.py` stream-exports assigned weights without constructing the full model. `HybridStage.swift` validates the assigned quantized tensors and executes the stage. `HybridDeltaKernel.swift` ports the upstream unmasked recurrent Metal kernel; its MIT license is retained in `licenses/MLX-LM-MIT.txt`.

The USB connection uses the existing authenticated, loopback-only worker, explicit USB device selection, and no Wi-Fi fallback. BF16 activations travel as their original 16-bit representations. Socket timeouts are 120 seconds for this larger-model probe. File hashing now releases Foundation read buffers per chunk, avoiding file-sized temporary memory retention. The app remains foreground-only and restores its idle-timer behavior when the worker stops.

From the project root, using a new export directory:

```python
from mlx_peer.hybrid import export
export("/path/to/Qwen3.8-27B-MLX-4bit", "artifacts/my-split", 0, 12)
```

```sh
.venv/bin/python scripts/validate_hybrid.py prepare \
  --fixture artifacts/my-split --model /path/to/Qwen3.8-27B-MLX-4bit
```

Build/install using the [iOS instructions](../ios/README.md). Copy only `config.json`, `request.json`, `weights.safetensors`, and a private 64-character hex `wire-token.txt` into an app Documents subfolder; keep the Mac shard and reference outputs on the Mac. Launch with `--serve-fixture NAME`, then run `validate_hybrid.py usb` with the corresponding `--fixture`, `--serial`, and `--token-file`. Validation stops the listener afterward; restart it before generation.

```sh
.venv/bin/python scripts/record_hybrid_split.py \
  --model /path/to/Qwen3.8-27B-MLX-4bit --fixture artifacts/my-split \
  --output artifacts/my-generation --serial DEVICE_SERIAL \
  --token-file /path/to/private/wire-token.txt --exploratory
```

The recorder refuses a non-exploratory USB run unless the stage-validation report passes. For the corresponding Mac-only control, use a fresh output directory and `--local-stage`, omitting serial/token arguments. Do not run the two memory-intensive tests concurrently.

Raw completed generation: `artifacts/qwen38-usb-generation-20260915-03/`. Aligned references and phone checks: `artifacts/qwen38-split12-mlx031/`. Earlier attempts and failed mixed-version comparisons remain preserved in separate artifact folders. No model weights, credentials, or repository were published.
