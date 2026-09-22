# MLX Peer — capacity experiment

Can the user's 18 GiB M3 Pro MacBook, which runs Qwen3.5-9B at 4-bit in SiphonNet, move to a 27B model by placing some model weights and execution on an iPhone 16?

The confirmed model target is **Qwen3.8-27B**. Its installed MLX checkpoint uses 4-bit weights. The first [recorded Mac-only diagnostic](MAC_BASELINE.md) hit a predeclared system-swap guard during loading, before any tokens were generated; this is not an observed fatal out-of-memory error. The user reports a 24 GB memory requirement, whose scope still needs verification against that configuration. See the [updated experiment protocol](CAPACITY_EXPERIMENT.md) and [planning manifest](../experiments/qwen38_27b_capacity.json). The earlier 9B success and SiphonNet 27B nonresponse remain user observations.

This is a community MLX developer-tool prototype. It is not affiliated with Apple. **The current implementation has completed one exploratory Qwen3.8-27B generation across the Mac and physical iPhone. Twelve layers and 2.57 GB of weights ran on the phone; decoding reached 3.93 tokens/second. Numerical prefill parity and sustained capacity validation remain open.** See the [27B result and limitations](QWEN38_SPLIT.md).

The [USB test](USB_PROBE.md) measured 0.461 ms median request round-trip latency and 36.38/42.74 MB/s payload throughput to/from the phone. Live stage checks and full-model comparisons passed; a 24-token Qwen2.5-0.5B continuation completed across both devices. These are short debug-build tests, not a 27B result or a demonstrated speedup.

The [physical iPhone validation](IPHONE_VALIDATION.md) passed: the phone executed four real Qwen2.5 layers and matched the Mac references for prefill and cached continuation. Its tiny self-test reported 3.26 GiB of available process memory at that moment; a sustained larger-shard budget is still unmeasured.

The [matched 9B sanity test](MAC_9B_SANITY.md) completed successfully: 1.63-second loading, approximately 27.81 generated tokens/second, and no additional observed system swap. It used the same recorder, prompt, and execution settings as the 27B diagnostic; starting OS memory conditions differed.

The [September 16 functional retest](QWEN38_FUNCTIONAL_RETEST.md) completed five short cases on a retry, including exact arithmetic, JSON, and repeated-token agreement after resets. The first attempt stopped at the swap guard; the existing numerical prefill check still fails.

## Development environment and dependencies

| Component | Experiment environment | Status |
| --- | --- | --- |
| Mac | Apple M3 Pro, 18 GiB memory; macOS **26.7**, build **25G229** | macOS version verified locally |
| Xcode | **26.6**, build **17F113**, with the Metal Toolchain component | Installed; current app build passed |
| iPhone | **iPhone 16 with iOS 27.0**, build **24A437** | Signed app installed; Qwen2 validation passed and exploratory Qwen3.8 generation completed |
| Python / MLX | Python 3.12; package versions in `requirements-lock.txt` | Installed in the project `.venv` |
| Swift / MLX | MLX Swift 0.31.6; dependencies in `ios/Package.resolved` | Pinned |

iOS 27.0 is the verified device environment. The package minimum deployment targets remain macOS 14 and iOS 17; those older versions have not been validated. Python MLX is pinned to 0.31.1 to match the Swift embedded core; the original baseline reports used 0.32.2.

## What is built

- A Python CLI that inspects dense Qwen2 safetensors headers, estimates per-device budgets, and streams separate Mac/iPhone weight files without allocating the complete model.
- A Mac coordinator built with MLX-LM which constructs only its assigned layers, with stage-local KV caches and explicit failure/reset behavior.
- A Swift MLX implementation of a contiguous Qwen2 layer stage, a Mac fixture runner, and an iPhone app for foreground self-tests and imported reference fixtures.
- Synthetic and trained-model reference fixtures, cache-continuation tests, SHA256-bound shard manifests, and numerical comparison reports.

The original Qwen2 path supports FP16/FP32 local fixtures and FP16 USB execution. The separate hybrid path supports the installed Qwen3.8 `qwen3_5` architecture with affine 4-bit weights, group size 64, and BF16 activations. It includes a streaming exporter, assigned-layer coordinator, Swift hybrid stage, and Metal recurrent kernel.

**Qwen3.8 support is experimental.** One 12-layer prefill comparison misses its original numerical tolerance, despite successful full-model text generation. Use the [dedicated hybrid scripts and result report](QWEN38_SPLIT.md); the older Qwen2 CLI and memory estimator do not handle hybrid checkpoints.

## Python setup

Use Python 3.12 on Apple silicon. Create an isolated `.venv` in your own checkout.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/mlx-peer doctor
```

MLX needs access to the Mac's Metal GPU. A sandbox can report `No Metal device available` even when the hardware supports MLX. Run GPU commands from an ordinary local terminal in that case.

Inspect and partition a local, complete Qwen2 checkpoint:

```sh
.venv/bin/mlx-peer inspect /path/to/checkpoint
.venv/bin/mlx-peer plan /path/to/checkpoint --start 4 --end 8 --context 4096
.venv/bin/mlx-peer split /path/to/checkpoint ./artifacts/my-split --start 4 --end 8
```

Layer ranges use an exclusive end. Output directories must be new to prevent overwrite. `plan` accepts `--mac-budget-gib` and `--iphone-budget-gib` only when those process budgets have been established; missing budgets remain unknown. Estimates include explicit reserve assumptions and are not a promise that a model fits.

The split contains `mac.safetensors`, `iphone.safetensors`, `config.json`, and a manifest recording original tensor names, exact storage bytes and streamed SHA256 identities. The capacity path does not use `mlx_lm.load()` on the full model first.

## Reproduce correctness checks

```sh
.venv/bin/python -m pytest -q
.venv/bin/mlx-peer make-fixture artifacts/my-stage-f16 --dtype float16
.venv/bin/mlx-peer verify-local artifacts/my-stage-f16
```

`make-fixture` creates a small random Qwen2 architecture; it is not a trained language model. `verify-local` compares the complete upstream model with the partitioned Python calculation, including multi-token prefill, single-token decoding and cached multi-token continuation. Both stages run on the Mac for this check.

For the Swift comparison, build the CLI using [the worker instructions](../ios/README.md), then run:

```sh
ios/DerivedData/Build/Products/Debug/mlx-peer-stage run-fixture artifacts/my-stage-f16
.venv/bin/mlx-peer verify-swift artifacts/my-stage-f16
```

`verify-swift` compares every output activation against Python references. Its report identifies tolerances; matching text alone is insufficient. Running the Swift CLI on macOS still counts as one physical device.

The downloaded test checkpoint is `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. Original weight SHA256: `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`. Original storage is BF16; the validation fixture explicitly converts it to FP16 and compares both implementations against that converted reference.

```sh
.venv/bin/mlx-peer checkpoint-fixture artifacts/qwen-0.5b-source artifacts/my-qwen-probe --start 4 --end 8
.venv/bin/mlx-peer verify-local artifacts/my-qwen-probe
ios/DerivedData/Build/Products/Debug/mlx-peer-stage run-fixture artifacts/my-qwen-probe
.venv/bin/mlx-peer verify-swift artifacts/my-qwen-probe
```

**`checkpoint-fixture` is a small-model reference-validation tool.** It deliberately loads a full reference and is limited to checkpoints of at most 2 GiB storage. It is separate from the streaming capacity path and must never be used as evidence that a larger model fits.

## Physical iPhone test

Open `ios/App/MLXPeerApp.xcodeproj`, select your development team and physical device, then run the app. See [iPhone build/setup](../ios/README.md) for signing and test details. The app's self-test is an internal cache consistency test. The imported fixture is the independent Python-versus-Swift test.

Create a phone-only bundle:

```sh
.venv/bin/mlx-peer pack-fixture artifacts/my-qwen-probe artifacts/phone-qwen-probe
```

This copies `config.json`, `request.json`, `weights.safetensors` and the selected input files, with a checksum inventory. It excludes the full reference, Mac shard and expected outputs. Return output files and `report.json` to a separate copy of the corresponding Mac fixture, then run `verify-swift`. Keep macOS and iPhone reports separate. Generate `artifacts/phone-qwen-probe` locally; generated fixtures are not distributed.

The worker allows one foreground session. Qwen2 retains its 1 GiB shard limit. The hybrid path allows a 3 GiB shard file and an 8,192-token ceiling, with a 384 MiB admission reserve. The USB payload limit is 8 MiB. Configured context limits are not claims that long-context execution has been tested.

## Next acceptance gates

1. Resolve the remaining hybrid prefill numerical difference without masking failed criteria.
2. Repeat matched Mac-only and split trials with controlled starting memory conditions; the first split succeeded near its swap guard, while the control crossed that guard.
3. Measure sustained generation, thermal behavior, larger contexts, disconnect recovery, and quality on a fixed task set.

No automatic fallback that secretly loads all phone weights on the Mac is permitted in a capacity run. A disconnected worker ends the session; restart after resetting caches.

## References and attribution

The Python implementation composes pinned upstream `mlx-lm` Qwen2 and Qwen3.5-family blocks. The Swift implementation follows both architectures and ports the MLX-LM recurrent Metal kernel; its MIT license is included in `licenses/MLX-LM-MIT.txt`. Dependencies retain their original licenses. Review licenses before distributing model weights or third-party code.

- [MLX](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [MLX Swift](https://github.com/ml-explore/mlx-swift)
- [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- Prior related projects: [DIM](https://github.com/dannydyl/DIM), [rmcluster iOS](https://github.com/rmcluster/ios-app). Their results have not been reproduced here.

The source repository is public. No package release or model weights are published.
