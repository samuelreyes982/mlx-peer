# MLX Peer iPhone companion and worker

Normal launches open the USB companion screen. Pair once using its six-digit code, select your model in the native Mac app, and let the Mac transfer the phone's weights automatically. See the [companion setup guide](../docs/COMPANION.md). The original fixture/self-test launch arguments remain available for development. App Store and TestFlight distribution are not configured yet.

This package executes an assigned contiguous range of **dense Qwen2 / Qwen2.5 transformer layers**, using MLX Swift, with a local KV cache. It never constructs embeddings, an output head, or unassigned layers. It supports local fixture execution and an opt-in USB developer transport. [Live USB stage execution and small-model Mac–iPhone generation have passed](../docs/USB_PROBE.md); A subsequent [Qwen3.8-27B exploratory split](../docs/QWEN38_SPLIT.md) also generated text; its strict prefill parity and sustained capacity validation remain open.

The package pins `mlx-swift` **0.31.6**, revision `0bb916c67f4b9e5c682cbe02a42c701c93ab5021`. Its package declares MLX 0.31.1. `Package.resolved` pins transitive dependencies. It requires Xcode with Swift 6.3 and the Metal Toolchain component, macOS 14+ or iOS 17+, and Apple silicon for GPU execution. Physical iPhone feasibility remains a separate acceptance test.

Experiment environment: **macOS 26.7 (25G229), Xcode 26.6 (17F113), and iPhone 16 with iOS 27.0 (24A437)**. These versions were checked locally. The signed app was installed and run on the physical phone; the self-test and trained Qwen2 stage comparison passed. These experiment versions are separate from the package's minimum deployment targets.

## Verification on 2026-09-15

- Xcode 26.6 / Swift 6.3.3: macOS CLI build passed, including Metal shaders.
- Generic arm64 iOS device: unsigned app build passed. No phone installation or execution is implied.
- Three Swift validation tests passed, including traversal and symlink-alias rejection.
- The CLI executed all three cached steps of the tiny float32, tiny float16, and trained Qwen2.5-0.5B float16 fixtures on the Mac. The trained stage held layers 4–7 with 119,299,072 bytes of weights. Python comparison results belong to the repository's experiment artifacts.
- The local self-test measured zero maximum absolute difference between full prefill and cached continuation. This is internal consistency evidence only.

The physical iPhone self-test and trained Qwen2 fixture now pass. The phone held 119,299,072 bytes of assigned weights and matched all three Python-reference outputs at the predeclared FP16 tolerances. See [physical-device results](../docs/IPHONE_VALIDATION.md). A subsequent USB test also passed live stage execution, full-model numerical comparisons, and a 24-token continuation across both devices. The new hybrid stage supports 4-bit Qwen3.8 text layers. A 12-layer phone shard has completed an exploratory 27B generation, with an unresolved prefill numerical check; see the dedicated result report.

## Build on the Mac

From this `ios` directory:

```sh
xcodebuild build \
  -scheme mlx-peer-stage \
  -destination 'platform=macOS' \
  -derivedDataPath DerivedData \
  -clonedSourcePackagesDirPath .build/SourcePackages \
  -skipPackagePluginValidation \
  CODE_SIGNING_ALLOWED=NO

DerivedData/Build/Products/Debug/mlx-peer-stage run-fixture ../artifacts/stage-f32
DerivedData/Build/Products/Debug/mlx-peer-stage self-test
```

Xcode builds the Metal shaders; `swift build` alone can compile-check Swift/C++ but does not produce the necessary Metal library. If Xcode reports a missing Metal Toolchain, install that Apple component through Xcode Settings → Components or `xcodebuild -downloadComponent MetalToolchain`.

The pinned MLX Swift release attaches a `CudaBuild` plugin on all platforms. Its inspected implementation returns no build commands on Apple platforms (`isCudaEnabled()` is compile-time false outside Linux). The flag above skips Xcode's interactive trust prompt **for this invocation only**; it does not change a persistent setting. If building in Xcode, review/trust the pinned package plugin there instead.

## File protocol v1

The fixture directory contains:

- `config.json`: original Hugging Face Qwen2 configuration.
- `weights.safetensors`: only assigned `model.layers.N.*` tensors with original indices and key names. Float16 or float32, one consistent dtype; dense only.
- `request.json`: request below.
- Input safetensors files: one `hidden_states` array of shape `[1, tokens, hidden_size]` matching the weights' dtype.

```json
{
  "protocol_version": 1,
  "layer_start": 1,
  "layer_end": 3,
  "session_id": "probe-001",
  "position": 0,
  "max_context": 256,
  "steps": [
    {"input": "input-0.safetensors", "output": "output-0.safetensors", "position": 0},
    {"input": "input-1.safetensors", "output": "output-1.safetensors", "position": 5}
  ]
}
```

`layer_end` is exclusive. The same stage instance executes all steps and retains cache. Position must match the cache exactly. If `steps` is omitted, one `input.safetensors` → `output.safetensors` request is run. Initial position is zero. Each output contains `hidden_states`. `report.json` records shape, dtype, materialized compute time, MLX memory counters, assigned weight bytes, and runtime information. Output tensors/report are local files; no prompt text or tensors go to a service.

The fixture runner accepts single filenames, rejects symlinks escaping the directory and output paths overlapping inputs, and limits file/context/request sizes. It is a developer probe for locally generated fixtures, not a hardened untrusted-file service. `system` and memory reports do not cryptographically establish hardware identity. The caller must correlate a physical-device run with its returned artifacts.

The original Qwen2 fixture path retains its limits: batch size one, context at most 4096, shard file at most 1 GiB, and activation file at most 64 MiB. The separate `qwen3_5` hybrid path accepts uniform affine 4-bit/group-64 weights and BF16 activations, up to a 3 GiB shard and 8,192-token ceiling, with a 384 MiB loading reserve. Long-context execution is not yet validated.

## Run on a physical iPhone

1. Open `App/MLXPeerApp.xcodeproj` in Xcode.
2. Select the **MLXPeerApp** target → Signing & Capabilities. Choose your Apple development team and a unique bundle identifier. The repository does not include a development-team ID, credentials, or a provisioning profile.
3. Connect and trust the iPhone, enable Developer Mode if Xcode requires it, select the physical device, and build/run. No App Store upload is involved.
4. Keep the app in the foreground. **Run device self-test** runs a tiny deterministic Qwen2 stage and compares full prefill to cached continuation on that device.
5. For the independent cross-language test, copy a Python-generated fixture folder to the iPhone's Files storage, choose **Open a Python reference fixture**, and select that folder. Return the output safetensors and `report.json` to the Mac for the Python comparator.

The self-test checks internal cache consistency. It does **not** prove Python/Swift parity, network sharding, or increased model capacity. Normal app launch has no listener. The opt-in `--serve-fixture` development mode described below opens a loopback listener for USB access; there is no discovery, background service, or telemetry.

### Developer launch probes

The development app accepts two optional launch arguments, through Xcode's scheme or a local `devicectl` launch:

- `--run-self-test`: runs the same self-test on the worker's serial queue, displays its report, and atomically writes `Documents/self-test.json` in the app container. On iOS the report includes `availableProcessMemoryBytes` from Apple's `os_proc_available_memory()`, captured after the probe, plus MLX active/cache/peak counters and runtime versions. This headroom is a changing process estimate, not physical RAM or a guaranteed budget for the next model.
- `--run-fixture NAME`: runs a fixture from the direct child folder `Documents/NAME`. Copy the fixture into the app's data container using Xcode or `devicectl` first. Only a single basename is allowed; paths and symbolic links are rejected. Outputs and `report.json` remain in that folder for copying back to the Mac.

Use one probe per launch and keep the app in the foreground. Neither of these two flags starts a network listener, installs model code, or runs in the background. The normal manual buttons remain available. A failed run is shown in the app; the presence of files from an earlier run alone does not establish a new run, so use a fresh fixture folder/session or remove the old self-test report before measuring.

For live USB testing, `--serve-fixture NAME` loads the local FP16 Qwen2 or BF16 hybrid fixture and reads a 64-character lowercase hex token from `wire-token.txt` inside that fixture. It binds only to phone loopback port 49172. The Mac's USB-only client forwards the connection via usbmux. This mode is mutually exclusive with the file/self-test launch modes. See [protocol, measured results, limits, and reproduction](../docs/USB_PROBE.md). No automatic server starts during normal app launch.

Unsigned compile check (does not install or run on a phone):

```sh
xcodebuild build \
  -project App/MLXPeerApp.xcodeproj -scheme MLXPeerApp \
  -destination 'generic/platform=iOS' \
  -derivedDataPath DerivedData-iOS \
  -clonedSourcePackagesDirPath .build/SourcePackages \
  -skipPackagePluginValidation CODE_SIGNING_ALLOWED=NO
```

## Implementation and upstream references

- `QwenStage.swift`: original dense Qwen2 stage.
- `HybridStage.swift` and `HybridDeltaKernel.swift`: validated quantized tensor layouts, Gated DeltaNet recurrent state, full-attention caches, and explicit evaluation boundaries for Qwen3.8 text layers.
- `FixtureRunner.swift`: bounded file contract, sequential execution, outputs and measurements.
- `SelfTest.swift`: tiny deterministic prefill/cache consistency check.
- `App/MLXPeerApp.swift`: foreground SwiftUI probe app.

The implementation follows the Qwen2 architecture and uses upstream public APIs. Primary references: [MLX Swift](https://github.com/ml-explore/mlx-swift/tree/0.31.6), [MLX Swift Qwen2 model](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLLM/Models/Qwen2.swift), [MLX Python Qwen2](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py). MLX is a separate third-party dependency with its own license.
