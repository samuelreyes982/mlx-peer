# Companion 0.2 preview validation

Checked September 22, 2026, on Apple Silicon with the repository's pinned Python dependencies, Python 3.12, and Xcode 26.6 / Swift 6.3.3.

| Check | Result |
| --- | --- |
| Full Python suite | 117 tests and 25 subtests passed |
| Swift validation suite | 5 tests passed |
| Signed iPhone Debug build | Built and installed successfully on the development iPhone |
| Native Mac app build | Compiled successfully |
| Mac bundle | Ad-hoc signature passed deep/strict verification |
| Frozen engine outside source tree | Tiny Qwen GPU inference and real local Qwen tokenizer passed with Python environment overrides removed |
| Mac UI | Opened, layout inspected, model folder selected successfully |
| New protocol physical-device end-to-end check | **Pending**: phone was locked when the corrected app needed to launch |
| Developer ID/notarization | Not completed; no Developer ID Application identity available in the build environment |
| Clean-machine downloaded-app installation | Not validated |
| TestFlight / App Store | Not submitted |

New automated coverage includes invalid/out-of-sequence replies, frame-size bounds, rejection of unpaired access responses, private credential-file permissions, symlink rejection, partial transfer resume, reuse without retransmitting completed files, cancellation before commit, wrong offsets, invalid model identities, checksum failures, immutable committed models, BF16/FP32 conversion, and non-finite weight rejection.

The packaged engine's `--self-check` executes a tiny Qwen model on the Mac GPU. An optional local checkpoint folder also exercises its bundled tokenizer:

```sh
"dist/MLX Peer.app/Contents/Helpers/MLX Peer Engine.app/Contents/MacOS/mlx-peer-engine" --self-check /path/to/local/qwen-folder
```

This is an independent packaging check, not a two-device inference test. Earlier physical iPhone results in [USB_PROBE.md](USB_PROBE.md) validate the original developer transport and model stages. They do **not** establish that the new onboarding, pairing, transfer, and reconnect workflow has passed on a physical device.

Before broad distribution, complete the pending first-pairing → model transfer → reference comparison → chat → reconnect/cancellation test on an unlocked phone, then validate signed downloads and more device/OS combinations. No speedup, reduced resource use, or environmental benefit is established by these checks.
