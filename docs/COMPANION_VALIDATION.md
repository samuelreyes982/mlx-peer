# Companion 0.2 preview validation

Updated September 23, 2026. Mac preview **0.2.0-alpha.2** was tested on Apple Silicon with a physical **iPhone 16 over USB**, running the previously installed iPhone 0.2.0 development build (build 2, protocol v2). The phone app remains compatible with this Mac update. Build tools: pinned Python dependencies, Python 3.12, and Xcode 26.6 / Swift 6.3.3.

| Check | Result |
| --- | --- |
| Full Python suite | 118 tests and 25 subtests passed, including the real Transformers tokenizer regression |
| Swift validation suite | 5 tests passed |
| Signed iPhone Debug build | Built and installed successfully on the development iPhone |
| Native Mac app build | Compiled successfully |
| Mac bundle | Ad-hoc signature passed deep/strict verification |
| Frozen engine outside source tree | Tiny Qwen GPU inference and real local Qwen tokenizer passed with Python environment overrides removed |
| Mac UI | Connected using saved pairing, selected and loaded the chat model, and visibly answered `12 + 7` with `19` |
| New protocol physical-device end-to-end check | **Passed**: authentication, preparation/transfer, numerical reference checks, generation, saved pairing, cached model reuse, cancellation, and reconnect |
| Packaged engine with iPhone | Passed from `/private/tmp` with development Python environment overrides removed; Qwen2.5-0.5B-Instruct answered `2 + 2` with `4`, then `3 + 3` with `6` after reconnecting |
| Developer ID/notarization | Not completed; no Developer ID Application identity available in the build environment |
| Clean-machine downloaded-app installation | Not validated |
| TestFlight / App Store | Not submitted |

New automated coverage includes invalid/out-of-sequence replies, frame-size bounds, rejection of unpaired access responses, private credential-file permissions, symlink rejection, partial transfer resume, reuse without retransmitting completed files, cancellation before commit, wrong offsets, invalid model identities, checksum failures, immutable committed models, BF16/FP32 conversion, and non-finite weight rejection.

## Physical-device results

The initial live chat attempt found a bug in alpha.1: Transformers 5 returns a `BatchEncoding` from `apply_chat_template` by default. Passing that result directly into MLX failed. Alpha.2 explicitly requests token IDs with `return_dict=False`; a regression test exercises the real tokenizer interface. **Use alpha.2 or newer for chat.**

First pairing and automatic model preparation/transfer succeeded. Repeating an already completed transfer sent **zero file chunks**. Saved credentials reconnected without another code. The Qwen2.5-0.5B base checkpoint passed all three phone-stage and all three full-model reference checks under the project's existing FP16 thresholds:

| Comparison | Relative / absolute tolerance | Maximum absolute differences, in request order |
| --- | --- | --- |
| Phone's assigned layers | 0.01 / 0.01 | 0.03125, 0.001708984375, 0.00390625 |
| Full split-model logits | 0.02 / 0.05 | 0.0390625, 0.0234375, 0.0234375 |

Tolerance is evaluated elementwise as `absolute_error <= atol + rtol * abs(reference)`; a maximum absolute difference can exceed `atol` and still pass. Full-model argmax tokens agreed at every tested position. An initial helper used a stricter full-model absolute tolerance of 0.03 and failed its prefill check. That result is retained in the [numerical report](companion-device-20260923.json). The subsequent criterion comes from the existing `scripts/benchmark_usb.py` standard, not a claim of bitwise identity.

For the packaged-app and native-window checks, the separately downloaded **Qwen2.5-0.5B-Instruct** model was converted to FP16. Four layers occupied **119,299,072 weight bytes on the iPhone**; the Mac held **868,766,464 weight bytes**. Initial preparation/transfer/load took approximately 7.93 seconds in this run. This is a single observation with a small model, not a performance guarantee.

The packaged engine generated `2 + 2 equals 4.`, stopped an in-progress response after three tokens when cancelled, answered again immediately afterward, rejected an oversized conversation without dropping the connection, and reconnected/reloaded to generate `6` for `3 + 3`. The native Mac window then visibly answered `The sum of 12 and 7 is 19.`. See the [packaged-app report](companion-packaged-20260923.json).

The original base checkpoint produced repetitive output with its chat template; the same output was reproduced on a complete Mac-only reference. That observation is not attributed to USB or the split model. Use an instruction-tuned checkpoint for conversations. These arithmetic examples verify the app's operation, not broad model accuracy.

The packaged engine's `--self-check` executes a tiny Qwen model on the Mac GPU. An optional local checkpoint folder also exercises its bundled tokenizer:

```sh
"dist/MLX Peer.app/Contents/Helpers/MLX Peer Engine.app/Contents/MacOS/mlx-peer-engine" --self-check /path/to/local/qwen-folder
```

The `--self-check` command is an independent packaging check; it does not use the phone. The separate September 23 checks above exercised the new companion workflow on a physical iPhone. Earlier results in [USB_PROBE.md](USB_PROBE.md) describe the original developer transport.

Before broad distribution, finish Developer ID signing/notarization, clean-machine download validation, TestFlight/App Store distribution, and testing on more device/OS combinations and longer sessions. No speedup, reduced resource use, or environmental benefit is established by these checks.
