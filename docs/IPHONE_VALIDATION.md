# Physical iPhone worker validation

This report describes the initial file-based test. The subsequent [USB probe](USB_PROBE.md) passed live stage execution and small-model generation across the Mac and iPhone. Qwen3.8 capacity remains unproven.

On 2026-09-15, the signed MLX Peer app was installed and executed on the user's physical **iPhone 16 running iOS 27.0 (24A437)**. The self-test and independent comparison against saved Mac results both passed.

| Check | Observed result |
| --- | --- |
| Tiny self-test: full prefill versus cached continuation | Passed; maximum absolute difference 0 |
| Available app memory after that self-test | 3,495,819,520 bytes (3.50 GB / 3.26 GiB) |
| Real model fragment | Qwen2.5-0.5B, layers 4–7, FP16 |
| Weights held by the phone stage | 119,299,072 bytes |
| Peak MLX allocation during that fixture | 121,983,771 bytes |
| Five-token prefill compared with Python reference | Passed; maximum absolute error 0.0078125 |
| One-token cached continuation | Passed; maximum absolute error 0.00390625 |
| Three-token cached continuation | Passed; maximum absolute error 0.00390625 |

The FP16 comparison used the existing, predeclared relative and absolute tolerances of 0.01. Returned tensors also matched the reference shapes and dtypes. The phone's returned weights, configuration, request, and input files matched the original SHA256s. The model fragment contains real trained weights; the tiny self-test is a separate internal consistency check.

The app reported materialized compute times of 779.65 ms, 50.55 ms, and 17.03 ms for these three steps. These are small single-run measurements, include first-use effects, and exclude transfer overhead. They are not end-to-end token throughput or evidence of a speedup.

The Mac used Xcode's device tooling to install, launch, and transfer the fixture over USB. The app itself has no inference-network connection yet. It executed a complete fixture locally and returned output files afterward. This demonstrates physical-phone shard correctness, not live split text generation.

The user reports using a Thunderbolt 4 USB-C cable. The connected iPhone is visible as a USB device, and [Apple specifies USB 2 up to 480 Mb/s for iPhone 16](https://www.apple.com/iphone-16/specs/). The cable does not provide a Thunderbolt link to this phone. Actual application throughput and request latency remain unmeasured; record them once the inference transport is implemented.

## Implications for the 27B experiment

The phone's measured memory headroom makes a larger shard worth testing. It is an instantaneous estimate, not a demonstrated allocation budget. It includes space needed for future weights, model state, temporary tensors, and app/runtime growth; it cannot all be assigned to stored weights.

Reading the installed Qwen3.8-27B 4-bit checkpoint headers gives these exact storage sizes:

| Candidate language layers | Stored weight bytes | GiB |
| --- | --- | --- |
| 0–7 | 1,712,808,576 | 1.60 |
| 0–9 | 2,144,138,752 | 2.00 |
| 0–11 | 2,569,212,864 | 2.39 |

Twelve layers are a candidate for profiling, not an approved fit. The current phone worker supports dense Qwen2 only and has a 1 GiB fixture guard. The Qwen3.8 architecture, 4-bit execution, and larger-shard handling must be implemented and checked before such a run. A complete model run also needs communication between the Mac coordinator and phone worker.

**No Qwen3.8 layer has run on the phone, and no 27B capacity or speed improvement has been demonstrated.** The completed result establishes that a real phone-side model fragment can compute the expected output.

The next implementation step is Qwen3.8/4-bit stage support, followed by a validated larger-memory probe and live cross-device execution. The final comparison must use the recorded 27B checkpoint and matching settings, with per-device memory measurements and correct generated output.

See [IPHONE_VALIDATION_RESULT.json](IPHONE_VALIDATION_RESULT.json) for structured results. Installation/launch records, the phone reports, returned tensors, comparison artifacts, and the 27B layer-size calculation are in `artifacts/iphone-validation-20260915-01/`. The app's self-reported hardware identity is not cryptographically verified; these results are associated with the physical phone through the paired-device installation, launch, and file retrieval records.
