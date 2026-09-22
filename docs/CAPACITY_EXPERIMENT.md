# Capacity experiment protocol

Current status: [one exploratory 27B split generated an answer](QWEN38_SPLIT.md). The remaining numerical and sustained-operation acceptance gates below have not passed.

## Hypothesis

The Mac already runs a useful 9B model. Can partitioning a fixed Qwen3.8-27B configuration across the user's 18 GiB M3 Pro Mac and iPhone 16 enable usable generation when the same configuration on the Mac alone does not?

To test a capacity benefit, the iPhone must retain its assigned weights and model state and execute those layers. Sending calculations to the phone while retaining the complete model on the Mac does not establish the proposed memory benefit.

The experiment targets **iPhone 16 with iOS 27.0 (24A437)**, verified through Xcode's device tool on 2026-09-15. The phone is paired and was connected over USB for installation and physical tests. Its self-test and trained Qwen2 fragment comparison passed; see [physical-device results](IPHONE_VALIDATION.md). The tiny self-test reported 3,495,819,520 bytes of available process memory, but a sustained larger-shard budget has not been measured. The Mac runs **macOS 26.7 (25G229)** with **Xcode 26.6 (17F113)**, verified locally. Storage capacity and total advertised RAM are not app memory budgets.

## Observed baseline and proposed comparison

| Run | Model | Status |
| --- | --- | --- |
| A: useful baseline | Installed Qwen3.5-9B-MLX-4bit | User reports effective SiphonNet operation with context 8,192 and output limit 2,048. A separate [matched 64-token sanity test](MAC_9B_SANITY.md) completed in standalone MLX-LM at 27.81 generated tokens/second with no additional observed swap. |
| B: target control | Installed Qwen3.8-27B MLX 4-bit checkpoint | Initial standalone MLX-LM diagnostic stopped at the predeclared swap guard during loading; no tokens observed. This is separate from the user's earlier SiphonNet nonresponse. See [recorded result](MAC_BASELINE.md). |
| C: capacity test | The same checkpoint, split Mac/iPhone | One short exploratory generation completed with 12 phone layers. Runtime/loading policy was updated; see the new equivalent Mac-only control and limitations in the [report](QWEN38_SPLIT.md). |

A matched Mac-only versus split comparison tests the sharding effect; the historical B run has different runtime/loading settings and is not the equivalent current control. Comparison A versus C measures the practical tradeoff when moving from the working 9B model to 27B. Parameter count alone does not establish better answers; any quality claim needs a separate fixed task set.

Before B/C, capture SiphonNet's exact model ID, runtime versions, model-loading error or logs, time to first token, time to first visible answer, output-token accounting, Metal memory configuration, process footprint and swap. No response can reflect loading, paging, prompt processing, hidden reasoning, an unsupported operation or an application error. Do not label it an out-of-memory failure without evidence. Run one model at a time; do not leave the 9B resident during the 27B control.

## Controlled configuration

Record checkpoint/revision and shard SHA256s, architecture/config, quantization and cache dtype, runtime versions, prompt tokens, maximum context, generated-token count, sampling settings, power state and network. Keep these fixed between the principal Mac-only and split comparison. Publish a separate practical quantized Mac-only comparison when relevant.

Record the user's current 8,192 context and 2,048 output settings as the practical target, with the runtime's actual interpretation of those limits. Start diagnosis at a shorter prompt/output, then repeat B and C with identical target limits. Hold reasoning mode, template, tools/learning behavior and vision policy constant. Text-only is the first scope; if vision weights are omitted, omit them consistently and account for the actual loaded text weights.

Predeclare usable-memory and responsiveness criteria after measuring the devices, before choosing a favorable result. A Metal recommended working-set limit is not physical RAM exhaustion. Disk swapping can make a model slow without making execution impossible. Record both and use precise language.

## Measurements

- Mac physical RAM, Metal recommended working set, MLX active/peak/cache allocations, process footprint, memory pressure and swap change.
- Phone app footprint, `os_proc_available_memory()` headroom, MLX allocations, thermal state and foreground lifetime. Do not assume memory entitlements raise the limit on every phone.
- Stored and resident bytes for each shard, full-attention KV cache, recurrent state and convolution history; confirm only assigned layers are instantiated and no hidden full-model copy exists.
- Cold and warm load time, time to first token, prefill throughput, decode throughput, end-to-end latency, serialization and network time.
- Repeated prompts plus at least 15 minutes of sustained generation, including errors and interruptions.

## Target checkpoint and architecture

The confirmed model family is **Qwen3.8-27B**. The installed checkpoint is [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit), local download revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`. All three weight-file SHA256s were checked against the original download record for the Mac-only diagnostic. Pin this installed version for the eventual split comparison. An earlier metadata-only investigation used a newer revision, `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`; those metadata should not silently replace the tested checkpoint.

The installed weight index reports **16,054,262,240 bytes** of tensor storage (16.05 decimal GB / approximately 14.95 GiB). The configuration specifies affine 4-bit quantization, group size 64. The existing SiphonNet model files were reused; no additional 27B weight download was required. The independent diagnostic does not establish the precise cause of the user's prior application-level failure.

The user reports a **24 GB memory requirement**. Its scope is unverified: a recommendation for a computer's total RAM differs from the model process itself needing 24 GB. Both operating systems consume memory, iOS imposes app memory limits, and distributed execution adds some overhead. Adding the devices' advertised RAM cannot establish that the model fits. Compare the actual per-device weights, state and temporary allocations with measured usable process budgets.

That storage figure is not resident runtime memory: format conversion, nonquantized tensors, caches/state, workspace, loading peaks and other processes also matter. It also includes the checkpoint's stored components, which may differ from a text-only runtime's loaded subset. A few GiB placed on the phone may be useful, but no split size is promised before measuring phone headroom and per-layer sizes.

[Qwen's architecture description](https://huggingface.co/Qwen/Qwen3.8-27B) specifies 64 language layers arranged as 16 groups of three Gated DeltaNet layers plus one full-attention layer. Qwen3.8 retains the Qwen3.5 architecture; its configuration's `qwen3_5` identifier is intentional. The [MLX-LM implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py) uses recurrent-state caches for linear-attention layers and KV caches for full-attention layers. Our dense-Qwen2-only state estimator and layer adapter cannot be applied unchanged.

Implementation requirements (implemented for the exploratory run; strict prefill parity remains open):

1. Inspect and stream-partition the Qwen3.8 checkpoint's actual tensor namespace, keeping quantized weights, scales and biases together; explicitly handle the text-only/vision policy.
2. Execute matching 4-bit operations on Mac Python and iPhone Swift without expanding every weight to FP16 in memory. Validate sanitization, layout and quantization metadata against pinned references.
3. Implement Gated DeltaNet/recurrent state and full-attention/RoPE handling. Each device owns state for its assigned layers; transfer only boundary activations and required session metadata during generation.
4. Validate both layer types, prefill, single-token decoding, multi-token continuation and resets on a model that fits on the Mac before attempting 27B.
5. Measure per-layer storage/residency and select a contiguous phone range that fits its measured budget. Complete four-layer hybrid groups are an initial profiling option, not a correctness requirement or a promised split.
6. The hybrid path now permits a 3 GiB shard file and an 8,192-token ceiling with an admission reserve. A full 8,192-token context test remains outstanding.

The original Qwen2.5-0.5B FP16 fixture remains a regression test, not the target capacity demonstration. Its converted weights total 988,065,536 bytes; layers 4–7 use 119,299,072 bytes. The earlier exploratory 7B/14B/32B candidate list is superseded by this user-driven 9B-to-27B experiment.

## Acceptance

1. Small-model split execution agrees numerically with the complete reference, including prefill and cache continuation.
2. Mac-only either fails within the declared resource configuration or violates a predeclared practical paging/responsiveness criterion. Report which.
3. The identical split configuration completes correct repeated generation within both measured process budgets.
4. Each device owns only its assigned resident weights/cache. Exporting a checkpoint must not require loading the full model first.
5. The report identifies tradeoffs, including a split run that is slower but enables a larger model.

Until these pass on two physical devices, the result is an implementation milestone rather than proof of increased capacity.
