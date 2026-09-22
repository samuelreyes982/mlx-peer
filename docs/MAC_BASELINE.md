# First Mac-only Qwen3.8-27B diagnostic

A subsequent [9B sanity test](MAC_9B_SANITY.md) using the same recorder, prompt, and settings completed successfully with no additional swap observed. The starting OS memory states differed; both reports preserve that limitation.

On 2026-09-15, the Mac-only process hit the predeclared additional-swap guard while loading the model. Loading did not complete and no tokens were observed. The process was terminated by the recorder, not by an observed out-of-memory exception.

| Measurement | Result |
| --- | --- |
| Mac | M3 Pro, 18 GiB, macOS 26.7 |
| Model | `mlx-community/Qwen3.8-27B-4bit` |
| Local download revision | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| Weight-file verification | All three SHA256 hashes match the original local download record |
| Runtime | MLX 0.32.2, MLX-LM 0.31.1, Transformers 5.17.0 |
| Timed process duration | 7.65 seconds, including imports and termination |
| Last worker stage | Loading; no `load_complete` event |
| System swap | 0 → 4.34 GiB at the last monitored sample |
| Memory-pressure query | Reported system-wide memory free percentage fell from 55% to 15% |
| Generated tokens observed | 0 |
| Time to first token / throughput | Not measured; generation was never reached |
| Stop | Recorder's 4 GiB additional-system-swap guard; exit signal SIGTERM |

The predeclared limits were 180 seconds and 4 GiB of additional system swap. One-second sampling permits some overshoot. The requested generation used a short fixed prompt, at most 64 output tokens, thinking disabled, greedy sampling, seed 7, 128-token prefill chunks, and a 256 MiB MLX allocation-cache limit. The 8192-token context ceiling was not a filled 8192-token prompt. MLX-LM's ordinary text-only loader was used; no adapter or iPhone was involved.

Prompt: “In one short sentence, explain why the sky appears blue.”

This captures substantial paging during loading under the recorded conditions. It does **not** prove the model can never generate on this Mac, establish a fatal out-of-memory error, or tell us exactly how much memory to move to the phone. System swap includes other applications, and loading peaks can differ from steady-state inference memory.

This was a dedicated offline MLX-LM diagnostic using the model files installed in SiphonNet. It did not reproduce the SiphonNet UI, learning pipeline, or full application configuration. Runtime equivalence with SiphonNet has not been established. There was no local model process running before this test, but ordinary desktop applications remained open. The checkpoint was read for hashing before timing, so this is not a cold-disk benchmark. It is one attempt, not a repeated or long-context benchmark.

The machine-readable result is [MAC_BASELINE_RESULT.json](MAC_BASELINE_RESULT.json). Full logs, model/config/tokenizer hashes, raw memory snapshots, the exact script, and run settings are in `artifacts/mac-baseline-qwen38-27b-20260915-01/`. The local model files and SiphonNet settings were not changed.

To repeat with a new output directory:

```sh
.venv/bin/python -u scripts/record_mac_baseline.py \
  --model /path/to/models/Qwen3.8-27B-MLX-4bit \
  --output artifacts/mac-baseline-qwen38-27b-repeat \
  --max-tokens 64 --timeout 180 --max-swap-gib 4
```

The runner requires ordinary local access to Metal and system memory statistics. The supervisor's time and swap guards were smoke-tested against real child processes before running the model; both terminated the child and retained a report.

For the eventual sharded comparison, preserve this installed checkpoint and its hashes. First measure the phone's usable memory, implement the missing Qwen3.8 worker and transport, and validate small-model execution. Then repeat Mac-only and split trials with matching settings and comparable system load. A guard-stopped control must remain labeled as such; a split run is not successful until it produces correct output and completes the stated workload.
