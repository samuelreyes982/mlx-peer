# Matched 9B sanity test

The installed **Qwen3.5-9B-MLX-4bit** model successfully loaded and answered the test prompt on 2026-09-15. The recorder script, prompt, runtime versions, and all recorded execution settings match the initial [27B diagnostic](MAC_BASELINE.md). Only the model path and start time differ in the run plans.

| Measurement | Qwen3.5-9B 4-bit | Qwen3.8-27B 4-bit |
| --- | --- | --- |
| Outcome | Completed with a correct answer | Additional-swap guard stopped loading |
| Model load time | 1.63 seconds | Did not finish |
| First token after generation started | 0.93 seconds | No token observed |
| Decode speed reported by MLX-LM | 27.81 tokens/second | Not measured |
| Generated token count | 33 including the end-of-sequence token | 0 observed |
| Peak MLX allocation | 5,194,022,333 bytes (5.19 GB / 4.84 GiB) | Unavailable; worker stopped during loading |
| Additional system swap observed | 0 | 4.34 GiB at the stopping sample |

Prompt: “In one short sentence, explain why the sky appears blue.”

Actual answer:

> The sky appears blue because Earth's atmosphere scatters shorter wavelengths of sunlight (blue light) more effectively than longer wavelengths due to a process called Rayleigh scattering.

Both runs used a 64-token output ceiling, thinking disabled, temperature 0, seed 7, 128-token prefill chunks, a 256 MiB MLX allocation-cache limit, a 180-second deadline, and a stop at 4 GiB of additional system swap. The 9B answer finished naturally before the output ceiling. No other local model process was found before the run. This was standalone MLX-LM execution using installed SiphonNet model files.

The initial OS memory states were **not identical**. The 9B run began with about 3.82 GiB of existing system swap, which did not increase during the observed run, and more reported memory headroom. Ordinary applications remained open. These are sequential sanity tests, not a controlled repeated performance benchmark. MLX's peak allocation is not total process memory; system swap includes other applications. No claim about sustained generation, long contexts, or comparative answer quality follows from this one prompt.

The result confirms that this recorder and local MLX environment can run the installed 9B model. The larger model produced substantial paging during loading under its recorded conditions. Its guard stop remains distinct from an observed fatal out-of-memory failure.

The machine-readable result is [MAC_9B_SANITY_RESULT.json](MAC_9B_SANITY_RESULT.json). Raw logs, prompt, answer, memory samples, configuration, script snapshot, and weight hashes are saved under `artifacts/mac-baseline-qwen35-9b-20260915-01/`. Both 9B weight hashes match local download metadata for revision `8b2b98c00a6b4d291155e4890773ca8f769aee53`.

To repeat with a fresh output directory:

```sh
.venv/bin/python -u scripts/record_mac_baseline.py \
  --model /path/to/models/Qwen3.5-9B-MLX-4bit \
  --output artifacts/mac-baseline-qwen35-9b-repeat \
  --max-tokens 64 --timeout 180 --max-swap-gib 4
```
