# Qwen3.8-27B functional retest — September 16, 2026

The Mac+iPhone split completed all five short functional cases on the second attempt. The iPhone executed layers 0–11 with 2.57 GB of resident model weights and answered 46 forward requests. Both devices cleared model state between cases. This is a basic functionality result; the unresolved numerical comparison and memory sensitivity remain.

| Case | Actual answer | Result |
| --- | --- | --- |
| 17 × 23 | `391` | Exact answer passed |
| 4 red + 7 blue marbles, give away 2 | `9` | Exact answer passed |
| Requested JSON fields and values | `{"status":"ok","count":3}` | JSON structure and values passed |
| Repeat multiplication after other prompts and reset | `391` | Entire generated token sequence matched the first case |
| Explain the blue sky in one sentence | Complete sentence explaining that shorter blue wavelengths scatter more effectively in the atmosphere | Manual review passed; not a general quality benchmark |

The full monitored process took **28.72 seconds**, excluding shard fingerprinting and phone provisioning. Aggregate decoding after excluding each prompt's time to first token was **3.60 tokens/second**. First-token latency was 6.66 seconds for the first case and 0.78–1.61 seconds for subsequent cases. These are descriptive measurements on very short outputs, with uncontrolled file and kernel caches.

## Both attempts matter

The first attempt used the original **4 GiB additional-system-swap cutoff**. It stopped after 7.39 seconds during Mac loading, at 4.08 GiB additional swap, with zero generated tokens. Its starting system swap was 3.10 GiB. Cleanup completed successfully.

The second attempt was explicitly declared with a **6 GiB additional-swap allowance**, the same 180-second deadline, and the same 384 MiB phone-headroom reserve. Its starting swap was already 7.22 GiB after the first attempt. The observed additional swap peaked at **3.19 GiB**, which was below even the original threshold. The higher configured cutoff was therefore not reached. Changed starting OS memory conditions prevent attributing success solely to the cutoff change or treating this as a controlled reliability comparison.

Minimum sampled phone process headroom was **830 MB**. The worker stopped on completion; no benchmark child remained running. No other user applications were closed, no system-wide memory settings were changed, and no new model weights were downloaded.

## Numerical check

A fresh physical-device run reproduced the previous result at unchanged `rtol=0.03, atol=0.03`: cached single-token and multi-token continuation passed; initial 12-layer prefill failed. Relative RMS differences were 0.70%, 0.94%, and 0.64%. The full-model prompt run therefore retained the explicit exploratory label. Correct answers to these five simple cases do not erase that failed check.

The [structured report](QWEN38_FUNCTIONAL_RETEST_RESULT.json) includes answers, token IDs, per-case timings, memory results, both attempts, and limitations. Raw evidence is in `artifacts/qwen38-retest-20260916/`. The exact tested recorder is saved with its verified hash in the completed run folder.

## Repeat the functional suite

The cases and expected answers are declared before execution in `experiments/functional_smoke.json`. Unlock the connected iPhone and launch the existing worker with `--serve-fixture qwen38-split12`. Keep it in the foreground. From the project root, choose a new output directory:

```sh
.venv/bin/python scripts/record_hybrid_split.py \
  --model /path/to/Qwen3.8-27B-MLX-4bit \
  --fixture artifacts/qwen38-split12-mlx031 \
  --output artifacts/new-functional-run \
  --serial DEVICE_SERIAL --token-file /path/to/private/wire-token.txt \
  --suite experiments/functional_smoke.json --max-swap-gib 6 --exploratory
```

The original 4 GiB guard remains the recorder's default. The suite loads the Mac shard once, resets both stages between prompts, and saves every completed case. Its graders check literal arithmetic answers, JSON types/values, EOS completion, and repeated token IDs. It does not use another model to judge answers. Targeted grader checks passed for correct/wrong answers, truncated output, JSON type mismatches, and repeat mismatches.
