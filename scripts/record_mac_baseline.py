#!/usr/bin/env python3
"""Record a bounded, offline MLX-LM control without changing SiphonNet.

The parent does not import MLX. It can terminate a stalled Metal child and keeps
system samples even when the child never returns from loading or generation.
This is a short text-only diagnostic, not a reproduction of SiphonNet's UI.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=3)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def sample(pid=None):
    swap = command(["/usr/sbin/sysctl", "vm.swapusage"])
    match = re.search(r"used\s*=\s*([\d.]+)([KMG])", swap.get("stdout", ""))
    used = float(match[1]) * {"K": 1024, "M": 1024**2, "G": 1024**3}[match[2]] if match else None
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "swap_used_bytes": used,
        "swap": swap,
        "vm_stat": command(["/usr/bin/vm_stat"]),
        "memory_pressure_query": command(["/usr/bin/memory_pressure", "-Q"]),
        "process": command(["/bin/ps", "-p", str(pid), "-o", "pid=,rss=,%cpu=,etime="]) if pid else None,
    }


def fingerprint(model):
    index = json.loads((model / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()) | {
        "config.json", "model.safetensors.index.json", "tokenizer_config.json",
        "tokenizer.json", "chat_template.jinja", "generation_config.json",
    })
    result = {}
    for name in names:
        path = (model / name).resolve()
        if path.parent != model:
            raise ValueError("Unexpected checkpoint path")
        if not path.exists():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024**2), b""):
                digest.update(chunk)
        result[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def worker(args):
    run = Path(args.output)
    started = time.monotonic()
    events = (run / "events.jsonl").open("x", buffering=1)

    def event(kind, **values):
        row = {"event": kind, "elapsed_seconds": time.monotonic() - started, **values}
        events.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)

    result = {"status": "error", "physical_devices": 1, "iphone_used": False}
    try:
        event("import_start")
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        versions = {name: importlib.metadata.version(name) for name in
                    ("mlx", "mlx-lm", "transformers", "tokenizers")}
        mx.random.seed(7)
        mx.set_cache_limit(256 * 1024**2)
        event("load_start", versions=versions, device=mx.device_info())
        load_start = time.monotonic()
        model, tokenizer = load(args.model, tokenizer_config={
            "local_files_only": True, "trust_remote_code": False,
        })
        model.freeze()
        model.eval()
        mx.eval(model.parameters())
        load_seconds = time.monotonic() - load_start
        event("load_complete", load_seconds=load_seconds,
              mlx_active_bytes=mx.get_active_memory(), mlx_peak_bytes=mx.get_peak_memory())
        prompt = (run / "prompt.txt").read_text()
        tokens = list(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        ))
        if len(tokens) + args.max_tokens > 8192:
            raise ValueError("Prompt plus output exceeds the declared 8192-token limit")
        write_json(run / "prompt-tokens.json", tokens)
        event("generation_start", prompt_tokens=len(tokens), max_tokens=args.max_tokens)
        generation_start = time.monotonic()
        first_token = None
        first_text = None
        last = None
        with (run / "response.txt").open("x", buffering=1) as output:
            for response in stream_generate(
                model, tokenizer, tokens, max_tokens=args.max_tokens,
                sampler=make_sampler(temp=0), prefill_step_size=128,
                prompt_progress_callback=lambda n, total: event("prefill", processed=n, total=total),
            ):
                elapsed = time.monotonic() - generation_start
                if first_token is None:
                    first_token = elapsed
                if first_text is None and response.text.strip():
                    first_text = elapsed
                output.write(response.text)
                event("token", token=response.token, text=response.text,
                      generation_tokens=response.generation_tokens,
                      generation_elapsed_seconds=elapsed,
                      mlx_active_bytes=mx.get_active_memory(), mlx_peak_bytes=mx.get_peak_memory())
                last = response
        result.update(
            status="completed", versions=versions, load_seconds=load_seconds,
            first_token_after_generation_start_seconds=first_token,
            first_nonwhitespace_text_after_generation_start_seconds=first_text,
            generation_seconds=time.monotonic() - generation_start,
            prompt_tokens=len(tokens), generation_tokens=last.generation_tokens,
            prompt_tokens_per_second=last.prompt_tps,
            generation_tokens_per_second=last.generation_tps,
            finish_reason=last.finish_reason, mlx_peak_bytes=mx.get_peak_memory(),
        )
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        event("error", **result)
    finally:
        result["worker_elapsed_seconds"] = time.monotonic() - started
        result["process_max_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(run / "worker-result.json", result)
        events.close()
    return 0 if result["status"] == "completed" else 1


def stop(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def main(args):
    model = Path(args.model).resolve(strict=True)
    run = Path(args.output).resolve()
    run.mkdir(parents=True, exist_ok=False)
    prompt = "In one short sentence, explain why the sky appears blue."
    (run / "prompt.txt").write_text(prompt)
    print("Hashing the local checkpoint before the timed run…", flush=True)
    files = fingerprint(model)
    write_json(run / "checkpoint-files.json", files)
    config = json.loads((model / "config.json").read_text())
    plan = {
        "scope": "Single short text-only Mac diagnostic using MLX-LM; not SiphonNet UI reproduction or a long-context benchmark",
        "model_path": str(model), "model_type": config["model_type"],
        "quantization": config.get("quantization"), "max_tokens": args.max_tokens,
        "context_limit": 8192, "enable_thinking": False, "temperature": 0,
        "seed": 7, "prefill_step_size": 128, "mlx_cache_limit_bytes": 256 * 1024**2,
        "wired_limit_policy": "Unmodified MLX-LM stream_generate: device recommended working set during generation",
        "text_only": True, "vision_weights_loaded": False, "adapters": False,
        "deadline_seconds": args.timeout, "max_additional_system_swap_gib": args.max_swap_gib,
        "stop_interpretation": "A guard stop is not proof of an out-of-memory failure",
        "cache_state": "OS file cache uncontrolled; fingerprinting reads all weights before timing",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run / "plan.json", plan)
    baseline = sample()
    write_json(run / "system-before.json", baseline)
    if baseline["swap_used_bytes"] is None:
        raise RuntimeError("Cannot read swap usage; refusing an unmonitored run")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    argv = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
            "--model", str(model), "--output", str(run), "--max-tokens", str(args.max_tokens)]
    started = time.monotonic()
    reason = None
    with (run / "worker.log").open("x") as log, (run / "system-samples.jsonl").open("x", buffering=1) as samples:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while child.poll() is None:
                row = sample(child.pid)
                row["elapsed_seconds"] = time.monotonic() - started
                samples.write(json.dumps(row) + "\n")
                if row["elapsed_seconds"] >= args.timeout:
                    reason = "deadline_guard"
                elif row["swap_used_bytes"] is None:
                    reason = "monitor_unavailable"
                elif row["swap_used_bytes"] - baseline["swap_used_bytes"] >= args.max_swap_gib * 1024**3:
                    reason = "additional_swap_guard"
                if reason:
                    stop(child)
                    break
                time.sleep(1)
        finally:
            stop(child)
    status = "guard_stopped" if reason else ("completed" if child.returncode == 0 else "worker_failed")
    report = {"status": status, "stop_reason": reason, "exit_code": child.returncode,
              "elapsed_seconds": time.monotonic() - started, "artifact_directory": str(run)}
    write_json(run / "supervisor-result.json", report)
    write_json(run / "system-after.json", sample())
    print(json.dumps(report, indent=2), flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-swap-gib", type=float, default=4)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.timeout <= 0 or args.max_swap_gib <= 0:
        parser.error("Token, time and swap limits must be positive")
    sys.exit(worker(args) if args.worker else main(args))
