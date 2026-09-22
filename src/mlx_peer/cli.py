"""Small commands with explicit experiment boundaries."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys


def doctor():
    versions = {}
    for name in ("mlx", "mlx-lm", "numpy", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(python=platform.python_version(), platform=platform.platform(),
                  architecture=platform.machine(), packages=versions,
                  network_worker_implemented=False, physical_iphone_verified=False)
    try:
        import mlx.core as mx
        result["metal"] = mx.device_info()
        result["active_memory_bytes"] = mx.get_active_memory()
        result["peak_memory_bytes"] = mx.get_peak_memory()
        result["mlx_probe"] = (mx.array([1, 2]) + 1).tolist() == [2, 3]
    except (ImportError, RuntimeError) as error:
        result["mlx_error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description="MLX Peer capacity experiment — development build")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    inspect = sub.add_parser("inspect", help="Read checkpoint headers; no tensor allocation")
    inspect.add_argument("checkpoint", type=Path)
    split = sub.add_parser("split", help="Stream checkpoint into separate device files")
    split.add_argument("checkpoint", type=Path)
    split.add_argument("output", type=Path)
    split.add_argument("--start", type=int, required=True)
    split.add_argument("--end", type=int, required=True, help="exclusive")
    plan = sub.add_parser("plan", help="Estimate per-device weight/cache budgets; not a fit guarantee")
    plan.add_argument("checkpoint", type=Path)
    plan.add_argument("--start", type=int, required=True)
    plan.add_argument("--end", type=int, required=True)
    plan.add_argument("--context", type=int, default=4096)
    plan.add_argument("--cache-bytes", type=int, default=2)
    plan.add_argument("--mac-budget-gib", type=float)
    plan.add_argument("--iphone-budget-gib", type=float)
    probe = sub.add_parser("make-fixture", help="Create a tiny random Qwen stage correctness fixture")
    probe.add_argument("output", type=Path)
    probe.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    checkpoint_probe = sub.add_parser("checkpoint-fixture", help="Reference validation only; full small model (max 2 GiB) loaded")
    checkpoint_probe.add_argument("checkpoint", type=Path)
    checkpoint_probe.add_argument("output", type=Path)
    checkpoint_probe.add_argument("--start", type=int, required=True)
    checkpoint_probe.add_argument("--end", type=int, required=True)
    bundle = sub.add_parser("pack-fixture", help="Copy only the assigned weights and inputs for a phone test")
    bundle.add_argument("fixture", type=Path)
    bundle.add_argument("output", type=Path)
    for name in ("verify-local", "verify-swift"):
        cmd = sub.add_parser(name)
        cmd.add_argument("fixture", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "inspect":
            from .manifest import inspect_checkpoint
            result = inspect_checkpoint(args.checkpoint).to_dict()
        elif args.command == "split":
            from .manifest import export_shards
            result = export_shards(args.checkpoint, args.output, args.start, args.end)
        elif args.command == "plan":
            from .manifest import inspect_checkpoint, partition_tensors
            from .capacity import estimate_capacity
            checkpoint = inspect_checkpoint(args.checkpoint)
            shards = partition_tensors(checkpoint, args.start, args.end)
            result = estimate_capacity(checkpoint.config,
                mac_weight_bytes=sum(t.nbytes for t in shards["mac"]),
                iphone_weight_bytes=sum(t.nbytes for t in shards["iphone"]),
                remote_layer_count=args.end - args.start, context_length=args.context,
                cache_dtype_bytes=args.cache_bytes,
                mac_budget_bytes=None if args.mac_budget_gib is None else int(args.mac_budget_gib * 2**30),
                iphone_budget_bytes=None if args.iphone_budget_gib is None else int(args.iphone_budget_gib * 2**30))
        elif args.command == "make-fixture":
            from .probe import make_fixture
            result = make_fixture(args.output, args.dtype)
        elif args.command == "checkpoint-fixture":
            from .probe import make_checkpoint_fixture
            result = make_checkpoint_fixture(args.checkpoint, args.output, args.start, args.end)
        elif args.command == "pack-fixture":
            from .bundle import pack_fixture
            result = pack_fixture(args.fixture, args.output)
        elif args.command == "verify-local":
            from .probe import verify_local
            result = verify_local(args.fixture)
        else:
            from .probe import verify_swift
            result = verify_swift(args.fixture)
        print(json.dumps(result, indent=2))
        return 1 if result.get("passed") is False else 0
    except (ValueError, OSError, ImportError) as error:
        print(f"mlx-peer: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
