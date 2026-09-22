"""Reproducible stage fixtures, intentionally tiny and not language-capable."""
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.qwen2 import Model, ModelArgs
from mlx_lm.models.cache import KVCache

from .runtime import LayerStage, MacCoordinator


def tiny_config():
    return dict(model_type="qwen2", hidden_size=32, intermediate_size=64,
                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                rms_norm_eps=1e-6, vocab_size=128, max_position_embeddings=256,
                rope_theta=1_000_000.0, tie_word_embeddings=True,
                use_sliding_window=False)


def make_fixture(output, dtype="float32"):
    """Create independent Python reference outputs for a Swift stage runner."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = tiny_config()
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    mx.random.seed(719)
    reference = Model(ModelArgs.from_dict(config))
    target_dtype = {"float32": mx.float32, "float16": mx.float16}[dtype]
    weights = {k: v.astype(target_dtype) for k, v in tree_flatten(reference.parameters())}
    reference.load_weights(list(weights.items()))
    mx.eval(weights)
    mx.save_safetensors(str(output / "full.safetensors"), weights, metadata={"format": "mlx-peer"})
    remote = {k: v for k, v in weights.items() if k.startswith(("model.layers.1.", "model.layers.2."))}
    local = {k: v for k, v in weights.items() if k not in remote}
    mx.save_safetensors(str(output / "weights.safetensors"), remote, metadata={"format": "mlx-peer"})
    mx.save_safetensors(str(output / "mac.safetensors"), local, metadata={"format": "mlx-peer"})
    stage = LayerStage(config, output / "weights.safetensors", 1, 3)
    steps = []
    position = 0
    for index, length in enumerate((5, 1, 3)):
        hidden = mx.random.normal((1, length, config["hidden_size"])).astype(target_dtype)
        expected = stage.forward(hidden, position)
        mx.save_safetensors(str(output / f"input-{index}.safetensors"), {"hidden_states": hidden}, metadata={"format": "mlx-peer"})
        mx.save_safetensors(str(output / f"expected-{index}.safetensors"), {"hidden_states": expected}, metadata={"format": "mlx-peer"})
        steps.append(dict(input=f"input-{index}.safetensors", output=f"output-{index}.safetensors", position=position))
        position += length
    request = dict(protocol_version=1, layer_start=1, layer_end=3,
                   session_id="deterministic-stage-probe", position=0, max_context=256,
                   steps=steps)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    metadata = dict(fixture="random-tiny-qwen2", trained_model=False,
                    purpose="cross-runtime correctness only; not a capacity demonstration",
                    dtype=dtype, seed=719, python=platform.python_version(), mlx=mx.__version__,
                    full_weight_bytes=sum(v.nbytes for v in weights.values()),
                    mac_weight_bytes=sum(v.nbytes for v in local.values()),
                    worker_weight_bytes=sum(v.nbytes for v in remote.values()))
    (output / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def verify_local(output):
    """Compare split execution with the complete upstream model on this Mac."""
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    reference = Model(ModelArgs.from_dict(config))
    reference.load_weights(list(mx.load(str(output / "full.safetensors")).items()))
    mx.eval(reference.parameters())
    request = json.loads((output / "request.json").read_text())
    start_layer, end_layer = request["layer_start"], request["layer_end"]
    stage = LayerStage(config, output / "weights.safetensors", start_layer, end_layer)
    coordinator = MacCoordinator(config, output / "mac.safetensors", start_layer, end_layer, stage)
    cache = [KVCache() for _ in range(config["num_hidden_layers"])]
    reports = []
    start = time.perf_counter()
    for token_ids in ([1, 3, 7, 2, 11], [5], [6, 9, 8]):
        tokens = mx.array([token_ids], mx.int32)
        expected = reference(tokens, cache=cache)
        actual = coordinator.forward(tokens)
        mx.eval(expected, actual)
        error = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))).item()
        matched = mx.allclose(actual, expected, rtol=1e-4, atol=1e-5).item()
        reports.append(dict(position=coordinator.position, max_absolute_error=error,
                            passed=bool(matched)))
    result = dict(test="local-partition-reference", physical_devices=1,
                  iphone_executed=False, capacity_demonstrated=False,
                  passed=all(x["passed"] for x in reports), steps=reports,
                  elapsed_seconds=time.perf_counter()-start,
                  mac_layer_indices=list(coordinator.layers), worker_layer_indices=list(range(start_layer, end_layer)),
                  mac_weight_bytes=coordinator.weight_bytes, worker_weight_bytes=stage.weight_bytes)
    (output / "python-report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_swift(output):
    output = Path(output)
    meta = json.loads((output / "fixture.json").read_text())
    request = json.loads((output / "request.json").read_text())
    tolerance = (0.01, 0.01) if meta["dtype"] == "float16" else (1e-4, 1e-4)
    results = []
    for index, step in enumerate(request["steps"]):
        expected = mx.load(str(output / f"expected-{index}.safetensors"))["hidden_states"]
        actual = mx.load(str(output / step["output"]))["hidden_states"]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError("Swift result tensor shape/dtype mismatch")
        error = mx.max(mx.abs(actual.astype(mx.float32)-expected.astype(mx.float32))).item()
        passed = mx.allclose(actual, expected, rtol=tolerance[0], atol=tolerance[1]).item()
        results.append(dict(position=step["position"], max_absolute_error=error, passed=bool(passed)))
    report = dict(test="python-swift-stage-parity", execution_location="see Swift report.json",
                  capacity_demonstrated=False, passed=all(x["passed"] for x in results),
                  rtol=tolerance[0], atol=tolerance[1], steps=results)
    (output / "parity-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def make_checkpoint_fixture(checkpoint, output, start_layer, end_layer):
    """Convert a SMALL reference checkpoint to FP16 and generate stage fixtures.

    This reference validation intentionally loads the full small model. Enforce
    a 2 GiB storage ceiling so it cannot be mistaken for the bounded capacity
    path (`split`), which never materializes the whole checkpoint.
    """
    from .manifest import inspect_checkpoint, export_shards
    from .runtime import checked_args
    checkpoint = inspect_checkpoint(checkpoint, hash_sources=True)
    if checkpoint.total_weight_bytes > 2 * 1024**3:
        raise ValueError("Reference fixtures are limited to 2 GiB checkpoints; use split for capacity models")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(checkpoint.config)
    config["torch_dtype"] = "float16"
    checked_args(config)
    weights = {}
    for source in checkpoint.source_files:
        weights.update(mx.load(str(source.path)))
    weights = {name: value.astype(mx.float16) for name, value in weights.items()}
    mx.eval(weights)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    mx.save_safetensors(str(output / "full.safetensors"), weights, metadata={"format": "mlx-peer"})
    del weights
    manifest = export_shards(output / "full.safetensors", output / "partition", start_layer, end_layer)
    # Hardlinks are disk-only references; the runtime stage loads its own file.
    (output / "weights.safetensors").hardlink_to(output / "partition" / "iphone.safetensors")
    (output / "mac.safetensors").hardlink_to(output / "partition" / "mac.safetensors")
    reference = Model(ModelArgs.from_dict(config))
    reference.load_weights(list(mx.load(str(output / "full.safetensors")).items()))
    mx.eval(reference.parameters())
    stage = LayerStage(config, output / "weights.safetensors", start_layer, end_layer)
    cache = [KVCache() for _ in range(start_layer)]
    from mlx_lm.models.base import create_attention_mask
    position = 0
    steps = []
    for index, token_ids in enumerate(([1, 3, 7, 2, 11], [5], [6, 9, 8])):
        hidden = reference.model.embed_tokens(mx.array([token_ids], mx.int32))
        mask = create_attention_mask(hidden, cache[0] if cache else None)
        for layer, layer_cache in zip(reference.model.layers[:start_layer], cache):
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        mx.eval(hidden)
        expected = stage.forward(hidden, position)
        mx.save_safetensors(str(output / f"input-{index}.safetensors"), {"hidden_states": hidden}, metadata={"format": "mlx-peer"})
        mx.save_safetensors(str(output / f"expected-{index}.safetensors"), {"hidden_states": expected}, metadata={"format": "mlx-peer"})
        steps.append(dict(input=f"input-{index}.safetensors", output=f"output-{index}.safetensors", position=position))
        position += len(token_ids)
    request = dict(protocol_version=1, layer_start=start_layer, layer_end=end_layer,
                   session_id="trained-qwen-stage-probe", position=0, max_context=256, steps=steps)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    metadata = dict(fixture="trained-qwen-checkpoint", trained_model=True,
                    purpose="small-model reference validation only; not the capacity execution path",
                    dtype="float16", source_checkpoint=checkpoint.to_dict(),
                    conversion="Original floating weights cast to float16 with MLX; both references use converted weights",
                    python=platform.python_version(), mlx=mx.__version__,
                    full_weight_bytes=manifest["total_weight_bytes"],
                    mac_weight_bytes=manifest["shards"]["mac"]["weight_bytes"],
                    worker_weight_bytes=manifest["shards"]["iphone"]["weight_bytes"])
    (output / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {key: value for key, value in metadata.items() if key != "source_checkpoint"}
