"""Prepare supported local checkpoints without loading an entire model into RAM."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import uuid

from .manifest import inspect_checkpoint, partition_tensors, _check_unchanged


def inspect_model(path):
    root = Path(path).expanduser().resolve()
    checkpoint = inspect_checkpoint(root)
    config = checkpoint.config
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("This preview supports unquantized Qwen2 / Qwen2.5 safetensors. Quantized and GGUF support is not included yet.")
    if config.get("rope_scaling") or config.get("use_sliding_window") or config.get("hidden_act", "silu") != "silu" or config.get("attention_bias", True) is not True:
        raise ValueError("This Qwen attention configuration is not supported")
    if not 2 <= checkpoint.layer_count <= 128:
        raise ValueError("Unsupported number of model layers")
    if {t.dtype for t in checkpoint.tensors.values()} - {"F16", "BF16", "F32"}:
        raise ValueError("Use a model with FP16, BF16, or FP32 weights")
    if not (checkpoint.root / "tokenizer.json").is_file():
        raise ValueError("Select the complete model folder, including tokenizer.json and config.json")
    return checkpoint


def fp16_size(tensors):
    return sum(t.nbytes // {"F16": 2, "BF16": 2, "F32": 4}[t.dtype] * 2 for t in tensors)


def choose_layers(checkpoint, available_phone_bytes, requested=None):
    # Conservative preview policy: at most four layers / 512 MiB, plus runtime/cache reserve.
    budget = min(512 * 1024**2, max(0, int(available_phone_bytes) - 768 * 1024**2))
    end = min(4, max(1, checkpoint.layer_count // 4)) if requested is None else int(requested)
    if not 1 <= end < checkpoint.layer_count:
        raise ValueError("Phone layers must be between 1 and the model's layer count minus one")
    while end > 0:
        parts = partition_tensors(checkpoint, 0, end)
        if fp16_size(parts["iphone"]) <= budget:
            return end
        if requested is not None:
            break
        end -= 1
    raise ValueError("Not enough available iPhone memory for this model. Try a smaller model.")


def write_fp16(path, tensors, cancelled=lambda: False):
    import numpy as np
    header = {"__metadata__": {"format": "mlx"}}
    cursor = 0
    for tensor in tensors:
        size = fp16_size([tensor])
        header[tensor.name] = {"dtype": "F16", "shape": tensor.shape, "data_offsets": [cursor, cursor + size]}
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with Path(path).open("xb") as out:
        out.write(struct.pack("<Q", len(encoded)) + encoded)
        for tensor in tensors:
            with tensor.source_file.open("rb") as source:
                source.seek(tensor.absolute_offset)
                remaining = tensor.nbytes
                while remaining:
                    if cancelled(): raise InterruptedError("Model preparation cancelled")
                    block = source.read(min(remaining, 4 * 1024**2))
                    if not block: raise ValueError("Model changed or was truncated during preparation")
                    remaining -= len(block)
                    if tensor.dtype == "BF16":
                        block = (np.frombuffer(block, dtype="<u2").astype("<u4") << 16).view("<f4").astype("<f2").tobytes()
                    elif tensor.dtype == "F32":
                        block = np.frombuffer(block, dtype="<f4").astype("<f2").tobytes()
                    if not np.isfinite(np.frombuffer(block, dtype="<f2")).all():
                        raise ValueError("This model contains weights that cannot be represented as finite FP16 values")
                    out.write(block)
        out.flush(); os.fsync(out.fileno())


def prepare(checkpoint, cache, end, context=512, cancelled=lambda: False):
    if not 64 <= context <= 1024:
        raise ValueError("Context must be between 64 and 1024 tokens")
    _check_unchanged(checkpoint)
    # Content hashes keep a changed checkpoint from silently reusing an old prepared model.
    identity = hashlib.sha256(checkpoint.config_bytes)
    for file in checkpoint.source_files:
        if cancelled(): raise InterruptedError("Model preparation cancelled")
        with file.path.open("rb") as stream:
            while chunk := stream.read(4 * 1024**2):
                if cancelled(): raise InterruptedError("Model preparation cancelled")
                identity.update(chunk)
    identity.update(f"fp16-v1:0:{end}:{context}".encode())
    root = Path(cache); root.mkdir(parents=True, exist_ok=True)
    destination = root / identity.hexdigest()
    if (destination / "complete.json").is_file():
        return destination
    total = fp16_size(checkpoint.tensors.values())
    if shutil.disk_usage(root).free < total + 512 * 1024**2:
        raise ValueError("Not enough free Mac storage to prepare this model")
    temporary = root / (".preparing-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        parts = partition_tensors(checkpoint, 0, end)
        # Tied heads can be reconstructed by the coordinator from the embedding.
        if checkpoint.config.get("tie_word_embeddings"):
            parts["mac"] = [t for t in parts["mac"] if t.name != "lm_head.weight"]
        write_fp16(temporary / "weights.safetensors", parts["iphone"], cancelled)
        write_fp16(temporary / "mac.safetensors", parts["mac"], cancelled)
        (temporary / "config.json").write_bytes(checkpoint.config_bytes)
        request = {"protocol_version": 1, "layer_start": 0, "layer_end": end,
                   "session_id": identity.hexdigest(), "position": 0, "max_context": context}
        (temporary / "request.json").write_text(json.dumps(request))
        _check_unchanged(checkpoint)
        (temporary / "complete.json").write_text(json.dumps({"source": str(checkpoint.root), "context": context, "phone_layers": end}))
        temporary.rename(destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary)
        raise
