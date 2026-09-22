"""Inspect and partition dense Qwen2 safetensors without loading tensor data.

This module has no MLX dependency. Exporting weights establishes placement on
disk only; executing each shard requires the separate runtime adapters.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_HEADER_BYTES = 100_000_000
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
    "U32": 4, "I32": 4, "U64": 8, "I64": 8,
    "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1,
}
LAYER_NAME = re.compile(r"^model\.layers\.(0|[1-9][0-9]*)\.(.+)$")
REQUIRED_LAYER_WEIGHTS = frozenset({
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
})


class ManifestError(ValueError):
    """A checkpoint cannot be safely interpreted or partitioned."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ManifestError(f"Duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    def invalid_constant(value: str) -> None:
        raise ManifestError(f"Non-finite JSON number in {label}: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ManifestError(f"Non-finite JSON number in {label}: {value}")
        return parsed

    try:
        value = json.loads(raw, object_pairs_hook=_unique_object,
                           parse_constant=invalid_constant, parse_float=finite_float)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"Invalid JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must contain a JSON object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ManifestError(f"{label} must be a positive integer")
    return value


def _sha256(path: Path, chunk_size: int = DEFAULT_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(root: Path, name: str) -> Path:
    # Checkpoint indices normally reference sibling files. Restricting to a
    # basename also rejects Windows separators when running on macOS/Linux.
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ManifestError("Checkpoint file name must be a nonempty basename")
    if "/" in name or "\\" in name or "\x00" in name:
        raise ManifestError(f"Unsafe checkpoint file name: {name!r}")
    path = root / name
    resolved = path.resolve()
    if resolved.parent != root.resolve():
        raise ManifestError(f"Checkpoint file escapes its directory: {name!r}")
    if not resolved.is_file():
        raise ManifestError(f"Missing checkpoint file: {name}")
    return path


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_file: Path
    data_start: int
    data_end: int
    header_bytes: int

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start

    @property
    def absolute_offset(self) -> int:
        return 8 + self.header_bytes + self.data_start

    @property
    def layer(self) -> int | None:
        match = LAYER_NAME.match(self.name)
        return int(match.group(1)) if match else None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype,
                "shape": list(self.shape), "weight_bytes": self.nbytes,
                "source_file": self.source_file.name,
                "data_offsets": [self.data_start, self.data_end]}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    size_bytes: int
    mtime_ns: int
    header_sha256: str
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"file": self.path.name, "size_bytes": self.size_bytes,
                  "header_sha256": self.header_sha256}
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        return result


@dataclass(frozen=True)
class Checkpoint:
    root: Path
    config: dict[str, Any]
    config_bytes: bytes
    config_sha256: str
    tensors: dict[str, TensorInfo]
    source_files: tuple[SourceFile, ...]

    @property
    def layer_count(self) -> int:
        return self.config["num_hidden_layers"]

    @property
    def total_weight_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors.values())

    def to_dict(self) -> dict[str, Any]:
        per_layer = {str(index): 0 for index in range(self.layer_count)}
        non_layer = 0
        for tensor in self.tensors.values():
            if tensor.layer is None:
                non_layer += tensor.nbytes
            else:
                per_layer[str(tensor.layer)] += tensor.nbytes
        return {
            "model_type": self.config["model_type"],
            "layer_count": self.layer_count,
            "tensor_count": len(self.tensors),
            "total_weight_bytes": self.total_weight_bytes,
            "layer_weight_bytes": per_layer,
            "non_layer_weight_bytes": non_layer,
            "config_sha256": self.config_sha256,
            "source_files": [source.to_dict() for source in self.source_files],
            "note": "Weight sizes describe storage, not measured runtime memory.",
        }


def _inspect_file(path: Path, hash_sources: bool) -> tuple[dict[str, TensorInfo], SourceFile]:
    before = path.stat()
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ManifestError(f"Truncated safetensors length prefix: {path.name}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > MAX_HEADER_BYTES:
            raise ManifestError(f"Invalid safetensors header size: {header_size}")
        if 8 + header_size > before.st_size:
            raise ManifestError(f"Truncated safetensors header: {path.name}")
        raw_header = stream.read(header_size)
    if not raw_header.startswith(b"{"):
        raise ManifestError("Safetensors header must begin with '{'")
    header = _parse_json(raw_header, path.name)
    payload_size = before.st_size - 8 - header_size
    tensors: dict[str, TensorInfo] = {}
    for name, descriptor in header.items():
        if name == "__metadata__":
            if (not isinstance(descriptor, dict)
                    or not all(isinstance(key, str) and isinstance(value, str)
                               for key, value in descriptor.items())):
                raise ManifestError("Safetensors metadata must map strings to strings")
            continue
        if not name or not isinstance(descriptor, dict):
            raise ManifestError(f"Invalid tensor descriptor: {name!r}")
        if set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ManifestError(f"Unexpected fields for tensor {name!r}")
        dtype, shape, offsets = (descriptor["dtype"], descriptor["shape"],
                                 descriptor["data_offsets"])
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise ManifestError(f"Unsupported dtype for {name!r}: {dtype!r}")
        if (not isinstance(shape, list) or len(shape) > 32
                or any(type(dim) is not int or dim < 0 for dim in shape)):
            raise ManifestError(f"Invalid tensor shape for {name!r}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int or offset < 0 for offset in offsets)):
            raise ManifestError(f"Invalid data offsets for {name!r}")
        start, end = offsets
        expected = math.prod(shape) * DTYPE_BYTES[dtype]
        if end < start or end > payload_size or end - start != expected:
            raise ManifestError(f"Tensor byte count/offset mismatch for {name!r}")
        tensors[name] = TensorInfo(name, dtype, tuple(shape), path, start, end, header_size)
    cursor = 0
    for tensor in sorted(tensors.values(), key=lambda item: (item.data_start, item.data_end)):
        if tensor.data_start != cursor:
            raise ManifestError(f"Overlapping tensors or payload gap in {path.name}")
        cursor = tensor.data_end
    if cursor != payload_size:
        raise ManifestError(f"Unreferenced payload bytes in {path.name}")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ManifestError(f"Checkpoint changed while inspecting {path.name}")
    source = SourceFile(path, before.st_size, before.st_mtime_ns,
                        hashlib.sha256(raw_header).hexdigest(),
                        _sha256(path) if hash_sources else None)
    return tensors, source


def _validate_qwen(config: dict[str, Any], tensors: dict[str, TensorInfo]) -> None:
    if config.get("model_type") != "qwen2":
        raise ManifestError("This partitioner supports dense model_type='qwen2' only")
    count = _positive_int(config.get("num_hidden_layers"), "num_hidden_layers")
    if count > 4096:
        raise ManifestError("num_hidden_layers exceeds the supported inspection limit (4096)")
    for key in ("attention_bias", "tie_word_embeddings"):
        if key in config and type(config[key]) is not bool:
            raise ManifestError(f"{key} must be a boolean")
    hidden = _positive_int(config.get("hidden_size"), "hidden_size")
    heads = _positive_int(config.get("num_attention_heads"), "num_attention_heads")
    kv_heads = _positive_int(config.get("num_key_value_heads", heads), "num_key_value_heads")
    _positive_int(config.get("intermediate_size"), "intermediate_size")
    if hidden % heads or heads % kv_heads:
        raise ManifestError("Hidden size/attention head counts are inconsistent")
    found: dict[int, set[str]] = {index: set() for index in range(count)}
    for name in tensors:
        if name.startswith("model.layers."):
            match = LAYER_NAME.match(name)
            if not match or int(match.group(1)) >= count:
                raise ManifestError(f"Invalid or out-of-range layer tensor: {name}")
            found[int(match.group(1))].add(match.group(2))
    for index, suffixes in found.items():
        required_layer = set(REQUIRED_LAYER_WEIGHTS)
        if config.get("attention_bias", True):
            required_layer.update({"self_attn.q_proj.bias", "self_attn.k_proj.bias",
                                   "self_attn.v_proj.bias"})
        missing = required_layer - suffixes
        if missing:
            raise ManifestError(f"Incomplete layer {index}; missing: {', '.join(sorted(missing))}")
    required = {"model.embed_tokens.weight", "model.norm.weight"}
    if not config.get("tie_word_embeddings", False):
        required.add("lm_head.weight")
    if missing := required - tensors.keys():
        raise ManifestError(f"Missing model weights: {', '.join(sorted(missing))}")


def inspect_checkpoint(path: str | Path, *, hash_sources: bool = False) -> Checkpoint:
    """Read bounded safetensors headers and validate a complete local checkpoint.

    ``path`` is a directory containing config.json and either model.safetensors
    or model.safetensors.index.json, or a specific single .safetensors file.
    Full source hashing is optional and streams in bounded chunks.
    """
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() and not candidate.is_dir():
        raise ManifestError(f"Checkpoint path does not exist: {candidate}")
    root = candidate if candidate.is_dir() else candidate.parent
    config_path = _safe_file(root, "config.json")
    if config_path.stat().st_size > MAX_HEADER_BYTES:
        raise ManifestError("config.json is too large")
    raw_config = config_path.read_bytes()
    config = _parse_json(raw_config, "config.json")
    index_path = root / "model.safetensors.index.json"
    weight_map: dict[str, str] | None = None
    if candidate.is_file():
        if candidate.suffix != ".safetensors":
            raise ManifestError("An explicit checkpoint file must end in .safetensors")
        files = [_safe_file(root, candidate.name)]
    elif index_path.exists():
        if (root / "model.safetensors").exists():
            raise ManifestError("Both a single checkpoint and an index exist; choose an explicit file")
        index_path = _safe_file(root, index_path.name)
        if index_path.stat().st_size > MAX_HEADER_BYTES:
            raise ManifestError("Safetensors index is too large")
        index = _parse_json(index_path.read_bytes(), index_path.name)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ManifestError("Safetensors index needs a nonempty weight_map")
        if not all(isinstance(name, str) and name and isinstance(file, str)
                   and file.endswith(".safetensors") for name, file in weight_map.items()):
            raise ManifestError("Invalid safetensors weight_map")
        files = [_safe_file(root, name) for name in sorted(set(weight_map.values()))]
    else:
        files = [_safe_file(root, "model.safetensors")]
    tensors: dict[str, TensorInfo] = {}
    sources: list[SourceFile] = []
    for file in files:
        entries, source = _inspect_file(file, hash_sources)
        if overlap := tensors.keys() & entries.keys():
            raise ManifestError(f"Duplicate tensors across checkpoint files: {sorted(overlap)}")
        if weight_map is not None:
            for name in entries:
                if weight_map.get(name) != file.name:
                    raise ManifestError(f"Index/file assignment mismatch for {name}")
        tensors.update(entries)
        sources.append(source)
    if weight_map is not None and set(weight_map) != set(tensors):
        raise ManifestError("Index lists tensors missing from the checkpoint files")
    _validate_qwen(config, tensors)
    return Checkpoint(root, config, raw_config, hashlib.sha256(raw_config).hexdigest(),
                      tensors, tuple(sources))


def partition_tensors(checkpoint: Checkpoint, start_layer: int,
                      end_layer: int) -> dict[str, list[TensorInfo]]:
    """Assign [start_layer, end_layer) to iPhone; all other tensors to Mac."""
    if (type(start_layer) is not int or type(end_layer) is not int
            or not 0 <= start_layer < end_layer <= checkpoint.layer_count):
        raise ManifestError(f"Require 0 <= start_layer < end_layer <= {checkpoint.layer_count}")
    result: dict[str, list[TensorInfo]] = {"mac": [], "iphone": []}
    for name in sorted(checkpoint.tensors):
        tensor = checkpoint.tensors[name]
        device = ("iphone" if tensor.layer is not None
                  and start_layer <= tensor.layer < end_layer else "mac")
        result[device].append(tensor)
    return result


def _check_unchanged(checkpoint: Checkpoint) -> None:
    for source in checkpoint.source_files:
        status = source.path.stat()
        if (status.st_size, status.st_mtime_ns) != (source.size_bytes, source.mtime_ns):
            raise ManifestError(f"Source checkpoint changed: {source.path.name}; inspect again")
    if (checkpoint.root / "config.json").read_bytes() != checkpoint.config_bytes:
        raise ManifestError("config.json changed; inspect again")


def _write_shard(path: Path, tensors: list[TensorInfo], chunk_size: int) -> dict[str, Any]:
    header: dict[str, Any] = {"__metadata__": {"format": "mlx_peer", "version": "1"}}
    cursor = 0
    for tensor in tensors:
        header[tensor.name] = {"dtype": tensor.dtype, "shape": list(tensor.shape),
                               "data_offsets": [cursor, cursor + tensor.nbytes]}
        cursor += tensor.nbytes
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ManifestError("Exported shard header is too large")
    digest = hashlib.sha256()
    with path.open("xb") as destination:
        prefix = struct.pack("<Q", len(header_bytes)) + header_bytes
        destination.write(prefix)
        digest.update(prefix)
        for tensor in tensors:
            with tensor.source_file.open("rb") as source:
                source.seek(tensor.absolute_offset)
                remaining = tensor.nbytes
                while remaining:
                    block = source.read(min(remaining, chunk_size))
                    if not block:
                        raise ManifestError(f"Source truncated while copying {tensor.name}")
                    destination.write(block)
                    digest.update(block)
                    remaining -= len(block)
        destination.flush()
        os.fsync(destination.fileno())
    return {"file": path.name, "sha256": digest.hexdigest(),
            "file_bytes": path.stat().st_size, "weight_bytes": cursor,
            "tensor_count": len(tensors), "tensor_names": [item.name for item in tensors]}


def export_shards(checkpoint_or_path: Checkpoint | str | Path,
                  output_dir: str | Path, start_layer: int, end_layer: int, *,
                  chunk_size: int = DEFAULT_CHUNK_BYTES) -> dict[str, Any]:
    """Create two valid safetensors files and a manifest in a NEW directory.

    Tensor payloads are copied with bounded reads. Full source hashes are also
    streamed; no complete model or tensor is materialized. A failed export
    removes only the new directory it created, never an existing output.
    """
    if type(chunk_size) is not int or not 1 <= chunk_size <= 64 * 1024 * 1024:
        raise ManifestError("chunk_size must be between 1 byte and 64 MiB")
    checkpoint = (checkpoint_or_path if isinstance(checkpoint_or_path, Checkpoint)
                  else inspect_checkpoint(checkpoint_or_path))
    assignments = partition_tensors(checkpoint, start_layer, end_layer)
    _check_unchanged(checkpoint)
    destination = Path(output_dir).expanduser().absolute()
    # mkdir(exist_ok=False) prevents any overwrite, including a symlink target.
    destination.mkdir(parents=True, exist_ok=False)
    try:
        sources = []
        for source in checkpoint.source_files:
            identity = source.to_dict()
            identity["sha256"] = _sha256(source.path, chunk_size)
            if source.sha256 is not None and source.sha256 != identity["sha256"]:
                raise ManifestError(f"Source hash changed: {source.path.name}")
            sources.append(identity)
        shards = {device: _write_shard(destination / f"{device}.safetensors", tensors,
                                       chunk_size)
                  for device, tensors in assignments.items()}
        _check_unchanged(checkpoint)
        (destination / "config.json").write_bytes(checkpoint.config_bytes)
        manifest = {
            "schema": "mlx-peer-shards", "schema_version": 1,
            "model_type": "qwen2", "config_file": "config.json",
            "config_sha256": checkpoint.config_sha256,
            "num_hidden_layers": checkpoint.layer_count,
            "remote_layer_range": {"start": start_layer, "end": end_layer,
                                   "end_exclusive": True},
            "source_files": sources, "shards": shards,
            "total_weight_bytes": checkpoint.total_weight_bytes,
            "scope": "Weight partition only; runtime memory and execution are unverified.",
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    except BaseException:
        shutil.rmtree(destination)
        raise
