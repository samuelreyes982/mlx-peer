"""Package only the files an offline iPhone stage fixture needs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import unicodedata
from pathlib import Path
from typing import Any

MAX_INPUT_COUNT = 64
MAX_WEIGHTS_BYTES = 1024 ** 3
MAX_ACTIVATION_BYTES = 64 * 1024 ** 2
MAX_METADATA_BYTES = 64 * 1024
COPY_CHUNK_BYTES = 8 * 1024 ** 2
RESERVED = frozenset({
    "config.json", "request.json", "weights.safetensors", "bundle.json",
    "full.safetensors", "mac.safetensors", "fixture.json", "report.json",
    "python-report.json", "parity-report.json", "expected.safetensors",
})


class BundleError(ValueError):
    """A fixture cannot be safely packed for the worker."""


def _alias(name: str) -> str:
    # The destination may be a case-insensitive, Unicode-normalizing filesystem.
    return unicodedata.normalize("NFC", name).casefold()


def _name(value: Any) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or "/" in value or "\\" in value or "\x00" in value
            or len(value.encode("utf-8")) > 255):
        raise BundleError("Fixture members must be single local filenames")
    return value


def _tensor_name(value: Any) -> str:
    name = _name(value)
    canonical = _alias(name)
    if (not name.endswith(".safetensors") or canonical in RESERVED
            or canonical.startswith("expected-")):
        raise BundleError(f"Reserved or invalid tensor filename: {name}")
    return name


def _member(root: Path, name: str, limit: int) -> Path:
    path = root / _name(name)
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise BundleError(f"Missing fixture file: {name}") from error
    if not stat.S_ISREG(info.st_mode) or path.resolve().parent != root:
        raise BundleError(f"Fixture file must be regular and cannot be a symlink: {name}")
    if not 0 < info.st_size <= limit:
        raise BundleError(f"File is empty or exceeds the {limit}-byte limit: {name}")
    return path


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _metadata(path: Path) -> dict[str, Any]:
    # The size was checked before this bounded read. Read one extra byte to
    # reject growth between the stat and open without a large allocation.
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise BundleError(f"Metadata exceeds size limit: {path.name}")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"Invalid JSON metadata: {path.name}") from error
    if not isinstance(value, dict):
        raise BundleError(f"Metadata must be a JSON object: {path.name}")
    return value


def _inputs(request: dict[str, Any]) -> list[str]:
    steps = request.get("steps")
    if steps is None:
        steps = [{"input": "input.safetensors", "output": "output.safetensors"}]
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_INPUT_COUNT:
        raise BundleError(f"Fixture requires 1 to {MAX_INPUT_COUNT} input steps")
    inputs: list[str] = []
    input_aliases: set[str] = set()
    outputs: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            raise BundleError("Each fixture step must be a JSON object")
        source = _tensor_name(step.get("input"))
        output = _tensor_name(step.get("output"))
        source_alias, output_alias = _alias(source), _alias(output)
        if source_alias in input_aliases or output_alias in outputs:
            raise BundleError("Duplicate or aliased fixture inputs/outputs")
        input_aliases.add(source_alias)
        outputs.add(output_alias)
        inputs.append(source)
    if input_aliases & outputs:
        raise BundleError("Fixture outputs cannot alias any input")
    return inputs


def _copy(source: Path, destination: Path, limit: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as reader:
        initial = os.fstat(reader.fileno())
        if not stat.S_ISREG(initial.st_mode) or not 0 < initial.st_size <= limit:
            raise BundleError(f"File changed or exceeds its size limit: {source.name}")
        with destination.open("xb") as writer:
            while block := reader.read(min(COPY_CHUNK_BYTES, limit - count + 1)):
                count += len(block)
                if count > limit:
                    raise BundleError(f"File grew beyond its size limit: {source.name}")
                writer.write(block)
                digest.update(block)
        final = os.fstat(reader.fileno())
        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
            raise BundleError(f"File changed while packing: {source.name}")
        if count != initial.st_size:
            raise BundleError(f"File length changed while packing: {source.name}")
    return {"file": destination.name, "bytes": count, "sha256": digest.hexdigest()}


def pack_fixture(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Create a new offline worker bundle without full/Mac weights or answers.

    Copies are bounded; inputs/outputs must have unique safe basenames. The
    bundle manifest identifies file contents only, not hardware or execution.
    Existing destinations are never modified. A failed copy removes only the
    new destination created by this call.
    """
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise BundleError(f"Fixture source is not a directory: {root}")
    config_path = _member(root, "config.json", MAX_METADATA_BYTES)
    request_path = _member(root, "request.json", MAX_METADATA_BYTES)
    _metadata(config_path)
    request = _metadata(request_path)
    input_names = _inputs(request)
    names_and_limits = [("config.json", MAX_METADATA_BYTES),
                        ("request.json", MAX_METADATA_BYTES),
                        ("weights.safetensors", MAX_WEIGHTS_BYTES)]
    names_and_limits.extend((name, MAX_ACTIVATION_BYTES) for name in input_names)
    members = [(_member(root, name, limit), limit) for name, limit in names_and_limits]
    target = Path(destination).expanduser().absolute()
    target.mkdir(parents=True, exist_ok=False)
    try:
        files = [_copy(path, target / path.name, limit) for path, limit in members]
        # Do not copy a request edited after validation: it could refer to files
        # we deliberately excluded. Revalidate the actual packaged metadata.
        if _metadata(target / "request.json") != request:
            raise BundleError("request.json changed while packing")
        result = {
            "schema": "mlx-peer-fixture-bundle", "schema_version": 1,
            "input_count": len(input_names), "files": files,
            "total_bytes": sum(item["bytes"] for item in files),
            "purpose": "Offline stage inputs only; packing does not establish device execution.",
        }
        (target / "bundle.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException:
        shutil.rmtree(target)
        raise
