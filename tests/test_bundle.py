import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_peer.bundle import BundleError, pack_fixture


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "fixture"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen2"}')
    (root / "request.json").write_text(json.dumps({
        "steps": [{"input": "input-0.safetensors", "output": "output-0.safetensors", "position": 0},
                  {"input": "input-1.safetensors", "output": "output-1.safetensors", "position": 2}]
    }))
    for name in ("weights.safetensors", "input-0.safetensors", "input-1.safetensors",
                 "full.safetensors", "mac.safetensors", "expected-0.safetensors",
                 "output-0.safetensors", "report.json", "python-report.json", "fixture.json"):
        (root / name).write_bytes(name.encode() * 16)
    return root


def write_steps(root, steps):
    (root / "request.json").write_text(json.dumps({"steps": steps}))


def test_bundle_copies_only_worker_inputs_and_records_hashes(source, tmp_path):
    destination = tmp_path / "bundle"
    result = pack_fixture(source, destination)
    expected = {"config.json", "request.json", "weights.safetensors",
                "input-0.safetensors", "input-1.safetensors", "bundle.json"}
    assert {path.name for path in destination.iterdir()} == expected
    assert result["input_count"] == 2
    assert result["total_bytes"] == sum((source / row["file"]).stat().st_size for row in result["files"])
    for row in result["files"]:
        raw = (destination / row["file"]).read_bytes()
        assert raw == (source / row["file"]).read_bytes()
        assert row["bytes"] == len(raw)
        assert row["sha256"] == hashlib.sha256(raw).hexdigest()
    assert json.loads((destination / "bundle.json").read_text()) == result


@pytest.mark.parametrize("request_data", [{}, {"steps": None}])
def test_default_single_input(source, tmp_path, request_data):
    (source / "request.json").write_text(json.dumps(request_data))
    (source / "input.safetensors").write_bytes(b"input")
    result = pack_fixture(source, tmp_path / "bundle")
    assert result["input_count"] == 1
    assert {row["file"] for row in result["files"]} == {
        "config.json", "request.json", "weights.safetensors", "input.safetensors"}


@pytest.mark.parametrize("filename", ["../outside.safetensors", "dir/input.safetensors",
                                      "..\\outside.safetensors", "/tmp/input.safetensors",
                                      "input\x00.safetensors", "full.safetensors",
                                      "mac.safetensors", "weights.safetensors",
                                      "expected-0.safetensors", "MAC.safetensors"])
def test_unsafe_or_reserved_inputs_rejected(source, tmp_path, filename):
    write_steps(source, [{"input": filename, "output": "output.safetensors"}])
    with pytest.raises(BundleError):
        pack_fixture(source, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("steps", [[], [{}], ["invalid"],
    [{"input": "input-0.safetensors", "output": "input-0.safetensors"}],
    [{"input": "input-0.safetensors", "output": "INPUT-0.safetensors"}],
    [{"input": "input-0.safetensors", "output": "input-1.safetensors"},
     {"input": "input-1.safetensors", "output": "other.safetensors"}],
    [{"input": "input-0.safetensors", "output": "out.safetensors"},
     {"input": "input-1.safetensors", "output": "OUT.safetensors"}],
    [{"input": "input-0.safetensors", "output": "../out.safetensors"}],
    [{"input": "input-0.safetensors", "output": "weights.safetensors"}],
])
def test_invalid_steps_and_aliases_rejected(source, tmp_path, steps):
    write_steps(source, steps)
    with pytest.raises(BundleError):
        pack_fixture(source, tmp_path / "bundle")


def test_input_count_limit(source, tmp_path):
    write_steps(source, [{"input": f"in-{i}.safetensors", "output": f"out-{i}.safetensors"}
                         for i in range(65)])
    with pytest.raises(BundleError, match="64"):
        pack_fixture(source, tmp_path / "bundle")


def test_missing_file_rejected_before_destination_created(source, tmp_path):
    (source / "input-1.safetensors").unlink()
    with pytest.raises(BundleError, match="Missing"):
        pack_fixture(source, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_source_symlink_rejected(source, tmp_path):
    path = source / "input-1.safetensors"
    path.unlink()
    external = tmp_path / "external.safetensors"
    external.write_bytes(b"external")
    path.symlink_to(external)
    with pytest.raises(BundleError, match="symlink"):
        pack_fixture(source, tmp_path / "bundle")


def test_existing_destination_untouched(source, tmp_path):
    target = tmp_path / "bundle"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        pack_fixture(source, target)
    assert (target / "keep.txt").read_text() == "keep"


def test_failed_copy_cleans_new_destination_only(source, tmp_path):
    target = tmp_path / "bundle"
    with patch("mlx_peer.bundle._copy", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            pack_fixture(source, target)
    assert not target.exists()
    assert (source / "weights.safetensors").is_file()


@pytest.mark.parametrize("filename,limit_name", [
    ("weights.safetensors", "MAX_WEIGHTS_BYTES"),
    ("input-0.safetensors", "MAX_ACTIVATION_BYTES"),
    ("config.json", "MAX_METADATA_BYTES"),
])
def test_oversized_file_rejected(source, tmp_path, filename, limit_name):
    with patch(f"mlx_peer.bundle.{limit_name}", 8):
        with pytest.raises(BundleError, match="limit"):
            pack_fixture(source, tmp_path / "bundle")


def test_duplicate_json_keys_rejected(source, tmp_path):
    (source / "request.json").write_text('{"steps":null,"steps":[]}')
    with pytest.raises(BundleError, match="Duplicate"):
        pack_fixture(source, tmp_path / "bundle")
