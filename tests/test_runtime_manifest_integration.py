"""Exercise the streaming exporter and runtime together, without a real phone.

These tests use tiny randomly initialized weights and cannot establish capacity
or language quality. The independent upstream model is the numerical reference.
"""

import hashlib
import json
import shutil

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen2 import Model, ModelArgs

from mlx_peer.manifest import export_shards, inspect_checkpoint
from mlx_peer.probe import make_fixture
from mlx_peer.runtime import LayerStage, checked_args, load_partitioned


def checkpoint_from_fixture(tmp_path, dtype="float32", tied=True):
    fixture = tmp_path / "fixture"
    make_fixture(fixture, dtype=dtype)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    shutil.copyfile(fixture / "full.safetensors", checkpoint / "model.safetensors")
    config = json.loads((fixture / "config.json").read_text())
    if not tied:
        config["tie_word_embeddings"] = False
        weights = mx.load(str(checkpoint / "model.safetensors"))
        weights["lm_head.weight"] = mx.random.normal(
            (config["vocab_size"], config["hidden_size"])
        ).astype(weights["model.embed_tokens.weight"].dtype)
        mx.eval(weights)
        mx.save_safetensors(str(checkpoint / "model.safetensors"), weights, metadata={"format": "mlx-peer"})
    (checkpoint / "config.json").write_text(json.dumps(config))
    return checkpoint, config


def exported_stage(tmp_path):
    checkpoint, config = checkpoint_from_fixture(tmp_path)
    exported = tmp_path / "export"
    export_shards(checkpoint, exported, 1, 3)
    stage = LayerStage(config, exported / "iphone.safetensors", 1, 3)
    return exported, config, stage


def rewrite_manifest(directory, update):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    update(manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("start,end", [(1, 3), (0, 1), (3, 4), (0, 4)])
def test_exported_split_matches_full_upstream_model(tmp_path, dtype, tied, start, end):
    checkpoint, config = checkpoint_from_fixture(tmp_path, dtype, tied)
    inspected = inspect_checkpoint(checkpoint)
    output = tmp_path / "export"
    manifest = export_shards(inspected, output, start, end, chunk_size=1024)
    stage = LayerStage(config, output / "iphone.safetensors", start, end)
    coordinator = load_partitioned(output, stage, max_context=32)
    reference = Model(ModelArgs.from_dict(config))
    reference.load_weights(list(mx.load(str(checkpoint / "model.safetensors")).items()))
    mx.eval(reference.parameters())
    caches = [KVCache() for _ in range(config["num_hidden_layers"])]
    tolerances = dict(rtol=0.002, atol=0.002) if dtype == "float16" else dict(rtol=1e-4, atol=1e-5)
    for ids in ([1, 2, 3, 4], [8], [9, 10, 11]):
        tokens = mx.array([ids], mx.int32)
        expected = reference(tokens, cache=caches)
        actual = coordinator.forward(tokens)
        mx.eval(expected, actual)
        assert mx.allclose(expected, actual, **tolerances).item()
    assert coordinator.position == stage.position == 8
    assert sorted(coordinator.layers) == [index for index in range(4) if not start <= index < end]
    assert coordinator.weight_bytes == manifest["shards"]["mac"]["weight_bytes"]
    assert stage.weight_bytes == manifest["shards"]["iphone"]["weight_bytes"]
    assert coordinator.weight_bytes + stage.weight_bytes == inspected.total_weight_bytes
    assert (coordinator.head is None) == tied


@pytest.mark.parametrize("name", ["config.json", "mac.safetensors"])
def test_modified_local_files_fail_checksum(tmp_path, name):
    output, _, stage = exported_stage(tmp_path)
    target = output / name
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 1
    target.write_bytes(raw)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_partitioned(output, stage)


@pytest.mark.parametrize("field,value", [("schema", "not-peer"), ("schema_version", 2)])
def test_wrong_manifest_schema_rejected(tmp_path, field, value):
    output, _, stage = exported_stage(tmp_path)
    rewrite_manifest(output, lambda manifest: manifest.update({field: value}))
    with pytest.raises(ValueError, match="manifest"):
        load_partitioned(output, stage)


def test_worker_range_mismatch_rejected(tmp_path):
    output, _, stage = exported_stage(tmp_path)
    rewrite_manifest(output, lambda manifest: manifest["remote_layer_range"].update(start=0))
    with pytest.raises(ValueError, match="layer range"):
        load_partitioned(output, stage)


def test_worker_configuration_mismatch_rejected_even_with_updated_config_hash(tmp_path):
    output, config, stage = exported_stage(tmp_path)
    config["rope_theta"] = 10_000.0
    raw = json.dumps(config).encode()
    (output / "config.json").write_bytes(raw)
    rewrite_manifest(output, lambda manifest: manifest.update(config_sha256=hashlib.sha256(raw).hexdigest()))
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_partitioned(output, stage)


def test_changed_remote_hash_rejected(tmp_path):
    output, _, stage = exported_stage(tmp_path)
    rewrite_manifest(output, lambda manifest: manifest["shards"]["iphone"].update(sha256="0" * 64))
    with pytest.raises(ValueError, match="(?i)(checksum|hash|identity)"):
        load_partitioned(output, stage)


def test_different_phone_weights_with_same_config_range_rejected(tmp_path):
    output, config, _ = exported_stage(tmp_path)
    phone_path = output / "iphone.safetensors"
    weights = mx.load(str(phone_path))
    name = next(iter(weights))
    weights[name] = weights[name] + mx.array(0.25, weights[name].dtype)
    alternate = tmp_path / "different-phone.safetensors"
    mx.save_safetensors(str(alternate), weights, metadata={"format": "mlx-peer"})
    stage = LayerStage(config, alternate, 1, 3)
    with pytest.raises(ValueError, match="(?i)(checksum|hash|identity)"):
        load_partitioned(output, stage)


def test_wrong_manifest_model_label_rejected(tmp_path):
    output, _, stage = exported_stage(tmp_path)
    rewrite_manifest(output, lambda manifest: manifest.update(model_type="qwen3"))
    with pytest.raises(ValueError, match="(?i)(model|manifest)"):
        load_partitioned(output, stage)


def test_wrong_manifest_layer_count_rejected(tmp_path):
    output, _, stage = exported_stage(tmp_path)
    rewrite_manifest(output, lambda manifest: manifest.update(num_hidden_layers=8))
    with pytest.raises(ValueError, match="(?i)(layer|manifest|config)"):
        load_partitioned(output, stage)


def test_tied_checkpoint_with_extra_projection_rejected(tmp_path):
    checkpoint, config = checkpoint_from_fixture(tmp_path)
    weights = mx.load(str(checkpoint / "model.safetensors"))
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    mx.eval(weights)
    mx.save_safetensors(str(checkpoint / "model.safetensors"), weights, metadata={"format": "mlx-peer"})
    output = tmp_path / "export"
    export_shards(checkpoint, output, 1, 3)
    stage = LayerStage(config, output / "iphone.safetensors", 1, 3)
    with pytest.raises(ValueError, match="unassigned"):
        load_partitioned(output, stage)


def test_omitted_tying_flag_uses_qwen2_untied_default(tmp_path):
    checkpoint, config = checkpoint_from_fixture(tmp_path, tied=False)
    del config["tie_word_embeddings"]
    (checkpoint / "config.json").write_text(json.dumps(config))
    output = tmp_path / "export"
    export_shards(checkpoint, output, 1, 3)
    stage = LayerStage(config, output / "iphone.safetensors", 1, 3)
    coordinator = load_partitioned(output, stage)
    assert coordinator.head is not None


def test_explicit_nonstandard_head_dimension_rejected(tmp_path):
    _, config = checkpoint_from_fixture(tmp_path)
    config["head_dim"] = 4  # The pinned Qwen2 implementation uses 32 / 4 = 8.
    with pytest.raises(ValueError, match="(?i)head"):
        checked_args(config)
