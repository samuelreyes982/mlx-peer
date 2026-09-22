import json
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_peer.probe import make_fixture, verify_local, tiny_config
from mlx_peer.runtime import LayerStage, MacCoordinator, checked_args


@pytest.fixture
def fixture_dir(tmp_path):
    result = tmp_path / "fixture"
    make_fixture(result)
    return result


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_split_matches_complete_upstream_model(tmp_path, dtype):
    path = tmp_path / dtype
    make_fixture(path, dtype)
    report = verify_local(path)
    assert report["passed"]
    assert report["mac_layer_indices"] == [0, 3]
    assert report["iphone_executed"] is False
    metadata = json.loads((path / "fixture.json").read_text())
    assert report["mac_weight_bytes"] + report["worker_weight_bytes"] == metadata["full_weight_bytes"]


def test_stage_rejects_bad_offset_without_corrupting_cache(fixture_dir):
    stage = LayerStage(tiny_config(), fixture_dir / "weights.safetensors", 1, 3)
    x = mx.ones((1, 2, 32))
    expected = stage.forward(x, 0)
    with pytest.raises(ValueError, match="offset"):
        stage.forward(x, 0)
    assert stage.position == 2
    stage.reset()
    assert mx.array_equal(stage.forward(x, 0), expected).item()


def test_stage_rejects_shape_dtype_and_context(fixture_dir):
    stage = LayerStage(tiny_config(), fixture_dir / "weights.safetensors", 1, 3, max_context=8)
    for bad in (mx.ones((2, 1, 32)), mx.ones((1, 1, 31)), mx.ones((1, 1, 32), mx.float16), mx.ones((1, 9, 32))):
        with pytest.raises(ValueError):
            stage.forward(bad, 0)
    assert stage.position == 0


def test_stage_rejects_full_checkpoint(fixture_dir):
    with pytest.raises(ValueError, match="unassigned"):
        LayerStage(tiny_config(), fixture_dir / "full.safetensors", 1, 3)


def test_coordinator_requires_reset_after_worker_failure(fixture_dir):
    stage = LayerStage(tiny_config(), fixture_dir / "weights.safetensors", 1, 3)
    coordinator = MacCoordinator(tiny_config(), fixture_dir / "mac.safetensors", 1, 3, stage)
    stage.position = 9
    with pytest.raises(ValueError, match="offset"):
        coordinator.forward(mx.array([[1, 2]]))
    with pytest.raises(ValueError, match="reset"):
        coordinator.forward(mx.array([[1]]))
    coordinator.reset()
    assert coordinator.forward(mx.array([[1]])).shape == (1, 1, 128)


def test_unsupported_model_features_fail_explicitly():
    for extra in ({"model_type": "qwen3"}, {"quantization": {"bits": 4}}, {"rope_scaling": {"factor": 2}}, {"use_sliding_window": True}):
        with pytest.raises(ValueError):
            checked_args(tiny_config() | extra)
