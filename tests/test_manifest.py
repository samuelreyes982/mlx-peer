import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_peer.manifest import (
    ManifestError, REQUIRED_LAYER_WEIGHTS, export_shards, inspect_checkpoint,
    partition_tensors,
)


def write_tensor_file(path, tensors):
    header = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, (shape, raw) in tensors.items():
        header[name] = {"dtype": "F16", "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def read_payloads(path):
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_len])
    data = raw[8 + header_len:]
    return {name: (item["shape"], data[item["data_offsets"][0]:item["data_offsets"][1]])
            for name, item in header.items() if name != "__metadata__"}


def fixture(root):
    config = {"model_type": "qwen2", "num_hidden_layers": 3,
              "hidden_size": 4, "intermediate_size": 8,
              "num_attention_heads": 2, "num_key_value_heads": 1,
              "vocab_size": 5, "tie_word_embeddings": False}
    (root / "config.json").write_text(json.dumps(config))
    shapes = {"model.embed_tokens.weight": [5, 4], "model.norm.weight": [4],
              "lm_head.weight": [5, 4]}
    for layer in range(3):
        prefix = f"model.layers.{layer}."
        for suffix in REQUIRED_LAYER_WEIGHTS:
            if "layernorm" in suffix:
                shape = [4]
            elif "gate_proj" in suffix or "up_proj" in suffix:
                shape = [8, 4]
            elif "down_proj" in suffix:
                shape = [4, 8]
            elif "k_proj" in suffix or "v_proj" in suffix:
                shape = [2, 4]
            else:
                shape = [4, 4]
            shapes[prefix + suffix] = shape
        for projection, size in (("q", 4), ("k", 2), ("v", 2)):
            shapes[prefix + f"self_attn.{projection}_proj.bias"] = [size]
    tensors = {name: (shape, bytes([(index + 1) % 256]) * (math.prod(shape) * 2))
               for index, (name, shape) in enumerate(sorted(shapes.items()))}
    write_tensor_file(root / "model.safetensors", tensors)
    return config, tensors


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "checkpoint"
        self.root.mkdir()
        self.config, self.original = fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_inspect_and_partition_reassembles_identical_tensor_bytes(self):
        checkpoint = inspect_checkpoint(self.root, hash_sources=True)
        self.assertEqual(checkpoint.layer_count, 3)
        self.assertEqual(checkpoint.total_weight_bytes,
                         sum(len(raw) for _, raw in self.original.values()))
        output = Path(self.tmp.name) / "export"
        manifest = export_shards(checkpoint, output, 1, 2, chunk_size=7)
        mac = read_payloads(output / "mac.safetensors")
        phone = read_payloads(output / "iphone.safetensors")
        self.assertFalse(mac.keys() & phone.keys())
        self.assertEqual(mac | phone, self.original)
        self.assertTrue(all(name.startswith("model.layers.1.") for name in phone))
        self.assertEqual(len(phone), 12)
        self.assertIn("model.embed_tokens.weight", mac)
        self.assertIn("lm_head.weight", mac)
        self.assertEqual(manifest["remote_layer_range"]["end"], 2)
        self.assertEqual(manifest["total_weight_bytes"],
                         sum(shard["weight_bytes"] for shard in manifest["shards"].values()))
        for name in ("mac", "iphone"):
            self.assertEqual(manifest["shards"][name]["sha256"],
                             hashlib.sha256((output / f"{name}.safetensors").read_bytes()).hexdigest())
        self.assertEqual((output / "config.json").read_bytes(),
                         (self.root / "config.json").read_bytes())
        self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)

    def make_indexed(self):
        pairs = list(self.original.items())
        files = {"part-1.safetensors": dict(pairs[:15]),
                 "part-2.safetensors": dict(pairs[15:])}
        (self.root / "model.safetensors").unlink()
        mapping = {}
        for name, tensors in files.items():
            write_tensor_file(self.root / name, tensors)
            mapping.update({tensor_name: name for tensor_name in tensors})
        index = {"metadata": {"total_size": sum(len(raw) for _, raw in self.original.values())},
                 "weight_map": mapping}
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        return index

    def test_indexed_checkpoint_exports_layers_across_source_files(self):
        self.make_indexed()
        checkpoint = inspect_checkpoint(self.root)
        self.assertEqual(len(checkpoint.source_files), 2)
        output = Path(self.tmp.name) / "indexed-export"
        export_shards(checkpoint, output, 0, 2)
        self.assertEqual(read_payloads(output / "mac.safetensors") |
                         read_payloads(output / "iphone.safetensors"), self.original)

    def test_index_path_traversal_rejected(self):
        index = self.make_indexed()
        index["weight_map"][next(iter(index["weight_map"]))] = "../outside.safetensors"
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaisesRegex(ManifestError, "Unsafe"):
            inspect_checkpoint(self.root)

    def test_index_backslash_path_rejected(self):
        index = self.make_indexed()
        index["weight_map"][next(iter(index["weight_map"]))] = "..\\outside.safetensors"
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaisesRegex(ManifestError, "Unsafe"):
            inspect_checkpoint(self.root)

    def test_external_symlink_rejected(self):
        source = self.root / "model.safetensors"
        outside = Path(self.tmp.name) / "outside.safetensors"
        source.rename(outside)
        source.symlink_to(outside)
        with self.assertRaisesRegex(ManifestError, "escapes"):
            inspect_checkpoint(self.root)

    def test_index_missing_and_misassigned_tensors_rejected(self):
        index = self.make_indexed()
        index["weight_map"]["missing.weight"] = "part-1.safetensors"
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaisesRegex(ManifestError, "missing"):
            inspect_checkpoint(self.root)
        del index["weight_map"]["missing.weight"]
        name = next(iter(index["weight_map"]))
        index["weight_map"][name] = "part-2.safetensors"
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaisesRegex(ManifestError, "assignment mismatch"):
            inspect_checkpoint(self.root)

    def test_duplicate_tensor_across_files_rejected(self):
        self.make_indexed()
        second = read_payloads(self.root / "part-2.safetensors")
        duplicate = next(iter(read_payloads(self.root / "part-1.safetensors")))
        second[duplicate] = self.original[duplicate]
        write_tensor_file(self.root / "part-2.safetensors", second)
        with self.assertRaisesRegex(ManifestError, "Duplicate tensors"):
            inspect_checkpoint(self.root)

    def test_incomplete_layer_rejected(self):
        del self.original["model.layers.1.mlp.down_proj.weight"]
        write_tensor_file(self.root / "model.safetensors", self.original)
        with self.assertRaisesRegex(ManifestError, "Incomplete layer 1"):
            inspect_checkpoint(self.root)

    def test_missing_attention_bias_rejected(self):
        del self.original["model.layers.1.self_attn.q_proj.bias"]
        write_tensor_file(self.root / "model.safetensors", self.original)
        with self.assertRaisesRegex(ManifestError, "Incomplete layer 1"):
            inspect_checkpoint(self.root)

    def test_layer_range_validation(self):
        checkpoint = inspect_checkpoint(self.root)
        for start, end in ((-1, 1), (1, 1), (0, 4), (False, 2), (0, 1.5)):
            with self.subTest(start=start, end=end), self.assertRaises(ManifestError):
                partition_tensors(checkpoint, start, end)

    def test_malformed_headers_rejected(self):
        valid = {"model.norm.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
        cases = [
            (b"{}", b"x"),  # trailing unreferenced data
            (b'{"x":{},"x":{}}', b""),
            (json.dumps({"x": {"dtype": "F16", "shape": [-1], "data_offsets": [0, 2]}}).encode(), b"xx"),
            (json.dumps({"x": {"dtype": "F16", "shape": [True], "data_offsets": [0, 2]}}).encode(), b"xx"),
            (json.dumps({"x": {"dtype": "F16", "shape": [2], "data_offsets": [0, 2]}}).encode(), b"xx"),
            (json.dumps({"x": {"dtype": "F16", "shape": [1], "data_offsets": [1, 3]}}).encode(), b"xxx"),
            (json.dumps({"x": {"dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 2]}}).encode(), b"xx"),
            (json.dumps(valid | {"x": valid["model.norm.weight"]}).encode(), b"xx"),
        ]
        for header, payload in cases:
            with self.subTest(header=header):
                (self.root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
                with self.assertRaises(ManifestError):
                    inspect_checkpoint(self.root)

    def test_truncated_and_oversized_headers_rejected(self):
        for raw in (b"", b"1234", struct.pack("<Q", 100_000_001),
                    struct.pack("<Q", 16) + b"{}"):
            (self.root / "model.safetensors").write_bytes(raw)
            with self.assertRaises(ManifestError):
                inspect_checkpoint(self.root)

    def test_source_mutation_rejected(self):
        checkpoint = inspect_checkpoint(self.root)
        with (self.root / "model.safetensors").open("ab") as stream:
            stream.write(b"x")
        with self.assertRaisesRegex(ManifestError, "changed"):
            export_shards(checkpoint, Path(self.tmp.name) / "export", 1, 2)

    def test_existing_output_preserved(self):
        output = Path(self.tmp.name) / "export"
        output.mkdir()
        (output / "keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            export_shards(self.root, output, 1, 2)
        self.assertEqual((output / "keep.txt").read_text(), "keep")

    def test_copy_failure_cleans_new_output_only(self):
        output = Path(self.tmp.name) / "export"
        with patch("mlx_peer.manifest._write_shard", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                export_shards(self.root, output, 1, 2)
        self.assertFalse(output.exists())
        self.assertTrue((self.root / "model.safetensors").exists())

    def test_streaming_export_never_reads_whole_tensor(self):
        checkpoint = inspect_checkpoint(self.root)
        real_open = Path.open
        reads = []

        class GuardedReader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def read(self, size=-1):
                if size < 0 or size > 7:
                    raise AssertionError(f"Unbounded tensor read: {size}")
                reads.append(size)
                return self.stream.read(size)

            def seek(self, offset):
                return self.stream.seek(offset)

        def guarded_open(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            if path.suffix == ".safetensors" and args and args[0] == "rb":
                return GuardedReader(stream)
            return stream

        with patch.object(Path, "open", guarded_open):
            export_shards(checkpoint, Path(self.tmp.name) / "export", 1, 2, chunk_size=7)
        self.assertTrue(reads)
        self.assertLessEqual(max(reads), 7)

    def test_missing_requested_path_rejected(self):
        with self.assertRaisesRegex(ManifestError, "does not exist"):
            inspect_checkpoint(self.root / "does-not-exist")

    def test_invalid_configuration_is_rejected_before_large_allocation(self):
        for update in ({"num_hidden_layers": 10 ** 50}, {"attention_bias": "false"},
                       {"tie_word_embeddings": "false"}, {"model_type": "qwen2_moe"}):
            with self.subTest(update=update):
                (self.root / "config.json").write_text(json.dumps(self.config | update))
                with self.assertRaises(ManifestError):
                    inspect_checkpoint(self.root)
        (self.root / "config.json").write_text('{"rope_theta":1e999}')
        with self.assertRaisesRegex(ManifestError, "Non-finite"):
            inspect_checkpoint(self.root)


if __name__ == "__main__":
    unittest.main()
