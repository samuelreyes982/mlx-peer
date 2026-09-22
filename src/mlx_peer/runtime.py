"""Dense Qwen2 stages built from the pinned upstream MLX-LM implementation.

The coordinator constructs only its own layers, never a whole model followed by
deletion. The file probe executes both stages on the Mac and proves correctness,
not iPhone capacity. The USB benchmark implements the same stateful stage interface.
"""

from pathlib import Path
import hashlib
import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen2 import ModelArgs, TransformerBlock
from mlx_lm.models.cache import KVCache
from mlx_lm.models.base import create_attention_mask


def checked_args(config):
    if config.get("model_type") != "qwen2":
        raise ValueError("Only dense model_type=qwen2 is supported")
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("Quantized checkpoints are not supported by this stage yet")
    if config.get("rope_scaling") or config.get("use_sliding_window"):
        raise ValueError("RoPE scaling and sliding-window attention are not supported")
    if config.get("hidden_act", "silu") != "silu" or config.get("attention_bias", True) is not True:
        raise ValueError("Only SiLU and biased Q/K/V projections are supported")
    if config.get("rope_traditional", False):
        raise ValueError("Only nontraditional Qwen RoPE is supported")
    config = dict(config)
    config.setdefault("tie_word_embeddings", False)
    if type(config["tie_word_embeddings"]) is not bool:
        raise ValueError("tie_word_embeddings must be a boolean")
    args = ModelArgs.from_dict(config)
    dimensions = (args.hidden_size, args.intermediate_size, args.num_hidden_layers,
                  args.num_attention_heads, args.num_key_value_heads, args.vocab_size)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("Model dimensions must be positive integers")
    if args.hidden_size % args.num_attention_heads:
        raise ValueError("hidden_size must divide evenly into attention heads")
    if args.num_attention_heads % args.num_key_value_heads:
        raise ValueError("attention heads must be a multiple of KV heads")
    if config.get("head_dim", args.hidden_size // args.num_attention_heads) != args.hidden_size // args.num_attention_heads:
        raise ValueError("Unsupported head_dim override")
    return args


class LayerStage:
    def __init__(self, config, weights_path, start, end, max_context=256):
        self.args = checked_args(config)
        if not 0 <= start < end <= self.args.num_hidden_layers:
            raise ValueError("Invalid layer range")
        if not 0 < max_context <= self.args.max_position_embeddings:
            raise ValueError("Invalid max_context")
        self.start, self.end = start, end
        self.max_context = max_context
        with Path(weights_path).open("rb") as stream:
            self.weights_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.layers = [TransformerBlock(self.args) for _ in range(start, end)]
        weights = mx.load(str(weights_path))
        expected = set()
        from mlx.utils import tree_flatten
        for i, layer in zip(range(start, end), self.layers):
            prefix = f"model.layers.{i}."
            pairs = [(k[len(prefix):], v) for k, v in weights.items() if k.startswith(prefix)]
            expected.update(prefix + key for key, _ in tree_flatten(layer.parameters()))
            layer.load_weights(pairs, strict=True)
        if set(weights) != expected:
            raise ValueError("Stage weights contain missing or unassigned tensors")
        dtypes = {value.dtype for value in weights.values()}
        if len(dtypes) != 1 or next(iter(dtypes)) not in (mx.float16, mx.float32):
            raise ValueError("Stage weights must be uniformly float16 or float32")
        self.dtype = next(iter(dtypes))
        mx.eval([layer.parameters() for layer in self.layers])
        self.weight_bytes = sum(x.nbytes for x in weights.values())
        self.reset()

    def reset(self):
        self.cache = [KVCache() for _ in self.layers]
        self.position = 0
        self.failed = False

    def forward(self, hidden, position):
        if self.failed:
            raise ValueError("Stage failed; reset before reuse")
        if position != self.position:
            raise ValueError(f"Cache offset mismatch: expected {self.position}, got {position}")
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != self.args.hidden_size:
            raise ValueError("Expected hidden states shaped [1, tokens, hidden_size]")
        if hidden.dtype != self.dtype:
            raise ValueError("Activation dtype must match stage weights")
        if not 0 < hidden.shape[1] <= self.max_context - self.position:
            raise ValueError("Context budget exceeded")
        try:
            mask = create_attention_mask(hidden, self.cache[0])
            for layer, cache in zip(self.layers, self.cache):
                hidden = layer(hidden, mask=mask, cache=cache)
            mx.eval(hidden, [c.state for c in self.cache])
            self.position += hidden.shape[1]
            return hidden
        except Exception:
            self.failed = True
            raise


class MacCoordinator:
    """Own only Mac weights; invoke an injected, stateful remote stage."""
    def __init__(self, config, weights_path, start, end, remote_stage, max_context=256):
        self.args = checked_args(config)
        if not 0 <= start < end <= self.args.num_hidden_layers:
            raise ValueError("Invalid remote layer range")
        if not 0 < max_context <= self.args.max_position_embeddings:
            raise ValueError("Invalid max_context")
        self.start, self.end = start, end
        self.max_context = max_context
        self.remote_stage = remote_stage
        self.embed = nn.Embedding(self.args.vocab_size, self.args.hidden_size)
        self.norm = nn.RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)
        self.head = None if self.args.tie_word_embeddings else nn.Linear(
            self.args.hidden_size, self.args.vocab_size, bias=False)
        self.layers = {i: TransformerBlock(self.args) for i in range(self.args.num_hidden_layers)
                       if not start <= i < end}
        weights = mx.load(str(weights_path))
        expected = {"model.embed_tokens.weight", "model.norm.weight"}
        self.embed.load_weights([("weight", weights["model.embed_tokens.weight"])])
        self.norm.load_weights([("weight", weights["model.norm.weight"])])
        if self.head is not None:
            self.head.load_weights([("weight", weights["lm_head.weight"])])
            expected.add("lm_head.weight")
        from mlx.utils import tree_flatten
        for i, layer in self.layers.items():
            prefix = f"model.layers.{i}."
            layer.load_weights([(k[len(prefix):], v) for k, v in weights.items() if k.startswith(prefix)])
            expected.update(prefix + key for key, _ in tree_flatten(layer.parameters()))
        # Some tied checkpoints include a duplicate projection; reject rather than retain it.
        if set(weights) != expected:
            raise ValueError("Mac weights contain missing or unassigned tensors")
        if len({x.dtype for x in weights.values()}) != 1 or self.embed.weight.dtype not in (mx.float16, mx.float32):
            raise ValueError("Only uniform float16/float32 weights are supported")
        mx.eval(list(weights.values()))
        self.weight_bytes = sum(x.nbytes for x in weights.values())
        self.reset()

    def reset(self):
        self.cache = {i: KVCache() for i in self.layers}
        self.position = 0
        self.failed = False
        self.remote_stage.reset()

    def forward(self, tokens):
        if self.failed:
            raise ValueError("Coordinator failed; reset before reuse")
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.dtype not in (mx.int32, mx.uint32, mx.int64):
            raise ValueError("Tokens must be integer [1, tokens]")
        length = tokens.shape[1]
        if not 0 < length <= self.max_context - self.position:
            raise ValueError("Context budget exceeded")
        if mx.any(tokens < 0).item() or mx.any(tokens >= self.args.vocab_size).item():
            raise ValueError("Token ID out of range")
        try:
            h = self.embed(tokens)
            representative_cache = next(iter(self.cache.values()), None)
            mask = create_attention_mask(h, representative_cache)
            for i in range(self.start):
                h = self.layers[i](h, mask=mask, cache=self.cache[i])
            mx.eval(h)
            h = self.remote_stage.forward(h, self.position)
            if h.shape != (1, length, self.args.hidden_size) or h.dtype != self.embed.weight.dtype:
                raise ValueError("Remote stage returned incompatible activation")
            for i in range(self.end, self.args.num_hidden_layers):
                h = self.layers[i](h, mask=mask, cache=self.cache[i])
            h = self.norm(h)
            logits = self.embed.as_linear(h) if self.head is None else self.head(h)
            mx.eval(logits, [c.state for c in self.cache.values()])
            self.position += length
            return logits
        except Exception:
            self.failed = True
            raise


def load_partitioned(directory, remote_stage, max_context=256):
    """Load the Mac shard of an exported checkpoint with an explicit worker.

    The caller supplies a compatible local or remote stage. This
    returns a coordinator, not the future tokenizer/generation convenience API.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "mlx-peer-shards" or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported shard manifest")
    if manifest.get("config_file") != "config.json" or manifest["shards"]["mac"]["file"] != "mac.safetensors":
        raise ValueError("Unexpected manifest file names")
    for name, expected in (("config.json", manifest["config_sha256"]),
                           ("mac.safetensors", manifest["shards"]["mac"]["sha256"])):
        path = directory / name
        if path.is_symlink():
            raise ValueError("Shard files must not be symlinks")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    config = json.loads((directory / "config.json").read_text())
    if manifest.get("model_type") != config.get("model_type") or manifest.get("num_hidden_layers") != config.get("num_hidden_layers"):
        raise ValueError("Manifest model configuration mismatch")
    interval = manifest["remote_layer_range"]
    if interval.get("end_exclusive") is not True:
        raise ValueError("Expected exclusive layer range")
    if (remote_stage.start, remote_stage.end) != (interval["start"], interval["end"]):
        raise ValueError("Worker has the wrong layer range")
    if remote_stage.args != checked_args(config):
        raise ValueError("Worker configuration mismatch")
    if remote_stage.weights_sha256 != manifest["shards"]["iphone"]["sha256"]:
        raise ValueError("Worker shard checksum mismatch")
    return MacCoordinator(config, directory / "mac.safetensors", interval["start"],
                          interval["end"], remote_stage, max_context)
