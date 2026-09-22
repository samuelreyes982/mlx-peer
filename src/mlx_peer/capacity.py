"""Explicit, conservative planning estimates; never a runtime fit guarantee."""

from __future__ import annotations

import math
from typing import Any, Mapping

GIB = 1024 ** 3
MIB = 1024 ** 2


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def kv_cache_bytes(config: Mapping[str, Any], *, layer_count: int,
                   context_length: int, batch_size: int = 1,
                   cache_dtype_bytes: int = 2) -> int:
    """Dense attention KV = 2 * layers * batch * tokens * KV heads * head dim * bytes.

    Models using grouped-query attention store KV heads, not all query heads.
    Allocator cache growth/alignment and temporary buffers are extra.
    """
    _integer(layer_count, "layer_count")
    _integer(context_length, "context_length", 1)
    _integer(batch_size, "batch_size", 1)
    _integer(cache_dtype_bytes, "cache_dtype_bytes", 1)
    hidden = _integer(config.get("hidden_size"), "hidden_size", 1)
    query_heads = _integer(config.get("num_attention_heads"), "num_attention_heads", 1)
    kv_heads = _integer(config.get("num_key_value_heads", query_heads), "num_key_value_heads", 1)
    if hidden % query_heads or query_heads % kv_heads:
        raise ValueError("Hidden size and attention head counts are inconsistent")
    head_dim = _integer(config.get("head_dim", hidden // query_heads), "head_dim", 1)
    return 2 * layer_count * batch_size * context_length * kv_heads * head_dim * cache_dtype_bytes


def estimate_capacity(config: Mapping[str, Any], *, mac_weight_bytes: int,
                      iphone_weight_bytes: int, remote_layer_count: int,
                      context_length: int, batch_size: int = 1,
                      cache_dtype_bytes: int = 2,
                      mac_budget_bytes: int | None = None,
                      iphone_budget_bytes: int | None = None,
                      mac_runtime_reserve_bytes: int = 2 * GIB,
                      iphone_runtime_reserve_bytes: int = GIB,
                      allocation_margin_fraction: float = 0.20,
                      prefill_chunk_tokens: int = 128) -> dict[str, Any]:
    """Estimate each process budget; budgets must exclude OS/other app needs.

    Storage weight sizes can understate runtime weight allocations (conversion,
    repacking or unsupported quantization). These reserves are configurable
    planning assumptions, not empirically established bounds or device limits.
    Context length means the maximum cached prompt + generated tokens.
    """
    total_layers = _integer(config.get("num_hidden_layers"), "num_hidden_layers", 1)
    _integer(remote_layer_count, "remote_layer_count", 1)
    if remote_layer_count > total_layers:
        raise ValueError("remote_layer_count exceeds num_hidden_layers")
    for value, name in ((mac_weight_bytes, "mac_weight_bytes"),
                        (iphone_weight_bytes, "iphone_weight_bytes"),
                        (mac_runtime_reserve_bytes, "mac_runtime_reserve_bytes"),
                        (iphone_runtime_reserve_bytes, "iphone_runtime_reserve_bytes")):
        _integer(value, name)
    for budget, name in ((mac_budget_bytes, "mac_budget_bytes"),
                         (iphone_budget_bytes, "iphone_budget_bytes")):
        if budget is not None:
            _integer(budget, name, 1)
    if (isinstance(allocation_margin_fraction, bool)
            or not isinstance(allocation_margin_fraction, (int, float))
            or not math.isfinite(allocation_margin_fraction)
            or not 0 <= allocation_margin_fraction <= 10):
        raise ValueError("allocation_margin_fraction must be finite and between 0 and 10")
    _integer(prefill_chunk_tokens, "prefill_chunk_tokens", 1)
    hidden = _integer(config.get("hidden_size"), "hidden_size", 1)
    cache_options = {"context_length": context_length, "batch_size": batch_size,
                     "cache_dtype_bytes": cache_dtype_bytes}
    mac_cache = kv_cache_bytes(config, layer_count=total_layers - remote_layer_count,
                               **cache_options)
    phone_cache = kv_cache_bytes(config, layer_count=remote_layer_count, **cache_options)
    # Budget four simultaneous hidden-state buffers for send/receive plus copies.
    # MLP and attention workspaces remain covered only by assumed reserves.
    activation_buffers = (4 * batch_size * min(prefill_chunk_tokens, context_length)
                          * hidden * cache_dtype_bytes)

    def device(weights: int, cache: int, reserve: int, budget: int | None) -> dict[str, Any]:
        base = weights + cache + activation_buffers
        margin = math.ceil(base * allocation_margin_fraction)
        required = base + margin + reserve
        return {
            "weight_storage_bytes": weights,
            "kv_cache_bytes": cache,
            "activation_buffer_allowance_bytes": activation_buffers,
            "allocation_margin_bytes": margin,
            "runtime_reserve_bytes": reserve,
            "estimated_required_bytes": required,
            "process_budget_bytes": budget,
            "estimated_headroom_bytes": None if budget is None else budget - required,
            "budget_status": ("budget_not_provided" if budget is None else
                              "exceeds_assumed_budget" if required > budget else
                              "within_assumed_budget_requires_measurement"),
        }

    baseline = device(mac_weight_bytes + iphone_weight_bytes, mac_cache + phone_cache,
                      mac_runtime_reserve_bytes, mac_budget_bytes)
    return {
        "estimate_only": True, "runtime_fit_verified": False,
        "context_length": context_length, "batch_size": batch_size,
        "cache_dtype_bytes": cache_dtype_bytes,
        "remote_layer_count": remote_layer_count,
        "kv_formula": "2 * layers * batch * cached_tokens * kv_heads * head_dim * cache_dtype_bytes",
        "assumptions": {
            "allocation_margin_fraction": allocation_margin_fraction,
            "prefill_chunk_tokens": prefill_chunk_tokens,
            "activation_copies": 4,
            "budgets_are_process_budgets": True,
            "weights_remain_in_stored_representation": True,
        },
        "mac": device(mac_weight_bytes, mac_cache, mac_runtime_reserve_bytes, mac_budget_bytes),
        "iphone": device(iphone_weight_bytes, phone_cache, iphone_runtime_reserve_bytes,
                         iphone_budget_bytes),
        "mac_only_baseline": baseline,
        "limitations": [
            "These planning assumptions do not prove a model fits or cannot run on either device.",
            "Measure resident memory, MLX allocations, swap, cache growth and sustained generation.",
            "iOS process limits are device/runtime dependent; do not use advertised storage or total RAM as the app budget.",
            "Prefill workspaces, logits, format conversion and allocator behavior may exceed these reserves.",
        ],
    }
