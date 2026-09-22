import unittest

from mlx_peer.capacity import GIB, estimate_capacity, kv_cache_bytes


CONFIG = {"num_hidden_layers": 24, "hidden_size": 896,
          "num_attention_heads": 14, "num_key_value_heads": 2}


class CapacityTests(unittest.TestCase):
    def test_grouped_query_cache_uses_kv_heads(self):
        actual = kv_cache_bytes(CONFIG, layer_count=6, context_length=4096)
        self.assertEqual(actual, 2 * 6 * 4096 * 2 * 64 * 2)

    def test_cache_scales_with_batch_and_precision(self):
        a = kv_cache_bytes(CONFIG, layer_count=6, context_length=4096)
        b = kv_cache_bytes(CONFIG, layer_count=6, context_length=4096,
                           batch_size=2, cache_dtype_bytes=4)
        self.assertEqual(b, 4 * a)

    def test_estimate_conserves_weights_cache_and_never_claims_fit(self):
        report = estimate_capacity(CONFIG, mac_weight_bytes=12 * GIB,
                                   iphone_weight_bytes=3 * GIB, remote_layer_count=6,
                                   context_length=4096, mac_budget_bytes=14 * GIB,
                                   iphone_budget_bytes=6 * GIB)
        self.assertTrue(report["estimate_only"])
        self.assertFalse(report["runtime_fit_verified"])
        self.assertEqual(report["mac"]["budget_status"], "exceeds_assumed_budget")
        self.assertEqual(report["iphone"]["budget_status"],
                         "within_assumed_budget_requires_measurement")
        self.assertEqual(report["mac_only_baseline"]["weight_storage_bytes"], 15 * GIB)
        self.assertEqual(report["mac_only_baseline"]["kv_cache_bytes"],
                         report["mac"]["kv_cache_bytes"] + report["iphone"]["kv_cache_bytes"])
        self.assertGreater(report["mac"]["estimated_required_bytes"],
                           report["mac"]["weight_storage_bytes"] + report["mac"]["kv_cache_bytes"])

    def test_unknown_device_budget_remains_unknown(self):
        report = estimate_capacity(CONFIG, mac_weight_bytes=1000, iphone_weight_bytes=1000,
                                   remote_layer_count=6, context_length=16)
        self.assertIsNone(report["iphone"]["estimated_headroom_bytes"])
        self.assertEqual(report["iphone"]["budget_status"], "budget_not_provided")

    def test_invalid_configuration_and_values(self):
        for kwargs in ({"remote_layer_count": 25}, {"remote_layer_count": 0},
                       {"context_length": 0}, {"iphone_weight_bytes": -1},
                       {"cache_dtype_bytes": False}, {"batch_size": 0},
                       {"allocation_margin_fraction": float("nan")},
                       {"iphone_budget_bytes": -1}):
            args = {"mac_weight_bytes": 100, "iphone_weight_bytes": 100,
                    "remote_layer_count": 6, "context_length": 128}
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                estimate_capacity(CONFIG, **args)
        with self.assertRaises(ValueError):
            kv_cache_bytes(CONFIG | {"hidden_size": 895}, layer_count=1, context_length=2)


if __name__ == "__main__":
    unittest.main()
