from mlx_peer.desktop_service import main

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        import json
        import mlx.core as mx
        from mlx_peer.runtime import checked_args
        from mlx_lm.models.qwen2 import Model
        from transformers import AutoTokenizer
        args = checked_args(dict(model_type="qwen2", hidden_size=8, intermediate_size=16,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            vocab_size=32, rms_norm_eps=1e-6, rope_theta=10000, max_position_embeddings=64))
        model = Model(args)
        result = model(mx.array([[1, 2]], dtype=mx.int32))
        mx.eval(result)
        assert result.shape == (1, 2, 32) and bool(mx.all(mx.isfinite(result)).item())
        report = {"gpu_inference": "passed", "shape": list(result.shape)}
        if len(sys.argv) > 2:
            tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True, trust_remote_code=False)
            assert tokenizer.encode("Hello")
            report["local_tokenizer"] = "passed"
        print(json.dumps(report))
    else:
        main()
