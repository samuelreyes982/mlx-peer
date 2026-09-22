"""Upstream full-model parity for the hybrid partition, including cached chunks."""
from copy import deepcopy
import json

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_peer.hybrid import HybridCoordinator, HybridStage, export, checked_config


def config():
    return {'model_type':'qwen3_5','quantization':{'group_size':64,'bits':4,'mode':'affine'},
        'text_config':{'model_type':'qwen3_5_text','hidden_size':128,'intermediate_size':256,
            'num_hidden_layers':8,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':64,
            'linear_num_value_heads':2,'linear_num_key_heads':2,'linear_key_head_dim':32,
            'linear_value_head_dim':32,'linear_conv_kernel_dim':4,'full_attention_interval':4,
            'rms_norm_eps':1e-6,'vocab_size':256,'max_position_embeddings':256,
            'tie_word_embeddings':False,'attention_bias':False,'hidden_act':'silu',
            'rope_parameters':{'rope_type':'default','rope_theta':10000000,'partial_rotary_factor':.25}}}


@pytest.fixture
def model_and_shards(tmp_path):
    cfg=config();mx.random.seed(9)
    reference=TextModel(TextModelArgs.from_dict(deepcopy(cfg['text_config'])))
    reference.update(tree_map(lambda x:x.astype(mx.bfloat16),reference.parameters()))
    nn.quantize(reference,group_size=64,bits=4,mode='affine')
    reference.eval()
    weights={'language_model.'+k:v for k,v in tree_flatten(reference.parameters())}
    source=tmp_path/'source';source.mkdir()
    mx.save_safetensors(str(source/'model.safetensors'),weights,metadata={'format':'mlx'})
    (source/'config.json').write_text(json.dumps(cfg))
    (source/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model.safetensors' for k in weights}}))
    shard=tmp_path/'shard';manifest=export(source,shard,0,4)
    remote=HybridStage(cfg,shard/'weights.safetensors',0,4)
    remote.status={'weights_sha256':manifest['shards']['iphone']['sha256'],
        'config_sha256':manifest['config_sha256'],'start':0,'end':4}
    return reference,remote,shard


def test_hybrid_split_matches_full_upstream_and_reset(model_and_shards):
    reference,remote,shard=model_and_shards
    coordinator=HybridCoordinator(shard,remote)
    assert set(coordinator.layers)=={4,5,6,7}
    for _ in range(2):
        cache=reference.make_cache();coordinator.reset()
        for ids in ([1,42,53,64,75],[86],[97,108,119]):
            tokens=mx.array([ids],mx.int32)
            expected=reference(tokens,cache=cache)
            actual=coordinator.forward(tokens)
            assert mx.allclose(actual,expected,atol=.02,rtol=.02).item()


def test_hybrid_worker_failure_requires_reset(model_and_shards):
    _,remote,shard=model_and_shards
    coordinator=HybridCoordinator(shard,remote)
    remote.position=4
    with pytest.raises(ValueError,match='position'):coordinator.forward(mx.array([[1]]))
    with pytest.raises(ValueError,match='Reset'):coordinator.forward(mx.array([[1]]))
    coordinator.reset()
    assert coordinator.forward(mx.array([[1]])).shape==(1,1,256)


def test_hybrid_rejects_wrong_remote_identity(model_and_shards):
    _,remote,shard=model_and_shards
    remote.status['weights_sha256']='wrong'
    with pytest.raises(ValueError,match='identity'):HybridCoordinator(shard,remote)


def test_hybrid_rejects_bad_inputs_without_advancing(model_and_shards):
    _,remote,_=model_and_shards
    for x in (mx.array(1),mx.zeros((1,1,128)),mx.zeros((2,1,128),mx.bfloat16),mx.zeros((1,1,127),mx.bfloat16)):
        with pytest.raises(ValueError):remote.forward(x,0)
    assert remote.position==0


@pytest.mark.parametrize('field,value',[('num_experts',8),('attention_bias',True),('hidden_act','relu'),('tie_word_embeddings',True),('full_attention_interval',8)])
def test_hybrid_unsupported_variant_rejected(field,value):
    cfg=config();cfg['text_config'][field]=value
    with pytest.raises(ValueError):checked_config(cfg)
