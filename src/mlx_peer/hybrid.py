"""Text-only Qwen3.8/qwen3_5 affine-4bit partitioning and assigned-layer runtimes.

Uses pinned MLX-LM blocks. Export never constructs a model or loads weight data
into MLX; the coordinator never constructs the phone's layers or vision tower.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .manifest import _inspect_file, _safe_file, _parse_json, _write_shard

PREFIX = "language_model.model.layers."
LAYER = re.compile(r"^language_model\.model\.layers\.(\d+)\.")


def checked_config(config):
    text = config.get('text_config', {})
    if config.get('model_type') != 'qwen3_5' or text.get('model_type') != 'qwen3_5_text':
        raise ValueError('Expected Qwen3.5 architecture used by Qwen3.8')
    if config.get('quantization') != {'group_size':64,'bits':4,'mode':'affine'}:
        raise ValueError('Only uniform affine 4-bit, group 64 is supported')
    if text.get('num_experts',0) or text.get('attention_bias',False) or text.get('hidden_act') != 'silu':
        raise ValueError('Only dense bias-free SiLU text models are supported')
    if text.get('tie_word_embeddings',False) or text.get('full_attention_interval') != 4:
        raise ValueError('Expected untied embeddings and four-layer hybrid groups')
    if text.get('rope_parameters',{}).get('rope_type', 'default') != 'default':
        raise ValueError('Only default text RoPE is supported')
    return text


def export(model_path, output, start, end, phone_only=False):
    root=Path(model_path).resolve(strict=True)
    config_bytes=(root/'config.json').read_bytes()
    config=_parse_json(config_bytes,'config.json')
    text=checked_config(config)
    if not 0 <= start < end <= text['num_hidden_layers']:
        raise ValueError('Invalid layer range')
    index=_parse_json((root/'model.safetensors.index.json').read_bytes(),'index')
    tensors={}; sources=[]
    for name in sorted(set(index['weight_map'].values())):
        found, source=_inspect_file(_safe_file(root,name),False)
        if tensors.keys() & found.keys():raise ValueError('Duplicate checkpoint tensor')
        if any(index['weight_map'].get(key)!=name for key in found):raise ValueError('Index/header mismatch')
        tensors.update(found);sources.append(source)
    if set(tensors)!=set(index['weight_map']):raise ValueError('Incomplete checkpoint')
    phone=[];mac=[];omitted=[]
    for key,tensor in sorted(tensors.items()):
        if key.startswith('vision_tower.') or key.startswith('model.visual.'):
            omitted.append(key);continue
        if not key.startswith('language_model.') or '.mtp.' in key:
            raise ValueError(f'Unsupported tensor namespace: {key}')
        match=LAYER.match(key)
        if match and int(match[1]) >= text['num_hidden_layers']:raise ValueError('Invalid layer index')
        if 'conv1d.weight' in key and tensor.shape[-1]!=1:
            raise ValueError('Unsanitized convolution checkpoint is not supported')
        (phone if match and start<=int(match[1])<end else mac).append(tensor)
    for i in range(text['num_hidden_layers']):
        prefix=f'{PREFIX}{i}.'
        suffix='linear_attn.in_proj_qkv.weight' if (i+1)%4 else 'self_attn.q_proj.weight'
        if prefix+suffix not in tensors:raise ValueError(f'Missing layer {i}')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_bytes(config_bytes)
    shards={'iphone':_write_shard(out/'weights.safetensors',phone,8*1024**2)}
    if not phone_only:shards['mac']=_write_shard(out/'mac.safetensors',mac,8*1024**2)
    for source in sources:
        st=source.path.stat()
        if (st.st_size,st.st_mtime_ns)!=(source.size_bytes,source.mtime_ns):raise ValueError('Source changed during export')
    request={'protocol_version':1,'layer_start':start,'layer_end':end,'session_id':out.name,
             'position':0,'max_context':8192}
    (out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    manifest={'model_family':'Qwen3.8','architecture':'qwen3_5','config_sha256':hashlib.sha256(config_bytes).hexdigest(),
        'source_headers':[s.to_dict() for s in sources], 'start':start,'end':end,'shards':shards,
        'omitted_vision_tensor_count':len(omitted),'text_only':True,'quantization':config['quantization']}
    (out/'hybrid-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def _modules():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import TextModelArgs, DecoderLayer
    from mlx_lm.models.cache import ArraysCache,KVCache
    from mlx_lm.models.base import create_attention_mask
    return mx,nn,TextModelArgs,DecoderLayer,ArraysCache,KVCache,create_attention_mask


class HybridStage:
    def __init__(self, config, weights_path, start, end, max_context=8192):
        mx,nn,Args,Layer,_,_,_= _modules()
        self.config=config; self.args=Args.from_dict(deepcopy(checked_config(config)))
        if not 0<=start<end<=self.args.num_hidden_layers:raise ValueError('Invalid stage range')
        self.start,self.end,self.max_context=start,end,max_context
        self.layers={}; weights=mx.load(str(weights_path)); consumed=set()
        for i in range(start,end):
            block=Layer(self.args,i)
            prefix=f'{PREFIX}{i}.'
            selected={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
            nn.quantize(block,group_size=64,bits=4,mode='affine',
                class_predicate=lambda name,module: hasattr(module,'to_quantized') and f'{name}.scales' in selected)
            block.load_weights(list(selected.items()),strict=True);block.eval()
            mx.eval(block.parameters())
            self.layers[i]=block;consumed.update(prefix+k for k in selected)
        if consumed!=set(weights):raise ValueError('Stage weights missing or unassigned')
        self.weight_bytes=sum(v.nbytes for v in weights.values())
        self.reset()

    def reset(self):
        _,_,_,_,ArraysCache,KVCache,_=_modules()
        self.cache={i:ArraysCache(size=2) if layer.is_linear else KVCache() for i,layer in self.layers.items()}
        self.position=0;self.failed=False

    def forward(self,hidden,position):
        mx,_,_,_,_,_,mask_fn=_modules()
        if self.failed or position!=self.position:raise ValueError('Reset required or cache position mismatch')
        if hidden.ndim!=3 or hidden.shape[0]!=1 or hidden.shape[2]!=self.args.hidden_size or hidden.dtype!=mx.bfloat16:
            raise ValueError('Expected BF16 [1,tokens,hidden]')
        if not 0<hidden.shape[1]<=self.max_context-self.position:raise ValueError('Context limit')
        try:
            cache=next((self.cache[i] for i,l in self.layers.items() if not l.is_linear),None)
            mask=mask_fn(hidden,cache)
            for i,layer in self.layers.items():
                hidden=layer(hidden,mask=None if layer.is_linear else mask,cache=self.cache[i])
                mx.eval(hidden,self.cache[i].state)
            self.position+=hidden.shape[1]
            return hidden
        except Exception:
            self.failed=True;raise


class HybridCoordinator:
    def __init__(self, directory, remote, max_context=8192, progress=lambda **_:None):
        mx,nn,Args,Layer,_,_,_=_modules()
        self.root=Path(directory)
        self.config=json.loads((self.root/'config.json').read_text())
        self.args=Args.from_dict(deepcopy(checked_config(self.config)))
        self.remote=remote;self.max_context=max_context
        manifest=json.loads((self.root/'hybrid-manifest.json').read_text())
        self.start,self.end=manifest['start'],manifest['end']
        if remote.status['weights_sha256']!=manifest['shards']['iphone']['sha256'] or remote.status['config_sha256']!=manifest['config_sha256']:
            raise ValueError('Remote shard identity mismatch')
        if (remote.status['start'],remote.status['end'])!=(self.start,self.end):raise ValueError('Remote range mismatch')
        weights=mx.load(str(self.root/'mac.safetensors'))
        self.weight_bytes=sum(v.nbytes for v in weights.values())
        self.embed=nn.QuantizedEmbedding(self.args.vocab_size,self.args.hidden_size,group_size=64,bits=4)
        self.head=nn.QuantizedLinear(self.args.hidden_size,self.args.vocab_size,bias=False,group_size=64,bits=4)
        self.norm=nn.RMSNorm(self.args.hidden_size,eps=self.args.rms_norm_eps)
        consumed=set()
        for module,prefix in [(self.embed,'language_model.model.embed_tokens.'),(self.head,'language_model.lm_head.'),(self.norm,'language_model.model.norm.')]:
            selected={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
            module.load_weights(list(selected.items()),strict=True);mx.eval(module.parameters())
            consumed.update(prefix+k for k in selected)
        self.layers={}
        for i in range(self.args.num_hidden_layers):
            if self.start<=i<self.end:continue
            block=Layer(self.args,i);prefix=f'{PREFIX}{i}.'
            selected={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
            nn.quantize(block,group_size=64,bits=4,mode='affine',class_predicate=lambda name,module:hasattr(module,'to_quantized') and f'{name}.scales' in selected)
            block.load_weights(list(selected.items()),strict=True);block.eval();mx.eval(block.parameters())
            self.layers[i]=block;consumed.update(prefix+k for k in selected)
            progress(phase='loading_mac_layers',layer=i,active_bytes=mx.get_active_memory())
        if consumed!=set(weights):raise ValueError('Mac shard keys mismatch')
        self.reset()

    def reset(self):
        _,_,_,_,ArraysCache,KVCache,_=_modules()
        self.cache={i:ArraysCache(size=2) if layer.is_linear else KVCache() for i,layer in self.layers.items()}
        self.position=0;self.failed=False;self.remote.reset()

    def forward(self,tokens):
        mx,_,_,_,_,_,mask_fn=_modules()
        if self.failed:raise ValueError('Reset required')
        if tokens.ndim!=2 or tokens.shape[0]!=1 or not 0<tokens.shape[1]<=self.max_context-self.position:
            raise ValueError('Invalid token input or context')
        try:
            hidden=self.embed(tokens)
            cache=next((self.cache[i] for i,l in self.layers.items() if not l.is_linear),None)
            mask=mask_fn(hidden,cache)
            for i in range(self.args.num_hidden_layers):
                if i==self.start:
                    mx.eval(hidden);hidden=self.remote.forward(hidden,self.position)
                if self.start<=i<self.end:continue
                layer=self.layers[i]
                hidden=layer(hidden,mask=None if layer.is_linear else mask,cache=self.cache[i])
                mx.eval(hidden,self.cache[i].state)
            logits=self.head(self.norm(hidden));mx.eval(logits)
            self.position+=tokens.shape[1]
            return logits
        except Exception:
            self.failed=True;raise
