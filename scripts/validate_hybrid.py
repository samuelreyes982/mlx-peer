#!/usr/bin/env python3
"""Check the actual quantized hybrid layers against pinned MLX-LM, offline or USB.

The reference reads only selected embedding rows and the exported phone layers.
No full 27B model is constructed. Tolerances are recorded before Swift executes.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from mlx_peer.hybrid import HybridStage
from mlx_peer.manifest import _inspect_file, _safe_file
from mlx_peer.wire import WireClient, connect_usb


def prepare(root, model):
    config = json.loads((root/'config.json').read_text())
    request = json.loads((root/'request.json').read_text())
    index = json.loads((model/'model.safetensors.index.json').read_text())
    prefix = 'language_model.model.embed_tokens.'
    tensors = {}
    for name in {index['weight_map'][prefix+s] for s in ('weight','scales','biases')}:
        found, _ = _inspect_file(_safe_file(model,name),False)
        tensors.update(found)
    stage = HybridStage(config,root/'weights.safetensors',request['layer_start'],request['layer_end'])
    ids_steps = [[1,42,53,64,75],[86],[97,108,119]]
    steps = []
    for i, ids in enumerate(ids_steps):
        selected = []
        for suffix in ('weight','scales','biases'):
            info = tensors[prefix+suffix]
            dtype = '<u4' if info.dtype == 'U32' else '<u2'
            mapped = np.memmap(info.source_file,mode='r',dtype=dtype,offset=info.absolute_offset,shape=info.shape)
            rows = mx.array(np.array(mapped[ids],copy=True))
            selected.append(rows if suffix=='weight' else rows.view(mx.bfloat16))
            del mapped
        hidden = mx.dequantize(*selected,group_size=64,bits=4).reshape(1,len(ids),-1)
        pos = stage.position
        expected = stage.forward(hidden,pos)
        mx.eval(hidden,expected)
        mx.save_safetensors(str(root/f'input-{i}.safetensors'),{'hidden_states':hidden},metadata={'format':'mlx-peer-v1'})
        mx.save_safetensors(str(root/f'expected-{i}.safetensors'),{'hidden_states':expected},metadata={'format':'mlx-peer-v1'})
        steps.append({'input':f'input-{i}.safetensors','output':f'output-{i}.safetensors','position':pos})
    request['steps'] = steps
    (root/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    validation = {'reference':f'MLX-LM {importlib.metadata.version("mlx-lm")} qwen3_5 DecoderLayer; MLX {mx.__version__}',
        'reference_inputs':'Actual checkpoint embeddings of fixed token IDs',
        'token_steps':ids_steps,'dtype':'bfloat16','rtol':.03,'atol':.03,
        'tolerances_declared_before_swift_run':True,
        'rationale':'BF16 activations, quantized GPU reductions across different MLX builds; per-element criterion, not output text alone.',
        'weight_bytes':stage.weight_bytes,'reference_peak_mlx_bytes':mx.get_peak_memory()}
    (root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2),flush=True)


class RemoteHybridStage:
    def __init__(self, client, min_headroom_bytes=0):
        self.client=client
        self.min_headroom_bytes=min_headroom_bytes
        self.status,_=client.request('status')
        if self.status.get('dtype')!='bfloat16':raise ValueError('Expected BF16 worker')
        self.timings=[]
    def reset(self):self.client.request('reset')
    def forward(self,hidden,position):
        if hidden.dtype!=mx.bfloat16:raise ValueError('Wire requires BF16 activations')
        mx.eval(hidden)
        payload=np.array(hidden.view(mx.uint16)).astype('<u2').tobytes()
        started=time.perf_counter()
        reply,body=self.client.request('forward',payload,tokens=hidden.shape[1],position=position)
        self.timings.append({'roundtrip_ms':(time.perf_counter()-started)*1000,**reply})
        if reply.get('available_process_memory_bytes',0)<self.min_headroom_bytes:
            raise RuntimeError('Phone headroom fell below the declared reserve')
        if reply.get('shape')!=list(hidden.shape) or reply.get('dtype')!='bfloat16' or reply.get('position')!=position+hidden.shape[1]:
            raise ValueError('Invalid remote activation metadata')
        if len(body)!=hidden.size*2:raise ValueError('Invalid remote activation bytes')
        return mx.array(np.frombuffer(body,dtype='<u2').copy()).view(mx.bfloat16).reshape(hidden.shape)


def verify(args,root):
    validation=json.loads((root/'validation.json').read_text())
    request=json.loads((root/'request.json').read_text())
    client=None;remote=None
    report={'mode':args.mode,'rtol':validation['rtol'],'atol':validation['atol'],'checks':[]}
    try:
        if args.mode=='usb':
            client=WireClient(connect_usb(args.serial,timeout=120),Path(args.token_file).read_text())
            remote=RemoteHybridStage(client)
            manifest=json.loads((root/'hybrid-manifest.json').read_text())
            assert remote.status['weights_sha256']==manifest['shards']['iphone']['sha256']
            assert remote.status['config_sha256']==manifest['config_sha256']
            report['worker']=remote.status
            remote.reset()
        for i,step in enumerate(request['steps']):
            expected=mx.load(str(root/f'expected-{i}.safetensors'))['hidden_states']
            if remote:
                hidden=mx.load(str(root/step['input']))['hidden_states']
                actual=remote.forward(hidden,step['position'])
            else:
                actual=mx.load(str(root/step['output']))['hidden_states']
            if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise ValueError('Invalid output shape/dtype')
            a=np.array(actual.astype(mx.float32));e=np.array(expected.astype(mx.float32))
            passed=np.allclose(a,e,rtol=validation['rtol'],atol=validation['atol'])
            report['checks'].append({'position':step['position'],'tokens':actual.shape[1],
                'max_absolute_error':float(np.max(np.abs(a-e))),
                'relative_rms_error':float(np.linalg.norm(a-e)/max(np.linalg.norm(e),1e-12)),
                'passed':bool(passed)})
        if remote:report['timings']=remote.timings
        report['passed']=all(c['passed'] for c in report['checks'])
        destination=Path(args.output) if args.output else root/f'{args.mode}-validation-result.json'
        destination.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if not report['passed']:raise SystemExit(1)
    finally:
        if client:
            try:client.request('stop')
            except Exception:pass
            client.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','file','usb'])
    p.add_argument('--fixture',required=True)
    p.add_argument('--model');p.add_argument('--serial');p.add_argument('--token-file');p.add_argument('--output')
    args=p.parse_args();root=Path(args.fixture)
    mx.set_cache_limit(64*1024**2)
    if args.mode=='prepare':prepare(root,Path(args.model))
    else:verify(args,root)
