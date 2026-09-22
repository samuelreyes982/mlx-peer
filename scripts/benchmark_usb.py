#!/usr/bin/env python3
"""Application-level USB latency, throughput and live Qwen2 stage validation."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from mlx_peer.wire import WireClient, connect_usb


def stats(values):
    ordered = sorted(values)
    return {'samples': len(values), 'median_ms': statistics.median(values),
            'p95_ms': ordered[max(0, int(len(values) * .95 + .999999) - 1)],
            'min_ms': min(values), 'max_ms': max(values)}


def main(args):
    import numpy as np
    import mlx.core as mx
    from mlx_peer.runtime import MacCoordinator, checked_args

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    token = Path(args.token_file).read_text()
    fixture = Path(args.fixture)
    config = json.loads((fixture/'config.json').read_text())
    expected_weights = hashlib.file_digest((fixture/'weights.safetensors').open('rb'), 'sha256').hexdigest()
    expected_config = hashlib.sha256((fixture/'config.json').read_bytes()).hexdigest()
    client = WireClient(connect_usb(args.serial), token)
    report = {'transport': 'usbmux, physically USB-connected device only',
              'wifi_fallback': False, 'protocol_version': 1,
              'scope': 'Foreground debug build; application round trips and payload transfers. Not raw link speed or a 27B run.'}
    def save():
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    try:
        status, _ = client.request('status')
        assert status['weights_sha256'] == expected_weights
        assert status['config_sha256'] == expected_config
        report['worker'] = status
        print('Connected to verified USB worker.', flush=True)
        for _ in range(10): client.request('ping')
        measurements = []
        for _ in range(100):
            start = time.perf_counter(); client.request('ping')
            measurements.append((time.perf_counter()-start)*1000)
        report['ping'] = stats(measurements)
        payload = bytes(i % 251 for i in range(10 * 1024))
        measurements = []
        for _ in range(50):
            start = time.perf_counter(); _, body = client.request('echo', payload)
            measurements.append((time.perf_counter()-start)*1000)
            assert body == payload
        report['activation_sized_echo_10240_bytes_each_direction'] = stats(measurements)
        payload = bytes(range(256)) * (8*1024*1024//256)
        digest = hashlib.sha256(payload).hexdigest()
        report['throughput'] = {}
        for direction in ('upload','download'):
            values = []
            for _ in range(8):
                start = time.perf_counter()
                if direction == 'upload': reply, data = client.request('upload', payload)
                else: reply, data = client.request('download', bytes=len(payload))
                elapsed = time.perf_counter()-start
                if direction == 'upload': assert reply['sha256'] == digest and not data
                else: assert data == b'Z'*len(payload)
                values.append(len(payload)/elapsed/1e6)
            report['throughput'][direction] = {'payload_bytes':len(payload),'repetitions':len(values),
                'median_MB_per_second':statistics.median(values),'samples_MB_per_second':values}
        print(json.dumps({k:report[k] for k in ('ping','activation_sized_echo_10240_bytes_each_direction','throughput')},indent=2),flush=True)
        save()

        client.request('reset')
        request = json.loads((fixture/'request.json').read_text())
        report['live_stage_reference_checks'] = []
        for i, step in enumerate(request['steps']):
            hidden = mx.load(str(fixture/step['input']))['hidden_states']
            expected = np.array(mx.load(str(fixture/f'expected-{i}.safetensors'))['hidden_states'])
            payload = np.array(hidden).astype('<f2').tobytes()
            start = time.perf_counter()
            reply, body = client.request('forward', payload, tokens=hidden.shape[1], position=step['position'])
            elapsed = (time.perf_counter()-start)*1000
            actual = np.frombuffer(body,dtype='<f2').reshape(hidden.shape)
            passed = np.allclose(actual,expected,rtol=.01,atol=.01)
            report['live_stage_reference_checks'].append({'position':step['position'],
                'roundtrip_ms':elapsed,'phone_compute_ms':reply['compute_ms'],
                'max_absolute_error':float(np.max(np.abs(actual.astype('float32')-expected.astype('float32')))),
                'passed':bool(passed)})
            assert passed, 'Live stage output differs from reference'
        save()

        class RemoteStage:
            def __init__(self):
                self.args=checked_args(config)
                self.start,self.end=status['start'],status['end']
                self.weights_sha256=status['weights_sha256']
                self.timings=[]
            def reset(self): client.request('reset')
            def forward(self, hidden, position):
                mx.eval(hidden)
                body=np.array(hidden).astype('<f2').tobytes()
                start=time.perf_counter()
                reply, result=client.request('forward',body,tokens=hidden.shape[1],position=position)
                self.timings.append({'roundtrip_ms':(time.perf_counter()-start)*1000,'compute_ms':reply['compute_ms']})
                if reply.get('shape')!=list(hidden.shape) or reply.get('dtype')!='float16' or reply.get('position')!=position+hidden.shape[1]:
                    raise ValueError('Invalid remote activation metadata')
                return mx.array(np.frombuffer(result,dtype='<f2').copy().reshape(hidden.shape))

        # Compare full-model logits for fixed tokens, with the remote stage executed live.
        # The full reference is only a small validation model and is freed before the split.
        from mlx_lm.models.qwen2 import Model, ModelArgs
        from mlx_lm.models.cache import make_prompt_cache
        reference=Model(ModelArgs.from_dict(config))
        reference.load_weights(list(mx.load(str(fixture/'full.safetensors')).items()))
        mx.eval(reference.parameters())
        reference_cache=make_prompt_cache(reference)
        token_steps=[[1,42,53,64,75],[86],[97,108,119]]
        expected_logits=[]
        for ids in token_steps:
            logits=reference(mx.array([ids],mx.int32),cache=reference_cache)
            mx.eval(logits)
            expected_logits.append(np.array(logits))
        del reference,reference_cache,logits
        mx.clear_cache()
        remote=RemoteStage()
        coordinator=MacCoordinator(config,fixture/'mac.safetensors',remote.start,remote.end,remote,max_context=256)
        checks=[]
        for ids,expected in zip(token_steps,expected_logits):
            actual=np.array(coordinator.forward(mx.array([ids],mx.int32)))
            checks.append({'tokens':len(ids),'max_absolute_error':float(np.max(np.abs(actual.astype('float32')-expected.astype('float32')))),
                           'passed':bool(np.allclose(actual,expected,rtol=.02,atol=.05))})
        report['full_model_live_split_logits']={'tolerances':{'rtol':.02,'atol':.05},'checks':checks,
            'mac_weight_bytes':coordinator.weight_bytes,'iphone_weight_bytes':status['weight_bytes'],
            'capacity_gain_demonstrated':False,'scope':'Small trained model validation across two physical devices'}
        assert all(c['passed'] for c in checks), 'Split logits differ from complete reference'

        # Produce a short continuation through the same live split to exercise generation.
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
        prompt='Question: Why is the sky blue?\nAnswer:'
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        coordinator.reset()
        generated=[]; first=None
        start=time.perf_counter()
        for _ in range(24):
            logits=coordinator.forward(mx.array([ids],mx.int32))
            token_id=mx.argmax(logits[0,-1]).item()
            if first is None:first=time.perf_counter()-start
            if token_id==tokenizer.eos_token_id:break
            generated.append(token_id); ids=[token_id]
        elapsed=time.perf_counter()-start
        report['live_split_generation']={'prompt':prompt,'output':tokenizer.decode(generated),
            'generated_tokens':len(generated),'total_seconds':elapsed,'first_token_seconds':first,
            'scope':'24-token base-model continuation only; not a quality benchmark or capacity demonstration'}
        (output/'response.txt').write_text(report['live_split_generation']['output'])
        report['remote_request_timings']=remote.timings
        report['passed']=True
        save()
        print(json.dumps({k:report[k] for k in ('live_stage_reference_checks','full_model_live_split_logits','live_split_generation','passed')},indent=2),flush=True)
    finally:
        try: client.request('stop')
        except Exception: pass
        client.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial',required=True)
    p.add_argument('--token-file',required=True)
    p.add_argument('--fixture',required=True)
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--output',required=True)
    main(p.parse_args())
