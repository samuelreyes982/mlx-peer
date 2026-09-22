#!/usr/bin/env python3
"""Bounded text generation with separate hybrid shards; no full-model loader.

The optional local-stage control uses the same coordinator and shard files,
but executes both partitions on the Mac. The supervisor never imports MLX.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from record_mac_baseline import sample, write_json
from smoke_checks import load_cases, grade


def stop_worker(child):
    """Signal only this recorder's child; return cleanup errors for the report."""
    if child.poll() is not None:return None
    try:
        child.terminate()
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if child.poll() is None:child.kill()
            child.wait(timeout=5)
    except (OSError,subprocess.TimeoutExpired) as error:
        if child.poll() is None:return f'{type(error).__name__}: {error}'
    return None


def generate_case(mx, coordinator, tokenizer, generation_cfg, case, run, index, max_tokens, event):
    coordinator.reset()
    ids=list(tokenizer.apply_chat_template([{'role':'user','content':case['prompt']}],tokenize=True,
        add_generation_prompt=True,enable_thinking=False))
    if len(ids)+max_tokens>8192:raise ValueError('Context limit exceeded')
    write_json(run/f'case-{index}-prompt-tokens.json',ids)
    event('generation_start',case_id=case['id'],prompt_tokens=len(ids))
    started=time.monotonic();first=None;generated=[]
    for offset in range(0,len(ids),128):
        logits=coordinator.forward(mx.array([ids[offset:offset+128]],mx.int32))
    eos=set(generation_cfg['eos_token_id']);finish='length'
    for i in range(max_tokens):
        token=int(mx.argmax(logits[0,-1]).item())
        if first is None:first=time.monotonic()-started
        generated.append(token)
        event('token',case_id=case['id'],token=token,generation_tokens=len(generated),
            generation_elapsed_seconds=time.monotonic()-started,
            mlx_active_bytes=mx.get_active_memory(),mlx_peak_bytes=mx.get_peak_memory())
        response=tokenizer.decode(generated,skip_special_tokens=True)
        (run/f'case-{index}-response.txt').write_text(response)
        if index==0:(run/'response.txt').write_text(response)
        if token in eos:finish='stop';break
        if i+1<max_tokens:logits=coordinator.forward(mx.array([[token]],mx.int32))
    elapsed=time.monotonic()-started
    return {'case_id':case['id'],'prompt':case['prompt'],'prompt_tokens':len(ids),
        'generated_token_ids':generated,'generation_tokens':len(generated),
        'first_token_after_generation_start_seconds':first,'generation_seconds':elapsed,
        'decode_tokens_per_second':(len(generated)-1)/(elapsed-first) if len(generated)>1 else None,
        'finish_reason':finish,'response':response}


def worker(args):
    run=Path(args.output);root=Path(args.fixture);started=time.monotonic()
    events=(run/'events.jsonl').open('x',buffering=1)
    client=None;remote=None;old_wired=None
    result={'status':'error','physical_devices':1,'iphone_used':False,'iphone_requested':not args.local_stage,
        'exploratory':args.exploratory}
    def event(kind,**values):
        row={'event':kind,'elapsed_seconds':time.monotonic()-started,**values}
        events.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        import mlx.core as mx
        from mlx_lm.utils import load_tokenizer
        from mlx_peer.hybrid import HybridStage,HybridCoordinator
        from mlx_peer.wire import WireClient,connect_usb
        from validate_hybrid import RemoteHybridStage
        mx.random.seed(7);mx.set_cache_limit(256*1024**2)
        result['mlx_version']=mx.__version__
        # Apply the normal recommended working-set limit while loading too;
        # otherwise a near-capacity model can be paged out before generation.
        old_wired=mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
        manifest=json.loads((root/'hybrid-manifest.json').read_text())
        cfg=json.loads((root/'config.json').read_text())
        event('stage_load_start',local_stage=args.local_stage)
        if args.local_stage:
            remote=HybridStage(cfg,root/'weights.safetensors',manifest['start'],manifest['end'])
            remote.status={'weights_sha256':manifest['shards']['iphone']['sha256'],
                'config_sha256':manifest['config_sha256'],'start':manifest['start'],'end':manifest['end']}
        else:
            client=WireClient(connect_usb(args.serial,timeout=120),Path(args.token_file).read_text())
            remote=RemoteHybridStage(client,min_headroom_bytes=384*1024**2)
            if remote.status['available_process_memory_bytes']<384*1024**2:
                raise RuntimeError('Phone headroom below declared 384 MiB reserve')
        result['worker_initial_status']=remote.status
        if client:result.update(physical_devices=2,iphone_used=True)
        event('stage_ready',**remote.status)
        load_started=time.monotonic()
        coordinator=HybridCoordinator(root,remote,progress=lambda **kw:event('mac_load_progress',**kw))
        result['mac_weight_bytes']=coordinator.weight_bytes + (remote.weight_bytes if args.local_stage else 0)
        result['iphone_weight_bytes']=0 if args.local_stage else remote.status['weight_bytes']
        result['mac_load_seconds']=time.monotonic()-load_started
        event('load_complete',mlx_active_bytes=mx.get_active_memory(),mlx_peak_bytes=mx.get_peak_memory())
        generation_cfg=json.loads((Path(args.model)/'generation_config.json').read_text())
        tokenizer=load_tokenizer(args.model,{'local_files_only':True,'trust_remote_code':False},eos_token_ids=generation_cfg['eos_token_id'])
        cases=load_cases(run/'suite.json') if args.suite else [{'id':'baseline','prompt':(run/'prompt.txt').read_text()}]
        result['cases']=[]
        for index,case in enumerate(cases):
            timing_start=len(remote.timings) if client else 0
            generated=generate_case(mx,coordinator,tokenizer,generation_cfg,case,run,index,args.max_tokens,event)
            if args.suite:generated['assessment']=grade(case,generated,result['cases'])
            if client:
                generated['phone_requests']=len(remote.timings)-timing_start
                generated['phone_min_headroom_bytes']=min(t['available_process_memory_bytes'] for t in remote.timings[timing_start:])
            result['cases'].append(generated)
            event('case_complete',case_id=case['id'],response=generated['response'],assessment=generated.get('assessment'))
            write_json(run/'partial-result.json',{**result,'status':'running'})
        if not args.suite:
            result.update(result['cases'][0])
            (run/'prompt-tokens.json').write_bytes((run/'case-0-prompt-tokens.json').read_bytes())
        else:
            result['all_automated_smoke_checks_passed']=all(c['assessment']['passed'] for c in result['cases'])
            result['quality_benchmark_passed']=False
        result.update(status='completed',mlx_peak_bytes=mx.get_peak_memory(),mlx_active_bytes=mx.get_active_memory())
        if client:
            result['worker_final_status'],_=client.request('status')
            result['remote_timings']=remote.timings
    except Exception as error:
        result.update(error_type=type(error).__name__,error=str(error))
        traceback.print_exc();event('error',**result)
    finally:
        if old_wired is not None:
            mx.synchronize();mx.set_wired_limit(old_wired)
        if client:
            if remote:result['remote_timings']=remote.timings
            try:client.request('stop')
            except Exception:pass
            client.close()
        result['worker_elapsed_seconds']=time.monotonic()-started
        result['process_max_rss_bytes_macos']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(run/'worker-result.json',result);events.close()
    return 0 if result['status']=='completed' else 1


def main(args):
    root=Path(args.fixture).resolve(strict=True);run=Path(args.output).resolve()
    manifest=json.loads((root/'hybrid-manifest.json').read_text())
    validation_path=root/'usb-validation-result.json'
    parity_passed=validation_path.exists() and json.loads(validation_path.read_text()).get('passed') is True
    if not args.local_stage and not parity_passed and not args.exploratory:
        raise ValueError('Stage parity has not passed; use --exploratory only for explicitly labelled diagnostics')
    run.mkdir(parents=True,exist_ok=False)
    (run/'prompt.txt').write_text('In one short sentence, explain why the sky appears blue.')
    cases=load_cases(Path(args.suite)) if args.suite else None
    if cases:
        write_json(run/'suite.json',{'cases':cases})
        (run/'prompt.txt').write_text(cases[0]['prompt'])
    # Check exported files before timing, using bounded buffers and no model allocation.
    for name in ('iphone','mac'):
        info=manifest['shards'][name]
        with (root/info['file']).open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=info['sha256']:raise ValueError('Shard digest mismatch: '+name)
    plan={'model':'mlx-community/Qwen3.8-27B-4bit','revision':'3e6447f082e89cc7f0bc6e5441afd38dfce760ff',
        'local_stage_control':args.local_stage,'start':manifest['start'],'end':manifest['end'],
        'manifest':manifest,'max_tokens':args.max_tokens,'context_limit':8192,'prefill_step_size':128,
        'temperature':0,'seed':7,'thinking':False,'deadline_seconds':args.timeout,
        'numerical_stage_validation_passed':parity_passed,'exploratory':args.exploratory,
        'max_additional_system_swap_gib':args.max_swap_gib,'phone_headroom_reserve_bytes':384*1024**2,
        'mlx_cache_limit_bytes':256*1024**2,'wired_limit_policy':'MLX device recommended working set during loading and generation',
        'scope':'Short functional diagnostic, not sustained capacity or quality benchmark; phone provisioned before timed Mac process.',
        'suite_cases':cases,'reset_between_cases':True,
        'cache_state':'Uncontrolled OS cache; shard hashes read before timing',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'started_utc':datetime.now(timezone.utc).isoformat()}
    project=Path(__file__).resolve().parent.parent
    plan['source_sha256']={name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in (
        'src/mlx_peer/hybrid.py','scripts/validate_hybrid.py','scripts/smoke_checks.py','requirements-lock.txt',
        'ios/Sources/MLXPeerWorker/HybridStage.swift','ios/Sources/MLXPeerWorker/HybridDeltaKernel.swift',
        'ios/Sources/MLXPeerWorker/WireServer.swift','ios/Sources/MLXPeerWorker/FixtureRunner.swift')}
    write_json(run/'plan.json',plan)
    before=sample();write_json(run/'system-before.json',before)
    if before['swap_used_bytes'] is None:raise RuntimeError('Cannot monitor swap')
    argv=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--model',args.model,
        '--fixture',str(root),'--output',str(run),'--max-tokens',str(args.max_tokens)]
    if args.local_stage:argv+=['--local-stage']
    else:argv+=['--serial',args.serial,'--token-file',args.token_file]
    if args.exploratory:argv+=['--exploratory']
    if args.suite:argv+=['--suite',str(run/'suite.json')]
    started=time.monotonic();reason=None;cleanup_error=None
    with (run/'worker.log').open('x') as log,(run/'system-samples.jsonl').open('x',buffering=1) as samples:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false'))
        try:
            while child.poll() is None:
                row=sample(child.pid);row['elapsed_seconds']=time.monotonic()-started
                samples.write(json.dumps(row)+'\n')
                if row['elapsed_seconds']>=args.timeout:reason='deadline_guard'
                elif row['swap_used_bytes'] is None:reason='monitor_unavailable'
                elif row['swap_used_bytes']-before['swap_used_bytes']>=args.max_swap_gib*1024**3:reason='additional_swap_guard'
                if reason:
                    write_json(run/'guard-trigger.json',{'reason':reason,'sample':row})
                    cleanup_error=stop_worker(child);break
                time.sleep(1)
        finally:
            if child.poll() is None:cleanup_error=stop_worker(child) or cleanup_error
    after=sample();write_json(run/'system-after.json',after)
    result={'status':'guard_stopped' if reason else ('completed' if child.returncode==0 else 'worker_failed'),
        'stop_reason':reason,'exit_code':child.returncode,'elapsed_seconds':time.monotonic()-started,
        'cleanup_error':cleanup_error,'worker_still_running':child.poll() is None}
    write_json(run/'supervisor-result.json',result)
    if reason and not args.local_stage:
        from mlx_peer.wire import WireClient,connect_usb
        try:
            client=WireClient(connect_usb(args.serial,timeout=5),Path(args.token_file).read_text())
            client.request('stop');client.close()
        except Exception:pass
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','fixture','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--serial');p.add_argument('--token-file');p.add_argument('--local-stage',action='store_true')
    p.add_argument('--exploratory',action='store_true',help='Label a diagnostic with unresolved numerical parity explicitly')
    p.add_argument('--suite',help='JSON file containing predeclared functional smoke cases')
    p.add_argument('--max-tokens',type=int,default=64);p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--max-swap-gib',type=float,default=4);p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    if not args.local_stage and (not args.serial or not args.token_file):p.error('USB run needs serial and token file')
    if not 0<args.max_tokens<=8192 or args.timeout<=0 or args.max_swap_gib<=0:p.error('Invalid limits')
    sys.exit(worker(args) if args.worker else main(args))
