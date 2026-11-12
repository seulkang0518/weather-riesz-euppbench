"""Train matched station generators; torchrun supports four-GPU DDP."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import time
import fcntl
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import yaml
from eupp_weather.common import digest, read_json, save_json, disjoint_output
from eupp_weather.data import Cases, to_device
from eupp_weather.model import build_model, predict
from eupp_weather.objectives import distribution_loss, energy_score, flow_pair, masked_mse

METHODS = ('energy','riesz_softmax','riesz_uniform','flow_matching')


def validate_config(cfg):
    for key in ('epochs','microbatch_cases','global_batch_cases','train_particles','validation_members','sampling_steps'):
        if not isinstance(cfg[key],int) or cfg[key]<1: raise ValueError(f'Invalid {key}')
    if cfg['train_particles']<2 or cfg['validation_members']<2: raise ValueError('Need at least two scenarios')
    if cfg['energy_estimator'] not in ('empirical','unbiased'): raise ValueError('Invalid ES estimator')
    if not cfg['temperatures'] or not all(np.isfinite(t) and t>0 for t in cfg['temperatures']):
        raise ValueError('Positive finite temperatures required')
    if not np.isfinite(cfg['learning_rate']) or cfg['learning_rate']<=0: raise ValueError('Invalid LR')


@torch.no_grad()
def validate(model, data, cfg, method, device, limit=None):
    model.eval(); total=0.; count=0
    gen=torch.Generator(device=device).manual_seed(cfg['validation_seed'])
    for index,batch in enumerate(DataLoader(data,batch_size=cfg['microbatch_cases'],shuffle=False)):
        if limit is not None and index>=limit: break
        batch=to_device(batch,device)
        shape=(len(batch['target']),cfg['validation_members'],data.metadata['stations'],len(data.metadata['variables']))
        noise=torch.randn(shape,generator=gen,device=device)
        pred=predict(model,batch,noise,method,cfg['sampling_steps'])
        total+=energy_score(pred,batch['target'],batch['mask']).sum().item(); count+=len(pred)
    if count==0: raise ValueError('No validation cases')
    return total/count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/temperature48')
    p.add_argument('--config',default='configs/temperature48.yaml')
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--workdir',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke-steps',type=int,help='Development only; refuse benchmark evaluation of resulting fits')
    a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); validate_config(cfg)
    if a.smoke_steps is not None and a.smoke_steps<1: p.error('Positive smoke step count required')
    world=int(os.environ.get('WORLD_SIZE',1)); rank=int(os.environ.get('RANK',0)); local=int(os.environ.get('LOCAL_RANK',0))
    if world>1:
        dist.init_process_group('nccl' if a.device=='cuda' else 'gloo')
    device=torch.device(f'cuda:{local}' if a.device=='cuda' else 'cpu')
    if device.type=='cuda': torch.cuda.set_device(device)
    torch.set_num_threads(cfg.get('cpu_threads',2))
    if cfg['global_batch_cases']%(world*cfg['microbatch_cases']):
        raise ValueError('Global batch must be divisible by world_size * microbatch_cases')
    accumulation=cfg['global_batch_cases']//(world*cfg['microbatch_cases'])
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(a.seed)
    train, val=Cases(a.data,'train'),Cases(a.data,'val')
    if rank==0:
        for name, expected in train.metadata.get('files_sha256',{}).items():
            if digest(Path(a.data)/name)!=expected: raise ValueError(f'Prepared data changed: {name}')
    if world>1: dist.barrier()
    root=Path(a.workdir).resolve(); disjoint_output(root,[a.data])
    source_root=Path(__file__).resolve().parent
    code_hashes={str(f.relative_to(source_root)):digest(f) for f in [source_root/'train.py',*sorted((source_root/'eupp_weather').glob('*.py'))]}
    meta=train.metadata
    manifest={'config':cfg,'method':a.method,'seed':a.seed,'world_size':world,
        'data_sha256':digest(Path(a.data)/'metadata.json'),'data_metadata':meta,'code_sha256':code_hashes,
        'synthetic':bool(meta.get('synthetic')),'development_only':bool(meta.get('development_only')),'smoke_only':bool(a.smoke_steps),
        'torch':torch.__version__,'checkpoint_selection':'empirical validation ES with fixed noise for every method',
        'normalization_scope':'Riesz distance/RMS scales are local to each rank and microbatch; keep this partition fixed in comparisons'}
    lock=None
    if rank==0:
        root.mkdir(parents=True,exist_ok=True)
        lock=(root/'.run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'manifest.json').exists():
            if not a.resume: raise FileExistsError('Run exists; use --resume or a new directory')
            if read_json(root/'manifest.json')!=manifest: raise ValueError('Resume manifest mismatch')
        else:
            if a.resume: raise FileNotFoundError('No run to resume')
            save_json(root/'manifest.json',manifest)
    if world>1: dist.barrier()
    model=build_model(meta,cfg['model']).to(device)
    wrapped=DistributedDataParallel(model,device_ids=[local] if device.type=='cuda' else None) if world>1 else model
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg.get('weight_decay',0.))
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,seed=a.seed,shuffle=True,drop_last=True)
    loader=DataLoader(train,batch_size=cfg['microbatch_cases'],sampler=sampler,drop_last=True,num_workers=cfg.get('workers',0),pin_memory=device.type=='cuda')
    if not len(loader): raise ValueError('Not enough cases for a training batch')
    steps_per_epoch=(len(loader)+accumulation-1)//accumulation
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=cfg['learning_rate'],epochs=cfg['epochs'],steps_per_epoch=steps_per_epoch)
    noise_gen=torch.Generator(device=device).manual_seed(a.seed+1000+rank)
    epoch_start=0; step=0; best=float('inf')
    if a.resume:
        saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
        if saved['manifest']!=manifest: raise ValueError('Checkpoint manifest mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        epoch_start=saved['epoch']; step=saved['step']; best=saved['best']
        rng=saved['rng'][rank]; torch.set_rng_state(rng['torch']); noise_gen.set_state(rng['noise'])
        if device.type=='cuda': torch.cuda.set_rng_state(rng['cuda'],device)
    if a.smoke_steps and step>=a.smoke_steps:
        if rank==0: print('Requested smoke steps already completed; no refitting.',flush=True)
        if world>1: dist.destroy_process_group()
        return
    amp_enabled=bool(cfg.get('bf16',True)) and device.type=='cuda' and torch.cuda.is_bf16_supported()
    started=time.perf_counter()
    if rank==0: print(json.dumps({'method':a.method,'stations':meta['stations'],'variables':meta['variables'],
        'parameters':sum(p.numel() for p in model.parameters()),'training_particles':cfg['train_particles'],
        'world_size':world,'accumulation':accumulation,'workdir':str(root),'smoke':bool(a.smoke_steps)}),flush=True)
    for epoch in range(epoch_start,cfg['epochs']):
        sampler.set_epoch(epoch); wrapped.train(); losses=[]; optimizer.zero_grad(set_to_none=True)
        bucket_size=accumulation
        for micro,batch in enumerate(loader):
            if micro%accumulation==0: bucket_size=min(accumulation,len(loader)-micro)
            last=(micro%accumulation==bucket_size-1) or micro==len(loader)-1
            batch=to_device(batch,device); b=len(batch['target'])
            sync=contextlib.nullcontext() if world==1 or last else wrapped.no_sync()
            with sync, torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp_enabled):
                if a.method=='flow_matching':
                    noise=torch.randn(batch['target'].shape,device=device,generator=noise_gen)
                    time_flow=torch.randn(b,device=device,generator=noise_gen).sigmoid()
                    residual=(batch['target']-batch['baseline'])/batch['residual_scale']
                    residual=torch.where(batch['mask'],residual,0.)
                    state,target_velocity=flow_pair(residual,noise,time_flow,cfg.get('sigma_min',1e-5))
                    loss=masked_mse(wrapped(state,batch['condition'],time_flow).float(),target_velocity,batch['mask'])
                else:
                    noise=torch.randn(b,cfg['train_particles'],meta['stations'],len(meta['variables']),device=device,generator=noise_gen)
                    pred=predict(wrapped,batch,noise,a.method,cfg['sampling_steps']).float()
                    loss,_=distribution_loss(pred,batch['target'],batch['mask'],a.method,cfg['temperatures'],cfg['energy_estimator'])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                (loss/bucket_size).backward()
            losses.append(float(loss.detach()))
            if last:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.get('max_grad_norm',1.),error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step+=1
                if rank==0 and (step%cfg.get('log_every',10)==0 or a.smoke_steps):
                    print(json.dumps({'epoch':epoch+1,'step':step,'loss':float(np.mean(losses[-bucket_size:])),
                        'gradient_norm':float(norm),'lr':scheduler.get_last_lr()[0]}),flush=True)
                if a.smoke_steps and step>=a.smoke_steps: break
        score=None
        if rank==0:
            score=validate(model,val,cfg,a.method,device,limit=2 if a.smoke_steps else None)
        if world>1:
            score_tensor=torch.tensor([score if rank==0 else 0.],device=device,dtype=torch.float64)
            dist.broadcast(score_tensor,0); score=score_tensor.item()
        improved=score<best; best=min(best,score)
        local_rng={'torch':torch.get_rng_state(),'noise':noise_gen.get_state().cpu(),
                   'cuda':torch.cuda.get_rng_state(device).cpu() if device.type=='cuda' else None}
        states=[None]*world
        if world>1: dist.all_gather_object(states,local_rng)
        else: states=[local_rng]
        if rank==0:
            payload={'manifest':manifest,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'epoch':epoch+1,'step':step,'best':best,'rng':states}
            torch.save(payload,root/'latest.tmp'); os.replace(root/'latest.tmp',root/'latest.pt')
            if improved: torch.save(payload,root/'best.tmp'); os.replace(root/'best.tmp',root/'best.pt')
            record={'epoch':epoch+1,'step':step,'validation_es':score,'best_validation_es':best,
                    'elapsed_seconds':time.perf_counter()-started}
            with (root/'train.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if world>1: dist.barrier()
        if a.smoke_steps: break
    if world>1: dist.destroy_process_group()


if __name__=='__main__': main()
