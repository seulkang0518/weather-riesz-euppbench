"""Save matched-size ensembles and masked scores; validation is the default split."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from eupp_weather.common import digest, read_json, save_json, disjoint_output
from eupp_weather.data import Cases, to_device
from eupp_weather.model import build_model, predict
from eupp_weather.metrics import nearest_neighbors, score_arrays
from eupp_weather.preprocessing import inverse_targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--data',default='data/temperature48')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--members',type=int,default=51)
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--member-chunk',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--output',required=True)
    p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args()
    if min(a.members,a.steps,a.member_chunk,a.batch_size)<1 or len(a.checkpoints)>a.members:
        p.error('Positive counts required; at least one scenario per checkpoint')
    torch.set_num_threads(2); device=torch.device(a.device)
    dataset=Cases(a.data,a.split); meta=dataset.metadata
    root=Path(a.output).resolve(); disjoint_output(root,[a.data])
    for checkpoint in a.checkpoints:
        parent=Path(checkpoint).resolve().parent
        if root==parent or parent.is_relative_to(root):
            raise ValueError('Evaluation cannot overwrite or contain the checkpoint directory')
    states=[torch.load(c,map_location='cpu',weights_only=False) for c in a.checkpoints]
    manifests=[s['manifest'] for s in states]; method=manifests[0]['method']
    if len({m['seed'] for m in manifests})!=len(manifests): raise ValueError('Duplicate model seeds')
    for manifest in manifests:
        if manifest['data_sha256']!=digest(Path(a.data)/'metadata.json') or manifest['method']!=method:
            raise ValueError('Checkpoint data/method mismatch')
        if (manifest['synthetic'] or manifest.get('development_only') or manifest['smoke_only']) and not a.allow_smoke:
            raise ValueError('Development/synthetic checkpoints are not benchmark results; use --allow-smoke explicitly')
    identity={'data_sha256':digest(Path(a.data)/'metadata.json'),'split':a.split,'members':a.members,
        'steps':a.steps,'batch_size':a.batch_size,'member_chunk':a.member_chunk,'device':a.device,
        'checkpoints':{str(Path(c).resolve()):digest(c) for c in a.checkpoints},'allow_smoke':a.allow_smoke}
    if root.exists():
        if (root/'metrics.json').exists() or not (root/'experiment.json').exists() or read_json(root/'experiment.json')!=identity:
            raise FileExistsError('Evaluation belongs to another or completed run')
        print('Restarting the identical incomplete evaluation with fixed sampling seeds.',flush=True)
    root.mkdir(parents=True,exist_ok=True); save_json(root/'experiment.json',identity)
    output=np.lib.format.open_memmap(root/'predictions_anomaly.npy',mode='w+',dtype='float32',
        shape=(len(dataset),a.members,meta['stations'],len(meta['variables'])))
    calls=0; seconds=0.; offset=0; records=[]
    def sync():
        if device.type=='cuda': torch.cuda.synchronize()
    for index,(checkpoint,state) in enumerate(zip(a.checkpoints,states)):
        manifest=state['manifest']; cfg=manifest['config']; count=a.members//len(states)+int(index<a.members%len(states))
        model=build_model(meta,cfg['model']).to(device); model.load_state_dict(state['model']); model.eval()
        generator=torch.Generator(device=device).manual_seed(cfg['evaluation_seed']+manifest['seed'])
        position=0; warmed=False
        with torch.no_grad():
            for batch in DataLoader(dataset,batch_size=a.batch_size,shuffle=False):
                batch=to_device(batch,device); b=len(batch['target']); member=0
                while member<count:
                    take=min(a.member_chunk,count-member)
                    noise=torch.randn(b,take,meta['stations'],len(meta['variables']),device=device,generator=generator)
                    if not warmed:
                        predict(model,batch,noise,method,a.steps); sync(); warmed=True
                    sync(); start=time.perf_counter()
                    result=predict(model,batch,noise,method,a.steps); sync(); seconds+=time.perf_counter()-start
                    if not torch.isfinite(result).all(): raise FloatingPointError('Nonfinite generated scenarios')
                    output[position:position+b,offset+member:offset+member+take]=result.cpu().numpy()
                    calls+=a.steps if method=='flow_matching' else 1
                    member+=take
                position+=b
        records.append({'checkpoint':str(Path(checkpoint).resolve()),'sha256':digest(checkpoint),'seed':manifest['seed'],
                        'trained_epoch':state['epoch'],'members':count,'parameters':sum(p.numel() for p in model.parameters())})
        offset+=count; del model
    output.flush()
    neighbors=nearest_neighbors(meta['latitude'],meta['longitude'])
    per_case={}; raw_case={}
    for space in ('anomaly','physical'):
        pieces={}; raw_pieces={}
        for begin in range(0,len(dataset),a.batch_size):
            end=min(begin+a.batch_size,len(dataset)); sl=slice(begin,end)
            pred=np.asarray(output[sl]); target=np.asarray(dataset.arrays['target'][sl]); mask=np.asarray(dataset.arrays['mask'][sl])
            mean=np.asarray(dataset.arrays['obs_mean'][sl]); std=np.asarray(dataset.arrays['obs_std'][sl])
            if space=='physical':
                pred=inverse_targets(pred,mean[:,None],std[:,None],meta['variables'])
                target=inverse_targets(target,mean,std,meta['variables'])
            batch_scores=score_arrays(pred,target,mask,neighbors)
            for key,value in batch_scores.items(): pieces.setdefault(key,[]).append(value)
            if dataset.raw is not None:
                raw=np.asarray(dataset.raw[sl])
                if space=='anomaly':
                    raw=raw.astype('float64').copy()
                    for variable,name in enumerate(meta['variables']):
                        if name=='p10fg6': raw[...,variable]=np.log1p(raw[...,variable])
                    raw=(raw-mean[:,None])/std[:,None]
                for key,value in score_arrays(raw,target,mask,neighbors).items(): raw_pieces.setdefault(key,[]).append(value)
        for key,value in pieces.items(): per_case[space+'_'+key]=np.concatenate(value)
        for key,value in raw_pieces.items(): raw_case[space+'_'+key]=np.concatenate(value)
    valid_count=np.asarray(dataset.arrays['mask']).sum((1,2))
    means=lambda d:{key:float(np.average(value,weights=valid_count if key.endswith('mean_crps') else None)) for key,value in d.items()}
    np.savez_compressed(root/'per_case_scores.npz',**per_case,**{'raw_'+k:v for k,v in raw_case.items()},
                        initialization=dataset.index['initialization'],lead_hours=dataset.index['lead_hours'],valid_coordinates=valid_count)
    by_lead={str(lead):{k:float(np.average(v[dataset.index['lead_hours']==lead],
        weights=valid_count[dataset.index['lead_hours']==lead] if k.endswith('mean_crps') else None)) for k,v in per_case.items()}
        for lead in np.unique(dataset.index['lead_hours'])}
    summary={'method':method,'split':a.split,'data_sha256':digest(Path(a.data)/'metadata.json'),
        'evaluation_code_sha256':{str(f.relative_to(Path(__file__).resolve().parent)):digest(f)
            for f in [Path(__file__).resolve(),*sorted((Path(__file__).resolve().parent/'eupp_weather').glob('*.py'))]},
        'models':records,'members':a.members,'flow_network_evaluations_per_scenario':a.steps if method=='flow_matching' else 1,
        'cases':len(dataset),'metrics':means(per_case),'raw_ensemble_metrics':means(raw_case),'by_lead':by_lead,
        'inference':{'seconds':seconds,'network_batch_calls':calls,'batch_size':a.batch_size,'member_chunk':a.member_chunk,
            'scope':'model prediction, including Euler loop; excludes loading, noise draws, warmup, scoring and I/O',
            'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None},
        'synthetic':bool(meta.get('synthetic')),'development_only':bool(meta.get('development_only')),
        'smoke_only':any(m['smoke_only'] for m in manifests),
        'predictions_sha256':digest(root/'predictions_anomaly.npy'),
        'aggregation':'ES/VS/LVS equal case means; CRPS weighted by valid coordinate counts',
        'lvs':'5 nearest OTHER stations; p=.5; joint station-variable pairs; ordered pair sum; missing pairs excluded',
        'physical_units':{'t2m':'Kelvin (CRPS/differences identical to Celsius)','p10fg6':'m/s'},
        'note':'Independent matched-backbone adaptation; not a replication of the paper table; no significance claim.'}
    save_json(root/'metrics.json',summary)
    print(__import__('json').dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()

