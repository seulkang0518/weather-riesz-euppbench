"""Run a fixed objective comparison; test evaluation requires an explicit flag."""
import argparse
from pathlib import Path
import subprocess
import sys
import socket
import yaml
from eupp_weather.common import digest, read_json, save_json, disjoint_output
from eupp_weather.data import Cases
from train import METHODS, validate_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/temperature48')
    p.add_argument('--config',default='configs/temperature48.yaml')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--gpus',type=int,choices=[1,2,4],default=1)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--workdir',default='runs/temperature48_seed42')
    p.add_argument('--evaluate-test',action='store_true',help='Explicitly score the fixed recipe on the operational test years')
    p.add_argument('--smoke-steps',type=int)
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or len(set(a.methods))!=len(a.methods): p.error('No duplicated seeds/methods')
    cfg=yaml.safe_load(Path(a.config).read_text()); validate_config(cfg)
    root=Path(a.workdir).resolve(); dataset=Cases(a.data,'train')
    disjoint_output(root,[a.data])
    root.mkdir(parents=True,exist_ok=True)
    spec={'config':cfg,'data_sha256':digest(Path(a.data)/'metadata.json'),'methods':a.methods,'seeds':a.seeds,
          'world_size':a.gpus,'device':a.device,'smoke_steps':a.smoke_steps,'synthetic':bool(dataset.metadata.get('synthetic'))}
    if (root/'experiment.json').exists() and read_json(root/'experiment.json')!=spec:
        raise ValueError('Workdir belongs to another fixed comparison')
    save_json(root/'experiment.json',spec)
    def run(command,log):
        print('Running: '+' '.join(command),flush=True)
        with log.open('a') as f: subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,cwd=Path(__file__).resolve().parent)
    rows=[]
    for seed in a.seeds:
        for method in a.methods:
            work=root/f'{method}/seed{seed}'; work.mkdir(parents=True,exist_ok=True)
            method_cfg=dict(cfg)
            if method=='flow_matching': method_cfg['learning_rate']=cfg.get('flow_learning_rate',cfg['learning_rate'])
            cp=work/'config.yaml'; cp.write_text(yaml.safe_dump(method_cfg,sort_keys=False))
            if a.evaluate_test:
                if not (work/'best.pt').exists(): raise FileNotFoundError('Complete validation-phase training first')
            else:
                if a.gpus>1:
                    # Single-node jobs: use an explicit loopback rendezvous rather
                    # than relying on cluster/Mac hostname or IPv6 resolution.
                    with socket.socket() as listener:
                        listener.bind(('127.0.0.1',0)); port=listener.getsockname()[1]
                    command=[sys.executable,'-m','torch.distributed.run',f'--nproc_per_node={a.gpus}',
                             '--rdzv-backend=static','--master-addr=127.0.0.1',f'--master-port={port}']
                else: command=[sys.executable,'-u']
                command+=['train.py','--data',str(Path(a.data).resolve()),'--config',str(cp),'--method',method,
                          '--seed',str(seed),'--device',a.device,'--workdir',str(work)]
                if a.smoke_steps: command+=['--smoke-steps',str(a.smoke_steps)]
                if (work/'manifest.json').exists(): command+=['--resume']
                run(command,work/'runner.log')
            split='test' if a.evaluate_test else 'val'
            evaluation=work/f'evaluation_{split}'
            if not (evaluation/'metrics.json').exists():
                command=[sys.executable,'-u','evaluate.py','--data',str(Path(a.data).resolve()),'--checkpoints',str(work/'best.pt'),
                         '--split',split,'--members',str(cfg['evaluation_members']),'--steps',str(cfg['sampling_steps']),
                         '--member-chunk',str(cfg.get('member_chunk',8)),'--batch-size',str(cfg['microbatch_cases']),
                         '--device',a.device,'--output',str(evaluation)]
                if a.smoke_steps or dataset.metadata.get('synthetic') or dataset.metadata.get('development_only'): command+=['--allow-smoke']
                run(command,work/f'evaluation_{split}.log')
            result=read_json(evaluation/'metrics.json')
            if (result['data_sha256']!=spec['data_sha256'] or result['method']!=method or result['split']!=split
                    or result['members']!=cfg['evaluation_members'] or result['models'][0]['sha256']!=digest(work/'best.pt')
                    or result['predictions_sha256']!=digest(evaluation/'predictions_anomaly.npy')):
                raise ValueError('Saved evaluation identity changed')
            for name,expected in result.get('evaluation_code_sha256',{}).items():
                if digest(Path(__file__).resolve().parent/name)!=expected:
                    raise ValueError('Evaluation code changed; use a new evaluation output, not cached scores')
            rows.append({'method':method,'seed':seed,'evaluation':str(evaluation),'metrics':result['metrics'],'inference':result['inference']})
            means={m:{k:sum(r['metrics'][k] for r in rows if r['method']==m)/sum(r['method']==m for r in rows)
                for k in result['metrics']} for m in a.methods if any(r['method']==m for r in rows)}
            save_json(root/f'summary_{split}.json',{'complete':len(rows)==len(a.seeds)*len(a.methods),'split':split,
                'experiment_sha256':digest(root/'experiment.json'),'rows':rows,'mean_scores':means,
                'note':'Per-seed model scores averaged; models are not pooled into an ensemble. Descriptive only; no significance claim.'})
            print(f'{method}, seed {seed}: ES {result["metrics"]["anomaly_energy_score"]:.4f}; '
                  f'LVS {result["metrics"]["anomaly_local_variogram_score"]:.4f}',flush=True)
    print(f'Summary: {root/f"summary_{split}.json"}',flush=True)


if __name__=='__main__': main()
