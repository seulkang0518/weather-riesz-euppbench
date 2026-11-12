"""Small wiring check, explicitly NOT EUPPBench data or scientific evidence."""
import argparse
from pathlib import Path
import numpy as np
from eupp_weather.common import digest,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/synthetic')
    p.add_argument('--stations',type=int,default=122)
    p.add_argument('--variables',type=int,choices=[1,2],default=2)
    a=p.parse_args(); root=Path(a.output)
    if root.exists(): raise FileExistsError('Choose a new synthetic output')
    rng=np.random.default_rng(512); root.mkdir(parents=True)
    for split,n,year in [('train',16,1999),('val',6,2003),('test',6,2018)]:
        out=root/split; out.mkdir()
        condition=rng.normal(size=(n,a.stations,10)).astype('float32')
        baseline=condition[...,:a.variables]
        target=baseline+.2*rng.normal(size=(n,a.stations,a.variables))
        mask=rng.uniform(size=target.shape)>.1
        arrays={'condition':condition,'target':np.where(mask,target,0.).astype('float32'),'mask':mask,
            'baseline':baseline,'residual_scale':np.ones((n,1,a.variables),'float32'),
            'obs_mean':np.zeros(target.shape,'float32'),'obs_std':np.ones(target.shape,'float32')}
        for key,value in arrays.items(): np.save(out/(key+'.npy'),value)
        dates=np.datetime64(f'{year}-04-01')+np.arange(n).astype('timedelta64[D]')
        np.savez_compressed(out/'index.npz',initialization=dates,valid_time=dates+np.timedelta64(48,'h'),lead_hours=np.full(n,48))
        if split=='test':
            raw=rng.normal(size=(n,51,a.stations,a.variables)).astype('float32')
            if a.variables==2: raw[...,1]=np.abs(raw[...,1])
            np.save(out/'raw.npy',raw)
    meta={'complete':True,'synthetic':True,'development_only':True,'stations':a.stations,
        'variables':['t2m','p10fg6'][:a.variables],'features':10,'station_ids':list(range(a.stations)),
        'latitude':np.linspace(46,53,a.stations).tolist(),'longitude':np.linspace(3,10,a.stations).tolist(),
        'leads':[48],'note':'Synthetic smoke check, not weather benchmark results'}
    meta['files_sha256']={str(f.relative_to(root)):digest(f) for f in root.rglob('*') if f.is_file()}
    save_json(root/'metadata.json',meta)
    print(f'Created synthetic wiring check: {root.resolve()}')


if __name__=='__main__': main()

