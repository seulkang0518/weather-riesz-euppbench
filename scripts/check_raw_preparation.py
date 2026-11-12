from pathlib import Path
import sys, json, subprocess, tempfile
import numpy as np
import xarray as xr
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo))
from eupp_weather.preprocessing import COUNTRIES
root=Path(tempfile.mkdtemp(prefix='eupp_raw_schema_',dir='/tmp')).resolve()
raw=root/'raw'; raw.mkdir(); rng=np.random.default_rng(612)
for country_index,country in enumerate(COUNTRIES):
    for operational in (False,True):
        dates=np.array(['2017-01-15','2017-04-15','2018-07-15','2018-10-15'],dtype='datetime64[ns]') if operational else np.array([f'2017-{m:02d}-15' for m in range(1,13)],dtype='datetime64[ns]')
        years=np.arange(1,21)
        station=np.arange(country_index*2,country_index*2+2)+10000
        for observations in (False,True):
            dims=['station_id','time']+([] if observations else ['number'])+([] if operational else ['year'])+['step','surface']
            coords={'station_id':station,'time':dates,'step':np.array([0,48,120],dtype='timedelta64[h]'),'surface':[0.]}
            if not observations: coords['number']=np.arange(51 if operational else 11)
            if not operational: coords['year']=years
            shape=tuple(len(coords[d]) for d in dims)
            y=280+rng.normal(size=shape).astype('float32')
            if observations and country_index==0:
                y[0]=np.nan  # One station has no observations at all.
            if observations and country_index==1 and not operational:
                # A second station has held-out labels but no training support.
                # Neither those validation labels nor the operational labels
                # may determine its eligibility or climatology.
                y[0,:,~np.isin(years+1996,(2003,2010,2016))]=np.nan
            variables={'t2m':(dims,y)}
            if not observations:
                for name in ['u10','v10','tcc','tcwv']: variables[name]=(dims,rng.normal(size=shape).astype('float32'))
                coords.update({'station_latitude':('station_id',np.arange(2)*.1+46+country_index),
                    'station_longitude':('station_id',np.arange(2)*.1+3+country_index),
                    'station_altitude':('station_id',np.array([100.,200.])),
                    'station_land_usage':('station_id',np.array([1,2]))})
            y.flat[0]=np.nan
            prefix='stations_'+('forecasts_observations' if operational else 'reforecasts_observations') if observations else 'stations_ensemble_'+('forecasts' if operational else 'reforecasts')
            xr.Dataset(variables,coords=coords).to_zarr(raw/f'{prefix}_surface_{country}.zarr',consolidated=True,zarr_version=2)
output=root/'prepared'
with (root/'prepare.log').open('w') as log:
    subprocess.run([sys.executable,'-u','prepare_data.py','--raw',str(raw),'--output',str(output),'--allow-station-subset'],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
meta=json.loads((output/'metadata.json').read_text())
assert meta['complete'] and meta['stations']==10 and meta['counts']=={'train':204,'val':36,'test':3}
assert meta['development_only'] and len(meta['files_sha256'])>15
assert meta['target_support']['supported_coordinates_per_lead']==[8]
assert {r['station_id'] for r in meta['target_support']['unsupported_coordinates']}=={10000,10002}
for split in ['train','val','test']:
    condition=np.load(output/split/'condition.npy')
    assert np.isfinite(condition).all()
    assert condition.shape[2]==meta['features']
    masks=np.load(output/split/'mask.npy')
    assert not masks[:,[0,2]].any()
    assert masks[:,1].any()
normalization=np.load(output/'normalization.npz')
assert not normalization['obs_training_supported'][:,[0,2]].any()
np.testing.assert_array_equal(normalization['obs_mean'][:,:,[0,2]],0.)
np.testing.assert_array_equal(normalization['obs_std'][:,:,[0,2]],1.)
print('Preparation integration passed:',root)
(repo/'validation/raw_preparation_check.json').write_text(json.dumps({'development_only':True,
    'data':'synthetic raw Zarr stores shaped like the public archive; not real EUPPBench forecasts',
    'countries':5,'stations':10,'reforecast_years':[1997,2016],'counts':meta['counts'],
    'immutable_prepared_files':len(meta['files_sha256']),'normalization_training_only':True,
    'supported_target_coordinates':8,'unsupported_station_tokens_retained':2,
    'held_out_labels_cannot_restore_unsupported_training_coordinates':True},indent=2))
