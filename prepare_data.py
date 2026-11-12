"""Prepare station-only EUPPBench cases; fit every statistic on training years only."""
import argparse
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap
from eupp_weather.common import digest, save_json, disjoint_output
from eupp_weather.preprocessing import (COUNTRIES, VALIDATION_YEARS, open_stations, case_dataset,
    day_of_year, split_masks, seasonal_stats, error_scales)


def moments(ds, name):
    # Second moments in fp64 avoid catastrophic cancellation for pressure/
    # geopotential fields with a large offset and small ensemble spread.
    x = ds[name].astype('float64')
    finite_fraction=x.notnull().mean('number')
    if name == 'p10fg6': x = np.log1p(x)
    if name == 'cin': x = np.log1p(x.fillna(0))
    if name == 'q_700_hPa': x = x.fillna(0)
    # Compute jointly to avoid reading/decompressing the same Zarr chunks twice.
    import xarray as xr
    result = xr.Dataset({'first': x.mean('number', skipna=True),
                         'second': (x*x).mean('number', skipna=True),
                         'fraction': finite_fraction}).compute()
    return tuple(result[k].transpose('case', 'station_id').values
                 for k in ('first', 'second', 'fraction'))


def observation(ds, variables):
    result = []
    for name in variables:
        value = ds[name].transpose('case', 'station_id').values.astype('float32')
        result.append(np.log1p(value) if name == 'p10fg6' else value)
    return np.stack(result, -1)


def lookup(stat, dates, lead_idx):
    return stat[day_of_year(dates)-1, lead_idx]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', default='data/raw')
    p.add_argument('--output', default='data/temperature48')
    p.add_argument('--profile', choices=['temperature48', 'full'], default='temperature48')
    p.add_argument('--leads', nargs='+', type=int, help='Defaults: 48 for pilot; 6..120 for full')
    p.add_argument('--countries', nargs='+', choices=COUNTRIES, default=list(COUNTRIES))
    p.add_argument('--allow-station-subset', action='store_true', help='Development only, never label a full benchmark')
    a = p.parse_args(); root, raw = Path(a.output).resolve(), Path(a.raw).resolve()
    disjoint_output(root, [raw])
    if root.exists(): raise FileExistsError('Choose a new output directory; prepared data is immutable')
    leads = a.leads or ([48] if a.profile == 'temperature48' else list(range(6, 121, 6)))
    if len(set(leads)) != len(leads) or not all(h in range(6, 121, 6) for h in leads):
        raise ValueError('Use distinct 6-hour leads from 6 to 120')
    variables = ['t2m'] if a.profile == 'temperature48' else ['t2m', 'p10fg6']
    types = ['surface'] if a.profile == 'temperature48' else ['surface', 'surface_postprocessed', 'pressure_500', 'pressure_700', 'pressure_850']
    obs_types = ['surface'] if a.profile == 'temperature48' else ['surface', 'surface_postprocessed']
    records = {}
    for label, kind, obs_kind in [('reforecast', 'reforecast', 'reforecast_obs'), ('forecast', 'forecast', 'forecast_obs')]:
        fc = case_dataset(open_stations(raw, kind, types, a.countries), leads)
        obs = case_dataset(open_stations(raw, obs_kind, obs_types, a.countries), leads)
        # Station and time/lead alignment is exact, never implicit nearest matching.
        obs = obs.sel(station_id=fc.station_id)
        for coord in ('station_id', 'time', 'step'):
            if not np.array_equal(fc[coord].values, obs[coord].values):
                raise ValueError(f'Forecast/observation {coord} mismatch')
        records[label] = (fc, obs)
    ref, obs = records['reforecast']
    test_fc, test_obs = records['forecast']
    stations = ref.sizes['station_id']
    if (stations != 122 or len(a.countries) != 5) and not a.allow_station_subset:
        raise ValueError(f'Expected 122 stations from all five countries, got {stations}')
    if not np.array_equal(ref.station_id.values, test_fc.station_id.values):
        raise ValueError('Operational and reforecast station order differs')
    root.mkdir(parents=True)
    receipt_hashes = {f.name: digest(f) for f in raw.glob('download_*.json')}
    metadata = {'complete': False, 'profile': a.profile, 'variables': variables,
        'stations': stations, 'station_ids': ref.station_id.values.tolist(),
        'latitude': ref.station_latitude.values.tolist(), 'longitude': ref.station_longitude.values.tolist(),
        'leads': leads, 'validation_years': list(VALIDATION_YEARS), 'test_from': '2017-04-01',
        'test_until': '2018-12-31', 'source_receipt_sha256': receipt_hashes,
        'development_only': a.allow_station_subset, 'synthetic': False,
        'protocol': 'Independent adaptation; train-only normalization; see audit/README.txt'}
    save_json(root/'metadata.json', metadata)
    dates = ref.time.values
    hours = ref.step.values.astype('timedelta64[h]').astype(int)
    lead_idx = np.array([leads.index(int(h)) for h in hours])
    split = split_masks(dates)
    target = observation(obs, variables)
    omean, ostd, sparse = seasonal_stats(target, dates, lead_idx, split['train'], len(leads))
    np.savez_compressed(root/'normalization.npz', obs_mean=omean, obs_std=ostd)
    means, stds = lookup(omean, dates, lead_idx), lookup(ostd, dates, lead_idx)
    target_norm = (target-means)/stds
    raw_mean, raw_second = {}, {}
    feature_names = (['t2m', 'u10', 'v10', 'tcc', 'tcwv'] if a.profile == 'temperature48' else
                     sorted(n for n in ref.data_vars if 'number' in ref[n].dims and n not in ('r_850_hPa', 'ssrd6')))
    for variable in variables:
        if variable not in feature_names: feature_names.append(variable)
    if not set(feature_names).issubset(test_fc.data_vars):
        raise ValueError('Operational/reforecast feature sets differ')
    # 3 values per predictor: standardized ensemble mean/std and finite fraction.
    # Static metadata and train-derived seasonal target mean/std are shared across methods.
    land = np.asarray(ref.station_land_usage.values)
    categories = sorted(np.unique(land).tolist())
    static = np.stack([ref.station_latitude.values/90, ref.station_longitude.values/180,
                       ref.station_altitude.values/1500], -1).astype('float32')
    land_features = np.stack([land == c for c in categories], -1).astype('float32')
    features = 3*len(feature_names) + 3 + len(categories) + 5 + 2*len(variables)
    allocations = {}
    for label, (fc, observations) in records.items():
        d = fc.time.values; h = fc.step.values.astype('timedelta64[h]').astype(int)
        li = np.array([leads.index(int(x)) for x in h])
        y = target if label == 'reforecast' else observation(observations, variables)
        ym, ys = lookup(omean, d, li), lookup(ostd, d, li)
        masks = split_masks(d, operational=label == 'forecast')
        for name, selected in masks.items():
            selected = selected & np.isfinite(y).any((1, 2))
            indices = np.flatnonzero(selected)
            if not len(indices): raise ValueError(f'No usable {name} cases')
            out = root/name; out.mkdir()
            fields = {'condition': (len(indices), stations, features),
                'target': (len(indices), stations, len(variables)), 'mask': (len(indices), stations, len(variables)),
                'baseline': (len(indices), stations, len(variables)), 'residual_scale': (len(indices), 1, len(variables)),
                'obs_mean': (len(indices), stations, len(variables)), 'obs_std': (len(indices), stations, len(variables))}
            arrays = {key: open_memmap(out/(key+'.npy'), mode='w+', dtype=bool if key=='mask' else 'float32', shape=shape)
                      for key, shape in fields.items()}
            arrays['mask'][:] = np.isfinite(y[indices])
            arrays['target'][:] = np.nan_to_num((y[indices]-ym[indices])/ys[indices])
            arrays['obs_mean'][:] = ym[indices]; arrays['obs_std'][:] = ys[indices]
            arrays['condition'][:] = 0.
            np.savez_compressed(out/'index.npz', initialization=d[indices], lead_hours=h[indices],
                                valid_time=d[indices]+fc.step.values[indices], source_index=indices)
            allocations[name] = (arrays, indices, label)
    normalization = {'obs_mean': omean, 'obs_std': ostd}
    stats_metadata = {'seasonal_windows_falling_back_to_training_global': sparse}
    for number, name in enumerate(feature_names):
        print(f'Predictor {number+1}/{len(feature_names)}: {name}', flush=True)
        fm, fsq, frac = moments(ref, name)
        if name in variables:
            cm, cs, fallback = seasonal_stats(fm[..., None], dates, lead_idx, split['train'], len(leads), second=fsq[..., None])
            cm, cs = cm[..., 0], cs[..., 0]
            train_mu, train_sd = lookup(cm, dates, lead_idx), lookup(cs, dates, lead_idx)
            stats_metadata[name+'_seasonal_fallbacks'] = fallback
        else:
            valid_hour = ((dates + ref.step.values).astype('datetime64[h]').astype(int) % 24)
            cm, cs = np.zeros((24, stations), 'float32'), np.ones((24, stations), 'float32')
            for hour in np.unique(valid_hour):
                take = split['train'] & (valid_hour == hour)
                mu = np.nanmean(fm[take], 0); sq = np.nanmean(fsq[take], 0)
                if not np.isfinite(mu).all(): raise ValueError(f'No training values for {name}, hour {hour}')
                cm[hour] = mu; cs[hour] = np.maximum(sq-mu**2, 1e-8)**.5
            train_mu, train_sd = cm[valid_hour], cs[valid_hour]
        normalization[name+'_mean'], normalization[name+'_std'] = cm, cs
        for label, (fc, _) in records.items():
            if label == 'reforecast': first, second, fraction, mu, sd = fm, fsq, frac, train_mu, train_sd
            else:
                first, second, fraction = moments(fc, name)
                d = fc.time.values; h = fc.step.values.astype('timedelta64[h]').astype(int)
                li = np.array([leads.index(int(x)) for x in h])
                if name in variables: mu, sd = lookup(cm, d, li), lookup(cs, d, li)
                else:
                    vh = ((d+fc.step.values).astype('datetime64[h]').astype(int) % 24)
                    mu, sd = cm[vh], cs[vh]
            ensemble_mean = np.nan_to_num((first-mu)/sd)
            correction=fc.sizes['number']/max(fc.sizes['number']-1,1)
            ensemble_std = np.nan_to_num((np.maximum(second-first**2, 0.)*correction)**.5/sd)
            for out_name, (arr, indices, origin) in allocations.items():
                if origin != label: continue
                arr['condition'][..., number*3:number*3+3] = np.stack([ensemble_mean[indices], ensemble_std[indices], fraction[indices]], -1)
                if name in variables: arr['baseline'][..., variables.index(name)] = ensemble_mean[indices]
    train_arrays, train_indices, _ = allocations['train']
    scales = error_scales(np.where(train_arrays['mask'], train_arrays['target'], np.nan),
        np.asarray(train_arrays['baseline']), hours[train_indices], np.ones(len(train_indices), dtype=bool))
    normalization['residual_scale_coefficients'] = scales
    normalization['land_categories'] = np.asarray(categories)
    for out_name, (arr, indices, origin) in allocations.items():
        fc, _ = records[origin]; d=fc.time.values[indices]; h=fc.step.values[indices].astype('timedelta64[h]').astype(int)
        valid_time = d+fc.step.values[indices]
        phase = day_of_year(valid_time)/366*2*np.pi
        hour_phase = valid_time.astype('datetime64[h]').astype(int)%24/24*2*np.pi
        seasonal = np.concatenate([arr['obs_mean'], arr['obs_std']], -1)
        # Climatology features use training-wide scaling, never test-set centering.
        if out_name == 'train':
            normalization['clim_feature_mean'] = seasonal.mean(0)
            normalization['clim_feature_std'] = seasonal.std(0).clip(1e-4)
        seasonal = (seasonal-normalization['clim_feature_mean'])/normalization['clim_feature_std']
        time_features=np.stack([np.sin(phase), np.cos(phase), np.sin(hour_phase), np.cos(hour_phase), h/120],-1)
        meta=np.concatenate([np.broadcast_to(static,(len(indices),)+static.shape),
            np.broadcast_to(land_features,(len(indices),)+land_features.shape),
            np.broadcast_to(time_features[:,None],(len(indices),stations,5)), seasonal],-1)
        arr['condition'][...,3*len(feature_names):]=meta
        arr['residual_scale'][:]=np.maximum(scales[0]+scales[1]*h[:,None]/24, .05)[:,None]
        if out_name == 'test':
            raw_members = []
            for name in variables:
                raw_members.append(fc[name].isel(case=indices).transpose('case','number','station_id').values.astype('float32'))
            np.save(root/'test/raw.npy', np.stack(raw_members, -1))
        for key, array in arr.items():
            if key not in ('target',) and not np.isfinite(array).all(): raise ValueError(f'Nonfinite prepared {out_name}/{key}')
            array.flush()
    np.savez_compressed(root/'normalization.npz', **normalization)
    metadata.update(complete=True, features=features, predictors=feature_names, climatology=stats_metadata,
                    counts={name: len(value[1]) for name,value in allocations.items()},
                    raw_members=int(test_fc.sizes['number']), files_sha256={})
    metadata['files_sha256']={str(f.relative_to(root)):digest(f) for f in root.rglob('*') if f.is_file() and f.name!='metadata.json'}
    save_json(root/'metadata.json', metadata)
    print(f'Prepared {metadata["counts"]}, {stations} stations, variables {variables}.', flush=True)


if __name__ == '__main__': main()
