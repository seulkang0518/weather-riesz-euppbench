"""Independent EUPPBench station reader and training-only normalization."""
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr

COUNTRIES = ('austria', 'belgium', 'france', 'germany', 'netherlands')
VALIDATION_YEARS = (2003, 2010, 2016)


def normalized_store(path):
    """Restore relative reforecast years using the pinned authors' +1996 mapping."""
    try:
        ds = xr.open_zarr(path, consolidated=True)
    except (ValueError, KeyError):
        if Path(path).name!='stations_ensemble_forecasts_pressure_500_netherlands.zarr': raise
        # This release has malformed array-dimension metadata for this one store.
        # The upstream reader also repairs it; infer coordinates from its matching
        # surface store without editing any raw file.
        import zarr
        group=zarr.open_consolidated(str(path),mode='r')
        data=np.asarray(group['z']).squeeze()
        surface=normalized_store(Path(path).parent/'stations_ensemble_forecasts_surface_netherlands.zarr')
        dimensions=('station_id','number','time','step')
        expected=tuple(surface.sizes[d] for d in dimensions)
        if data.shape!=expected: raise ValueError(f'Unexpected Netherlands pressure shape {data.shape}, expected {expected}')
        return xr.Dataset({'z_500_hPa':(dimensions,data)},coords={d:surface[d] for d in dimensions})
    if not np.issubdtype(ds.step.dtype, np.timedelta64):
        ds = ds.assign_coords(step=ds.step.values.astype('timedelta64[h]'))
    for dimension in ('surface', 'depthBelowLandLayer', 'isobaricInhPa'):
        if dimension in ds.dims:
            if ds.sizes[dimension] != 1: raise ValueError(f'Unexpected {dimension} axis')
            if dimension == 'isobaricInhPa':
                level = int(ds[dimension].item())
                ds = ds.rename({name: f'{name}_{level}_hPa' for name in ds.data_vars})
            ds = ds.squeeze(dimension, drop=True)
    # Discard stale valid_time before rebuilding actual initialization dates.
    ds = ds.drop_vars('valid_time', errors='ignore')
    if 'year' in ds.dims:
        years = ds.year.values.astype(int)
        if np.all((years >= 1) & (years <= 20)):
            years = years + 1996
        elif not np.all((years >= 1900) & (years <= 2100)):
            raise ValueError(f'Unrecognized reforecast year encoding: {years}')
        base = ds.time.values.astype('datetime64[us]').astype(datetime)
        dates = np.array([time.replace(year=int(year)) for year in years for time in base], dtype='datetime64[ns]')
        ds = ds.stack(initialization=('year', 'time')).reset_index('initialization', drop=True)
        ds = ds.rename(initialization='time').assign_coords(time=dates)
    ds = ds.sortby('time')
    if len(np.unique(ds.time.values)) != ds.sizes['time']:
        raise ValueError('Duplicate reconstructed initialization times')
    return ds


def open_stations(directory, kind, types, countries=COUNTRIES):
    prefix = {'reforecast': 'stations_ensemble_reforecasts', 'forecast': 'stations_ensemble_forecasts',
              'reforecast_obs': 'stations_reforecasts_observations', 'forecast_obs': 'stations_forecasts_observations'}[kind]
    datasets = []
    for country in countries:
        parts = [normalized_store(Path(directory)/f'{prefix}_{typ}_{country}.zarr') for typ in types]
        # Merge only common lead times (postprocessed quantities have no step 0).
        aligned = xr.align(*parts, join='inner', exclude={'station_id'})
        country_data = xr.merge(aligned, compat='override')
        country_data = country_data.assign_coords(country=('station_id', [country]*country_data.sizes['station_id']))
        datasets.append(country_data)
    aligned = xr.align(*datasets, join='exact', exclude={'station_id'})
    result = xr.concat(aligned, 'station_id', data_vars='all', coords='minimal', compat='override')
    if len(np.unique(result.station_id.values)) != result.sizes['station_id']:
        raise ValueError('Duplicate station identities across countries')
    return result


def case_dataset(ds, leads):
    steps = ds.step.values.astype('timedelta64[h]').astype(int)
    if not set(leads).issubset(steps): raise ValueError(f'Requested leads absent: {leads}')
    ds = ds.sel(step=np.array(leads, dtype='timedelta64[h]'))
    return ds.stack(case=('time', 'step')).reset_index('case')


def day_of_year(dates):
    dates = np.asarray(dates, dtype='datetime64[D]')
    return (dates - dates.astype('datetime64[Y]')).astype(int) + 1


def split_masks(dates, *, operational=False):
    dates = np.asarray(dates, dtype='datetime64[D]')
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    if operational:
        return {'test': (dates >= np.datetime64('2017-04-01')) & (dates < np.datetime64('2019-01-01'))}
    validation = np.isin(years, VALIDATION_YEARS)
    return {'train': ~validation, 'val': validation}


def seasonal_stats(values, dates, lead_indices, train, n_leads, *, second=None, window=21):
    """Pooled 21-day, circular 366-day climatology; no validation/test fitting.

    second is E_member[x^2] for ensemble inputs, otherwise observations squared.
    Sparse seasonal windows fall back to the station/lead's training-wide stats.
    A coordinate with no training variance/observations is an explicit error.
    """
    x = np.asarray(values, dtype=np.float64)
    x2 = x*x if second is None else np.asarray(second, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(x2)
    idx = (day_of_year(dates)[train]-1, np.asarray(lead_indices)[train])
    shape = (366, n_leads) + x.shape[1:]
    count, total, square = [np.zeros(shape, dtype=np.float64) for _ in range(3)]
    np.add.at(count, idx, valid[train])
    np.add.at(total, idx, np.where(valid[train], x[train], 0.))
    np.add.at(square, idx, np.where(valid[train], x2[train], 0.))
    global_count = count.sum(0)
    if (global_count < 2).any(): raise ValueError('At least two training observations required per station/lead/variable')
    global_mean = total.sum(0)/global_count
    global_var = square.sum(0)/global_count-global_mean**2
    rolled = [sum(np.roll(a, shift, axis=0) for shift in range(-window//2+1, window//2+1))
              for a in (count, total, square)]
    n, sm, sq = rolled
    mean = sm/np.maximum(n, 1.)
    var = sq/np.maximum(n, 1.)-mean**2
    sparse = n < 2
    mean = np.where(sparse, global_mean[None], mean)
    var = np.where(sparse, global_var[None], var)
    if not np.isfinite(mean).all(): raise ValueError('Nonfinite climatology')
    std = np.maximum(var, 1e-8)**.5
    return mean.astype('float32'), std.astype('float32'), int(sparse.sum())


def target_transform(array, variable):
    if variable == 'p10fg6':
        if np.any(np.asarray(array) < 0): raise ValueError('Negative raw gust observation')
        return np.log1p(array)
    return array


def inverse_targets(value, mean, std, variables):
    out = np.asarray(value, dtype=np.float64)*std + mean
    for i, name in enumerate(variables):
        if name == 'p10fg6': out[..., i] = np.maximum(np.expm1(out[..., i]), 0.)
    return out


def error_scales(target, baseline, leads, train):
    """Positive linear lead-time fit to training residual RMS, shared across stations."""
    unique = np.unique(leads)
    v = target.shape[-1]
    observed = []
    for lead in unique:
        err = np.where(train[:, None, None] & (leads[:, None, None] == lead), target-baseline, np.nan)
        observed.append(np.sqrt(np.nanmean(err**2, axis=(0, 1))))
    observed = np.asarray(observed)
    if not np.isfinite(observed).all(): raise ValueError('Cannot fit lead scaling')
    design = np.stack([np.ones(len(unique)), unique/24], 1)
    coefficients = np.linalg.lstsq(design, observed, rcond=None)[0] if len(unique)>1 else np.stack([observed[0], np.zeros(v)])
    return coefficients.astype('float32')
