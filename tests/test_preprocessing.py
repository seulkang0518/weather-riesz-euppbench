import numpy as np
import pytest
import xarray as xr
from eupp_weather.preprocessing import seasonal_stats,split_masks,normalized_store,case_dataset,error_scales,inverse_targets


def test_climatology_ignores_poisoned_validation_values_and_error_scale_uses_training_only():
    dates=np.arange('1999-01-01','2001-01-01',dtype='datetime64[D]')
    dates=np.concatenate([dates,np.array(['2003-01-01'],dtype='datetime64[D]')])
    train=split_masks(dates)['train']; li=np.zeros(len(dates),int)
    rng=np.random.default_rng(1); y=rng.normal(size=(len(dates),6,2))
    mean,std,_=seasonal_stats(y,dates,li,train,1)
    modified=y.copy(); modified[~train]=1e9
    actual_mean,actual_std,_=seasonal_stats(modified,dates,li,train,1)
    np.testing.assert_array_equal(mean,actual_mean); np.testing.assert_array_equal(std,actual_std)
    coefficients=error_scales(y,np.zeros_like(y),np.full(len(y),48),train)
    np.testing.assert_allclose(coefficients,error_scales(modified,np.zeros_like(y),np.full(len(y),48),train))


def test_actual_reforecast_year_encoding_and_exact_lead_selection(tmp_path):
    # Same dimensions and relative-year convention as the public archive metadata.
    coords={'station_id':[101,102],'number':[0,1],'time':np.array(['2017-01-02','2017-06-05'],dtype='datetime64[ns]'),
            'year':np.arange(1,21),'step':np.array([0,48,120],dtype='timedelta64[h]'),'surface':[0.]}
    ds=xr.Dataset({'t2m':(('station_id','time','number','year','step','surface'),
        np.ones((2,2,2,20,3,1),'float32'))},coords=coords)
    path=tmp_path/'sample.zarr'; ds.to_zarr(path,consolidated=True,zarr_version=2)
    restored=normalized_store(path)
    years=restored.time.values.astype('datetime64[Y]').astype(int)+1970
    assert years.min()==1997 and years.max()==2016 and len(years)==40
    selected=case_dataset(restored,[48])
    assert selected.sizes['case']==40
    assert np.array_equal(selected.step.values,np.full(40,np.timedelta64(48,'h')))
    assert split_masks(selected.time.values)['val'].sum()==6


def test_test_cutoff_and_inverse_wind_transform():
    d=np.array(['2017-03-31','2017-04-01','2018-12-31','2019-01-01'],dtype='datetime64[D]')
    np.testing.assert_array_equal(split_masks(d,operational=True)['test'],[False,True,True,False])
    normalized=np.array([[[0.,1.]]]); mean=np.array([[[280.,0.]]]); std=np.array([[[2.,1.]]])
    recovered=inverse_targets(normalized,mean,std,['t2m','p10fg6'])
    np.testing.assert_allclose(recovered,[[[280.,np.expm1(1.)]]])


def test_large_offset_predictor_preserves_small_ensemble_variance():
    from prepare_data import moments
    values=np.array([99999.,100001.],dtype='float32').reshape(1,2,1)
    ds=xr.Dataset({'pressure':(('case','number','station_id'),values)})
    first,second,fraction=moments(ds,'pressure')
    np.testing.assert_allclose(first,100000.)
    np.testing.assert_allclose(second-first**2,1.)
    np.testing.assert_allclose(fraction,1.)
