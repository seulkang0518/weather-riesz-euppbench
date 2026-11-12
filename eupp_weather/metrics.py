"""Empirical scores, masked by observed coordinates and their available pairs."""
import numpy as np
import torch
from .objectives import energy_score


def nearest_neighbors(latitude, longitude, k=5):
    lat, lon = np.deg2rad(latitude), np.deg2rad(longitude)
    if len(lat) <= k: raise ValueError('Need more stations than neighbors')
    cosine = np.sin(lat[:,None])*np.sin(lat[None]) + np.cos(lat[:,None])*np.cos(lat[None])*np.cos(lon[:,None]-lon[None])
    distances = np.arccos(np.clip(cosine, -1, 1))
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind='stable')[:, :k]


def local_variogram(pred, target, mask, neighbors):
    """Return [B,center]. Sum ordered valid pairs, then average eligible centers.

    For missing observations, only valid station-variable pairs contribute.
    No division by the number of pairs, matching the source's sum convention.
    """
    b, m, s, v = pred.shape
    outputs = []
    for row in neighbors:
        p, y, valid = pred[:,:,row].reshape(b,m,-1), target[:,row].reshape(b,-1), mask[:,row].reshape(b,-1)
        pair_mask = valid[:,:,None] & valid[:,None,:]
        expected = np.sqrt(np.abs(p[:,:,:,None]-p[:,:,None,:])).mean(1)
        observed = np.sqrt(np.abs(y[:,:,None]-y[:,None,:]))
        value = np.where(pair_mask, (observed-expected)**2, 0.).sum((1,2))
        outputs.append(np.where(valid.sum(1)>=2, value, np.nan))
    return np.stack(outputs, 1)


def score_arrays(pred, target, mask, neighbors):
    """Per-case joint ES, coordinate CRPS, full VS, and local VS."""
    p, y = np.asarray(pred), np.asarray(target)
    mask = np.asarray(mask, dtype=bool)
    if p.ndim!=4 or not np.isfinite(p).all(): raise ValueError('Predictions must be finite [B,M,S,V]')
    es = energy_score(torch.from_numpy(p.astype('float32')),torch.from_numpy(y.astype('float32')),
                      torch.from_numpy(mask)).numpy()
    attraction = np.abs(p-y[:,None]).mean(1)
    # Coordinate CRPS without constructing [B,M,M,S,V].
    ordered = np.sort(p, axis=1); m=p.shape[1]
    coefficient = (2*np.arange(1,m+1)-m-1).reshape(1,m,1,1)
    crps = attraction - (ordered*coefficient).sum(1)/m**2
    crps = np.where(mask, crps, np.nan)
    flatp, flaty, flatmask = p.reshape(len(p),m,-1), y.reshape(len(y),-1), mask.reshape(len(y),-1)
    fullvs = np.zeros(len(p))
    # Row-by-row to bound memory independently of station count/member count.
    for i in range(flaty.shape[1]):
        obs = np.sqrt(np.abs(flaty[:,i,None]-flaty))
        exp = np.sqrt(np.abs(flatp[:,:,i,None]-flatp)).mean(1)
        fullvs += np.where(flatmask[:,i,None]&flatmask,(obs-exp)**2,0.).sum(1)
    lvs = local_variogram(p,y,mask,neighbors)
    return {'energy_score':es,'mean_crps':np.nanmean(crps,axis=(1,2)),
            'variogram_score':fullvs,'local_variogram_score':np.nanmean(lvs,axis=1)}

