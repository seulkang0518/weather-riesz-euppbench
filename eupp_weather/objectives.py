"""Conditional scores/fields: interaction sets never mix forecast cases."""
import os
os.environ.setdefault('DRIFT_COMPILE', '0')
import torch
from .riesz_topk_softmax import riesz_topk_softmax_loss
from .uniform_riesz import uniform_riesz_loss


def observed(pred, target, mask):
    if pred.ndim != 4 or target.shape != pred.shape[:1] + pred.shape[2:] or mask.shape != target.shape:
        raise ValueError('Require pred[B,M,S,V], target/mask[B,S,V]')
    if not mask.flatten(1).any(1).all():
        raise ValueError('Cases with no observed coordinates must be excluded')
    # Missing observations affect losses only, never the generator conditioning.
    p = torch.where(mask[:, None], pred.float(), 0.).flatten(2)
    y = torch.where(mask, target.float(), 0.).flatten(1)
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite observed predictions/targets')
    return p, y


def energy_score(pred, target, mask, *, estimator='empirical'):
    p, y = observed(pred, target, mask)
    m = p.shape[1]
    attraction = torch.linalg.vector_norm(p - y[:, None], dim=-1).mean(1)
    pair = torch.cdist(p, p)
    if estimator == 'empirical':
        repulsion = pair.mean((1, 2))
    elif estimator == 'unbiased' and m >= 2:
        repulsion = pair.sum((1, 2)) / (m * (m - 1))
    else:
        raise ValueError('Use empirical, or unbiased with at least two members')
    return attraction - .5 * repulsion


def distribution_loss(pred, target, mask, method, temperatures, estimator='empirical'):
    if method == 'energy':
        return energy_score(pred, target, mask, estimator=estimator).mean(), {}
    p, y = observed(pred, target, mask)
    if method == 'riesz_uniform':
        loss, diagnostics = uniform_riesz_loss(p, y, temperatures)
    elif method == 'riesz_softmax':
        loss, diagnostics = riesz_topk_softmax_loss(p, y[:, None], tau_list=temperatures,
            normalize_per_temperature=True, epsilon=1e-6, direction_epsilon=1e-8)
    else:
        raise ValueError('Unknown distributional objective')
    return loss.mean(), diagnostics


def flow_pair(target_residual, noise, time, sigma_min=1e-5):
    s = time.reshape(-1, 1, 1)
    return (1 - (1 - sigma_min) * s) * noise + s * target_residual, target_residual - (1 - sigma_min) * noise


def masked_mse(pred, target, mask):
    value = torch.where(mask, (pred - target).square(), 0.)
    return (value.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)).mean()

