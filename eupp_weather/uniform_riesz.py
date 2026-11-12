"""Weather-only uniform-neighbour ablation of the unchanged upstream Riesz loss."""
import math
import torch


def uniform_toward(source,target,*,exclude_self=False,direction_epsilon=1e-8):
    count=target.shape[1]-int(exclude_self)
    if count<1:
        return torch.zeros_like(source)
    distance=torch.cdist(source,target)
    weight=torch.full_like(distance,1.0/count)
    if exclude_self:
        if source.shape[1]!=target.shape[1]: raise ValueError('Self-exclusion requires equal counts')
        diagonal=torch.eye(source.shape[1],device=source.device,dtype=torch.bool)[None]
        weight=weight.masked_fill(diagonal,0.)
    coefficient=weight/distance.clamp_min(direction_epsilon)
    return torch.bmm(coefficient,target)-source*coefficient.sum(-1,keepdim=True)


def uniform_riesz_loss(pred,target,temperatures):
    """Only replace softmax weights; retain distance scale, unit vectors and RMS floors.

    pred [B,M,D], target [B,D]. No CFG or fixed negatives in this weather task.
    The same temperature prefactors and three normalized contributions are kept,
    including finite RMS floors; this avoids changing field count/update scale.
    """
    temperatures=tuple(float(t) for t in temperatures)
    if not temperatures or any(not math.isfinite(t) or t<=0 for t in temperatures):
        raise ValueError('Require positive finite temperatures')
    if pred.ndim!=3 or target.shape!=(pred.shape[0],pred.shape[2]) or pred.shape[1]<2:
        raise ValueError('Require predictions [B,M>=2,D] and one target [B,D]')
    gen=pred.float()
    with torch.no_grad():
        old=gen.detach(); pos=target.detach().float()[:,None]
        scale=torch.cdist(old,torch.cat([old,pos],1)).mean()/(1.0+1e-6)
        coordinate_scale=(scale/math.sqrt(gen.shape[-1])).clamp_min(1e-3)
        old_scaled=old/coordinate_scale; pos_scaled=pos/coordinate_scale
        positive=uniform_toward(old_scaled,pos_scaled)
        negative=uniform_toward(old_scaled,old_scaled,exclude_self=True)
        attraction=torch.zeros_like(old_scaled); repulsion=torch.zeros_like(old_scaled)
        for tau in temperatures:
            temp_attraction=positive/tau; temp_repulsion=-negative/tau
            field=temp_attraction+temp_repulsion
            magnitude=field.square().mean().clamp_min(1e-8).sqrt()
            attraction.add_(temp_attraction/magnitude)
            repulsion.add_(temp_repulsion/magnitude)
        velocity=attraction+repulsion
        goal=(old_scaled+velocity).detach()
    loss=(gen/coordinate_scale-goal).square().mean((-1,-2))
    return loss,{'scale':scale.detach(),'uniform_velocity_rms':velocity.square().mean().sqrt(),
                 'uniform_generated_ess':gen.new_tensor(float(gen.shape[1]-1))}
