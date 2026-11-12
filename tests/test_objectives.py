import pytest
import torch
from eupp_weather.objectives import energy_score, distribution_loss, flow_pair, masked_mse


def test_energy_formula_and_missing_coordinates_cannot_change_loss_or_gradient():
    torch.manual_seed(14)
    p=torch.randn(2,4,3,2,requires_grad=True); y=torch.randn(2,3,2)
    mask=torch.ones_like(y,dtype=torch.bool); mask[:,1,0]=False
    q=torch.where(mask[:,None],p,0.).flatten(2); target=torch.where(mask,y,0.).flatten(1)
    pair=torch.stack([torch.stack([torch.linalg.vector_norm(q[:,i]-q[:,j],dim=-1) for j in range(4)],1) for i in range(4)],1)
    a=torch.linalg.vector_norm(q-target[:,None],dim=-1).mean(1)
    for estimator,denom in [('empirical',16),('unbiased',12)]:
        actual=energy_score(p,y,mask,estimator=estimator)
        torch.testing.assert_close(actual,a-.5*pair.sum((1,2))/denom)
        grad=torch.autograd.grad(actual.sum(),p,retain_graph=True)[0]
        assert (grad[:,:,1,0]==0).all()
    changed=y.clone(); changed[:,1,0]=float('nan')
    torch.testing.assert_close(energy_score(p,y,mask),energy_score(p,changed,mask))


@pytest.mark.parametrize('method',['energy','riesz_softmax','riesz_uniform'])
def test_particle_permutation_stopped_target_and_masked_gradient(method):
    torch.manual_seed(15)
    p=torch.randn(2,5,4,2,requires_grad=True)
    y=torch.randn(2,4,2,requires_grad=True)
    mask=torch.ones_like(y,dtype=torch.bool); mask[:,0,1]=False
    loss,_=distribution_loss(p,y,mask,method,[.2,.05,.02])
    perm=torch.tensor([3,0,4,2,1])
    other,_=distribution_loss(p[:,perm],y,mask,method,[.2,.05,.02])
    torch.testing.assert_close(loss,other)
    gradient=torch.autograd.grad(loss,p,retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and (gradient[:,:,0,1]==0).all()
    if method!='energy':
        assert torch.autograd.grad(loss,y,allow_unused=True)[0] is None


def test_flow_velocity_is_the_derivative_and_masks_remove_missing_targets():
    a,z=torch.randn(3,4,2),torch.randn(3,4,2)
    t=torch.rand(3)
    state,v=flow_pair(a,z,t)
    state_next,_=flow_pair(a,z,t+1e-3)
    torch.testing.assert_close((state_next-state)/1e-3,v,rtol=3e-3,atol=3e-4)
    mask=torch.ones_like(a,dtype=torch.bool); mask[:,0]=False
    pred=v.clone().requires_grad_(); changed=v.clone(); changed[:,0]=1e6
    loss=masked_mse(pred,changed,mask); loss.backward()
    assert loss.item()==0. and (pred.grad==0).all()

