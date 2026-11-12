import numpy as np
import torch
from eupp_weather.model import StationTransformer,predict
from eupp_weather.metrics import local_variogram,nearest_neighbors,score_arrays


def test_euler_uses_exactly_requested_calls_and_reconstructs_scaled_forecast():
    class Constant(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls=0
        def forward(self,state,condition,time): self.calls+=1; return torch.full_like(state,2.)
    model=Constant(); noise=torch.randn(2,3,6,1)
    batch={'condition':torch.zeros(2,6,3),'baseline':torch.ones(2,6,1)*5,
           'residual_scale':torch.ones(2,1,1)*3}
    value=predict(model,batch,noise,'flow_matching',steps=8)
    assert model.calls==8
    torch.testing.assert_close(value,5+3*(noise+2))
    model.calls=0; value=predict(model,batch,noise,'energy')
    assert model.calls==1
    torch.testing.assert_close(value,torch.full_like(value,11.))


def test_station_attention_does_not_mix_forecast_cases():
    torch.manual_seed(5)
    model=StationTransformer(7,2,4,width=16,depth=2,heads=4,dropout=0.).eval()
    x,c=torch.randn(2,7,2),torch.randn(2,7,4); time=torch.zeros(2)
    first=model(x,c,time)[0]
    c[1]*=100; x[1]*=100
    torch.testing.assert_close(first,model(x,c,time)[0])


def test_joint_lvs_matches_explicit_ordered_pairs_and_constant_shift_invariance():
    rng=np.random.default_rng(22)
    p=rng.normal(size=(2,4,7,2)); y=rng.normal(size=(2,7,2)); mask=np.ones_like(y,bool)
    neighbors=nearest_neighbors(np.arange(7)*.1+49,np.arange(7)*.1+8)
    actual=local_variogram(p,y,mask,neighbors)
    expected=np.zeros_like(actual)
    for case in range(2):
        for center,row in enumerate(neighbors):
            coordinates=[(s,v) for s in row for v in range(2)]
            for s,v in coordinates:
                for r,w in coordinates:
                    observed=abs(y[case,s,v]-y[case,r,w])**.5
                    forecast=np.mean(abs(p[case,:,s,v]-p[case,:,r,w])**.5)
                    expected[case,center]+=(observed-forecast)**2
    np.testing.assert_allclose(actual,expected)
    np.testing.assert_allclose(local_variogram(p+10,y+10,mask,neighbors),actual)
    scores=score_arrays(p,y,mask,neighbors)
    brute=(np.abs(p-y[:,None]).mean(1)-.5*np.abs(p[:,:,None]-p[:,None]).mean((1,2))).mean((1,2))
    np.testing.assert_allclose(scores['mean_crps'],brute)

