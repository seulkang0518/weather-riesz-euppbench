"""Independent station-token Transformer shared by all experimental objectives."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class StationTransformer(nn.Module):
    def __init__(self, stations, variables, features, width=256, depth=4, heads=4,
                 dropout=.1, activation_checkpointing=False):
        super().__init__()
        self.stations, self.variables = stations, variables
        self.activation_checkpointing = activation_checkpointing
        self.input = nn.Sequential(nn.Linear(features + variables + 1, width), nn.SiLU())
        self.station = nn.Parameter(torch.randn(1, stations, width) * .02)
        self.register = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(width, heads,
            dim_feedforward=width, dropout=dropout, activation=nn.functional.silu,
            batch_first=True, norm_first=True) for _ in range(depth)])
        self.output = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, variables))

    def forward(self, state, condition, time):
        if state.shape != (condition.shape[0], self.stations, self.variables):
            raise ValueError('State/condition shape mismatch')
        t = time.reshape(-1, 1, 1).expand(-1, self.stations, 1)
        x = self.input(torch.cat([state, condition, t], -1)) + self.station
        x = torch.cat([x, self.register.expand(len(x), -1, -1)], 1)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.activation_checkpointing and self.training else block(x)
        return self.output(x[:, :self.stations])


def build_model(meta, config):
    return StationTransformer(meta['stations'], len(meta['variables']), meta['features'], **config)


def predict(model, batch, noise, method, steps=16):
    """noise[B,M,S,V]. steps is the EXACT number of network evaluations."""
    b, m, s, v = noise.shape
    state = noise.reshape(b * m, s, v)
    condition = batch['condition'][:, None].expand(-1, m, -1, -1).reshape(b * m, s, -1)
    if method == 'flow_matching':
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        for step in range(steps):
            time = state.new_full((b * m,), step / steps)
            state = state + model(state, condition, time) / steps
        residual = state.reshape(b, m, s, v)
    else:
        residual = model(state, condition, state.new_zeros(b * m)).reshape(b, m, s, v)
    return batch['baseline'][:, None] + batch['residual_scale'][:, None] * residual
