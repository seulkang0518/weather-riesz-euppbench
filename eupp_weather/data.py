"""Read-only memory-mapped prepared cases; each case is one initialization/lead."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import read_json


ARRAYS = ('condition', 'target', 'mask', 'baseline', 'residual_scale', 'obs_mean', 'obs_std')


class Cases(Dataset):
    def __init__(self, directory, split):
        self.root = Path(directory)
        self.metadata = read_json(self.root/'metadata.json')
        if not self.metadata.get('complete'):
            raise ValueError('Prepared data is incomplete')
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown split')
        self.split = split
        self.arrays = {key: np.load(self.root/split/(key+'.npy'), mmap_mode='r', allow_pickle=False) for key in ARRAYS}
        self.index = np.load(self.root/split/'index.npz', allow_pickle=False)
        self.raw = np.load(self.root/split/'raw.npy', mmap_mode='r', allow_pickle=False) if split == 'test' else None
        if len({len(a) for a in self.arrays.values()}) != 1:
            raise ValueError('Prepared array lengths differ')

    def __len__(self):
        return len(self.arrays['target'])

    def __getitem__(self, index):
        batch = {key: torch.from_numpy(np.array(value[index], copy=True)) for key, value in self.arrays.items()}
        batch['index'] = torch.tensor(index)
        return batch


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

