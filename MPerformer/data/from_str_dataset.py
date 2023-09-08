# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from functools import lru_cache
from unicore.data import UnicoreDataset
from unicore.data import BaseWrapperDataset

class FromStrLabelDataset(UnicoreDataset):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        return torch.tensor(list(map(float, samples)))
    
    
class FromNumpyDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return torch.from_numpy(self.dataset[idx])
