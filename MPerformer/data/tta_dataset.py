# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils

class TTADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, target, atom_H_num, bond, conf_size, noise_weight, noise_radio, noise_valid):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.target = target
        self.atom_H_num = atom_H_num
        self.bond = bond
        self.conf_size = conf_size
        self.noise_weight = noise_weight
        self.noise_radio = noise_radio
        self.noise_valid = noise_valid
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[smi_idx][self.atoms])
        
        noise_coord = np.random.randn(self.dataset[smi_idx][self.coordinates][coord_idx].shape[0],3) * self.noise_weight
        noise_prob = np.random.rand()
        
        if noise_prob > 0.2 and self.noise_valid == 1:
            coordinates = self.dataset[smi_idx][self.coordinates][coord_idx] + noise_coord
        else:       
            coordinates = self.dataset[smi_idx][self.coordinates][coord_idx]
        
        target = self.dataset[smi_idx][self.target]
        atom_H_num = self.dataset[smi_idx][self.atom_H_num]
        bond = self.dataset[smi_idx][self.bond]
        
        target_coordinates = self.dataset[smi_idx][self.coordinates][coord_idx]
        
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "target": target,
            "atom_H_num":atom_H_num,
            "bond": bond,
            "target_coordinates": target_coordinates.astype(np.float32)
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class TTADockingPoseDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        holo_coordinates,
        holo_pocket_coordinates,
        is_train=True,
        conf_size=10,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.holo_coordinates = holo_coordinates
        self.holo_pocket_coordinates = holo_pocket_coordinates
        self.is_train = is_train
        self.conf_size = conf_size
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[smi_idx][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
        pocket_atoms = np.array(
            [item[0] for item in self.dataset[smi_idx][self.pocket_atoms]]
        )
        pocket_coordinates = np.array(self.dataset[smi_idx][self.pocket_coordinates][0])
        if self.is_train:
            holo_coordinates = np.array(self.dataset[smi_idx][self.holo_coordinates][0])
            holo_pocket_coordinates = np.array(
                self.dataset[smi_idx][self.holo_pocket_coordinates][0]
            )
        else:
            holo_coordinates = coordinates
            holo_pocket_coordinates = pocket_coordinates

        smi = self.dataset[smi_idx]["smi"]
        pocket = self.dataset[smi_idx]["pocket"]

        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_coordinates": holo_coordinates.astype(np.float32),
            "holo_pocket_coordinates": holo_pocket_coordinates.astype(np.float32),
            "smi": smi,
            "pocket": pocket,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
