# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class RemoveHydrogenDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        target,
        atom_H_num,
        bond,
        target_coordinates,
        remove_hydrogen=False,
        remove_polar_hydrogen=False,
        split="train",
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.target = target
        self.atom_H_num = atom_H_num
        self.bond = bond
        self.target_coordinates = target_coordinates
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen
        self.split = split
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]
        target = dd[self.target]
        atom_H_num = dd[self.atom_H_num]
        bond = dd[self.bond]
        target_coordinates = dd[self.target_coordinates]

        if self.remove_hydrogen:
            mask_hydrogen = (atoms != "H")
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]
            target = target[mask_hydrogen]
            atom_H_num = atom_H_num[mask_hydrogen]

            try:
                bond = bond[mask_hydrogen][:,mask_hydrogen]             
            except:
                pass
            
            target_coordinates = target_coordinates[mask_hydrogen]
            
        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]
                target = target[:-end_idx]
                atom_H_num = atom_H_num[:-end_idx]
                bond = bond[:-end_idx][:,:-end_idx]
                target_coordinates = target_coordinates[:-end_idx]
                
        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        dd[self.target] = target.astype(np.int) 
        dd[self.atom_H_num] = atom_H_num.astype(np.int) 
        dd[self.bond] = bond.astype(np.int)
        dd[self.target_coordinates] = target_coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class RemoveHydrogenResiduePocketDataset(BaseWrapperDataset):
    def __init__(self, dataset, atoms, residues, coordinates, remove_hydrogen=True):
        self.dataset = dataset
        self.atoms = atoms
        self.residues = residues
        self.coordinates = coordinates
        self.remove_hydrogen = remove_hydrogen
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        residues = dd[self.residues]
        coordinates = dd[self.coordinates]
        if len(atoms) != len(residues):
            min_len = min(len(atoms), len(residues))
            atoms = atoms[:min_len]
            residues = residues[:min_len]
            coordinates = coordinates[:min_len, :]

        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            residues = residues[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]

        dd[self.atoms] = atoms
        dd[self.residues] = residues
        dd[self.coordinates] = coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class RemoveHydrogenPocketDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        holo_coordinates,
        remove_hydrogen=True,
        remove_polar_hydrogen=False,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.holo_coordinates = holo_coordinates
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]
        holo_coordinates = dd[self.holo_coordinates]

        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]
            holo_coordinates = holo_coordinates[mask_hydrogen]
        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]
                holo_coordinates = holo_coordinates[:-end_idx]
        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        dd[self.holo_coordinates] = holo_coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
