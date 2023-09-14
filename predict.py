# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import rdkit
import os
import re
import lmdb
import shutil
from multiprocessing import Pool
from tqdm import tqdm, trange
import sys
import shlex
import pickle
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw   
from rdkit.Chem import AllChem
from rdkit import Geometry
import dpdata
import time


def check_hydrogen_and_origin(mol):
    has_hydrogen = False
    hydrogen_at_origin = False
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            has_hydrogen = True
            position = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            x, y, z = position
            if (x, y, z) == (0.0000, 0.0000, 0.0000):
                hydrogen_at_origin = True
                break
    
    return has_hydrogen, hydrogen_at_origin


def numpy_seed(seed, *addl_seeds):
    """
    Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward
    """
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def read_xyz(filename_path, skipH=True):
    """
    Read xyz data

    :param filename: filename of .xyz file
    :param skipH: Do not read H atoms
    :return: atomtypes, coordinates and title section
    """
    with open(filename_path,'r') as fin:
        natoms = int(fin.readline())
        title = fin.readline()[:-1]
        q=0
        qin = re.search("(?:CHARGE|CHG)=([-+]?\d*\.\d+|\d+|[-+]?\d)",title,re.IGNORECASE)
        if qin:
            q = float(qin.group(1))
        coords = []
        atomtypes = []
        for x in range(natoms):
            line = fin.readline().split()
            if (line[0].lower()=='h') and skipH: continue
            atomtypes.append(line[0])
            coords.append([float(line[1]),float(line[2]),float(line[3])])

    return(atomtypes, coords, q, title)


def create_sdfile(name, atomtypes, coords, row1, row2, bond, charge):
    """
    Creates string with SD info

    :param name: molecule name
    :param atomtypes: atomic types
    :param coords:  coordinates
    :return: molblock
    """

    ins = name + "\n"

    # comment block
    ins += "MPerformer generated sdf\n"
    ins += "\n"
    ins += "%3d%3d  0  0  0  0  0  0  0  0  1 V2000\n" % (len(atomtypes), len(row1))

    # atomb block
    for at, xyz in zip(atomtypes, coords):
        ins += "%10.4f%10.4f%10.4f %-2s 0  0  0  0  0\n" % (xyz[0], xyz[1], xyz[2], at)
        # ins += "%2s %12.4f %12.4f %12.4f  \n" % ( at,xyz[0], xyz[1], xyz[2])

    # bond block
    for index in range(len(row1)):
        ins += "%3d%3d%3d  0  0  0  0\n" % (row1[index], row2[index], bond[index])

    add_charge_num = 0
    charge_ins = ""
    for idx, charge_num in enumerate(charge):
        if charge_num != 0:
            charge_ins += f"  {idx+1}  {charge_num}"
            add_charge_num = add_charge_num + 1
    
    if add_charge_num != 0:
        charge_ins = f"M  CHG  {add_charge_num}" + charge_ins
        ins += charge_ins
        ins += "\n"
            
    ins += "M  END"
    return(ins)


def check_consist_valid(mol_file):
    mol_file= mol_file
    pred_atom_charge = mol_file["pred_atom_charge"].view(-1)
    target = mol_file["target"]
    
    masked_tokens0 = target.ne(padding_idx)
    sample_size0 = masked_tokens0.long().sum()  
    
    pred_atom_charge0 = pred_atom_charge[masked_tokens0]
    
    charge_list = []
    for i in range(pred_atom_charge0.shape[0]):
        if pred_atom_charge0[i] < -0.5:
            charge_list.append(-1)
            continue
        if pred_atom_charge0[i] > 0.5:
            charge_list.append(1)
            continue
        charge_list.append(0)
    
    
    pred_atom_H = mol_file["pred_atom_H"].view(-1)
    atom_H_target = mol_file["atom_H_target"]
    
    masked_atom_H_tokens = atom_H_target.ne(atom_H_pad_idx)
    pred_atom_H0 = pred_atom_H[masked_atom_H_tokens]
            
    H_list = []
    for i in range(pred_atom_H0.shape[0]):
        if pred_atom_H0[i] > 2.5:
            H_list.append(3)
            continue
        if pred_atom_H0[i] < 0.5:
            H_list.append(0)
            continue
        if pred_atom_H0[i] < 1.5:
            H_list.append(1)
            continue
        H_list.append(2)
    
    
    pred_atom_bond = mol_file["pred_atom_bond"][:, :, 0]
    bond_masked_tokens = masked_tokens0 

    masked_pred_atom_bond = pred_atom_bond[bond_masked_tokens]  
    masked_bond_target = mol_file['bond_target'][bond_masked_tokens]     
    
    non_pad_pos = (masked_bond_target >= 0) & (masked_bond_target != bond_pad_idx)   
    masked_bond_target = masked_bond_target[non_pad_pos]
    masked_pred_atom_bond = masked_pred_atom_bond[non_pad_pos]
    
    bond_list = []
    
    for i in range(masked_pred_atom_bond.shape[0]):
        if masked_pred_atom_bond[i] < 0.5:
            bond_list.append(0)
            continue
        if masked_pred_atom_bond[i] > 2.5:
            bond_list.append(3)
            continue
        if masked_pred_atom_bond[i] < 1.25:
            bond_list.append(1)
            continue
        if masked_pred_atom_bond[i] > 1.75:
            bond_list.append(2)
            continue
        bond_list.append(4)
    
    row1 = []
    row2 = []
    bond_type = []
    index = 0
    for i in range(sample_size0):
        for j in range(sample_size0):
            if j == i:
                continue
            if bond_list[index] != 0 and i < j:
                row1.append(i + 1)
                row2.append(j + 1)
                bond_type.append(bond_list[index])
            index = index + 1
    
    ins = create_sdfile(mol_file['id_name'],  mol_file['atoms'],  mol_file['coords'], row1, row2, bond_type, charge_list)

    filename = os.path.join(args.outputs_path, mol_file['id_name'] + '.sdf')

    with open(filename, 'w') as f:
        f.write(ins)
    
    try:
        suppl = Chem.SDMolSupplier(filename)
        mol = [mol for mol in suppl if mol][0]
    except:
        try:
            suppl = Chem.SDMolSupplier(filename, sanitize=False)
            mol = [mol for mol in suppl if mol][0]
            system = dpdata.BondOrderSystem(rdkit_mol=mol)
            Chem.Kekulize(mol)
        except:
            return None
        
    with Chem.SDWriter(filename) as w:
        w.write(mol)
        
        
if __name__ == "__main__":
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    
    # padding设置
    padding_idx = 0
    bond_pad_idx = 6
    atom_H_pad_idx = 6
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help=".xyz or dir for prediction")
    parser.add_argument('--outputs_path', help="model output path", default='outputs')  
    
    parser.add_argument('--data_path', help="your root path", default='./') 
    parser.add_argument('--inputs_path', help="model input path", default='inputs')  

    parser.add_argument('--seed', help="random seed", type=int, default=0)  
    parser.add_argument('--noise_weight', type=float, default=0.03)  
    parser.add_argument('--add_noise', action='store_true', help='whether add noise to coordinates\n', default=False)
    
    parser.add_argument('--cache_path', help="model cache path", default='cache')  
    parser.add_argument('--task_name', help="your data name", default='cache')
    parser.add_argument('--result', help="your dict_name", default='cache/weight_test_cpu.out.pkl')
    parser.add_argument('--weight_path', help="model output path", default='weight/checkpoint.pt')  

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--task_num', type=int, default=3)
    parser.add_argument('--loss_func', help="loss function", default='finetune_focal_loss_all_constraint_infer')
    parser.add_argument('--dict_name', help="your dict_name", default='dict.txt')
    parser.add_argument('--conf_size', type=int, default=1)
    parser.add_argument('--only_polar', type=int, default=0)
    parser.add_argument('--noise_valid', type=int, default=0)
    parser.add_argument('--noH', action='store_true', help='omit Hydrogen atom when learning\n',default=True)

    args = parser.parse_args()
    
    args.inputs_path = os.path.join(args.data_path, args.inputs_path)
    
    args.task_name = args.cache_path
    args.result = os.path.join(args.cache_path, 'weight_test_cpu.out.pkl')
    

    if os.path.exists(args.inputs_path):
        shutil.rmtree(args.inputs_path)
    os.mkdir(args.inputs_path)
    
    if os.path.exists(args.outputs_path):
        shutil.rmtree(args.outputs_path)
    os.mkdir(args.outputs_path)
    
    if os.path.exists(args.cache_path):
        shutil.rmtree(args.cache_path)
    os.mkdir(args.cache_path)
        
    if os.path.isdir(args.filename):
        args.inputs_path = args.filename
    else:
        os.system('cp ' + os.path.join(args.data_path, args.filename) + ' ' + args.inputs_path)
    
    outputfilename = os.path.join(args.cache_path, 'test.lmdb')
    env = lmdb.open(
        outputfilename,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(1000e9),
    )
    txn_writer = env.begin(write=True)

    numpy_seed(args.seed)
    
    for filename in os.listdir(args.inputs_path):
        
        if not filename.endswith('.xyz'):
            continue
        
        filename_path = os.path.join(args.inputs_path, filename)
        atomtypes, coords, q, title = read_xyz(filename_path, skipH=args.noH)

        coords = np.array(coords, dtype=np.float32)
        
        if args.add_noise:
            noise_coords = np.random.randn(coords.shape[0], 3) * args.noise_weight
            coords = coords + noise_coords
            
        inner_output = {}
        inner_output['data'] = filename.split('.')[0]        
        inner_output['atoms'] = atomtypes
        inner_output['coordinates'] = [coords] 
        
        inner_output['target'] = np.array([2] * len(atomtypes), dtype=np.float32)
        inner_output['atom_H_num'] = np.array([2] * len(atomtypes), dtype=np.float32)
        
        bond = np.array(list(np.zeros((len(atomtypes),len(atomtypes)), dtype=int)))
        for i in range(len(atomtypes)):
            bond[i][i] = -1
        
        inner_output['bond'] = bond 
        
        txn_writer.put(f"{inner_output['data']}".encode("ascii"), pickle.dumps(inner_output, protocol=-1))

    txn_writer.commit()
    env.close()
    
    t1 = time.time()
    
    cmd = "python ./MPerformer/infer.py --user-dir {} {} --task-name {} --valid-subset test --results-path {} --noise-valid {} --num-workers 1 --ddp-backend=c10d --batch-size {} --task MPerformer --loss {} --arch MPerformer_base --classification-head-name {} --num-classes {} --dict-name {} --conf-size {} --only-polar {} --path {} --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --log-interval 50 --log-format simple".format(os.path.join(args.data_path, 'MPerformer'), args.data_path, args.task_name, args.cache_path, args.noise_valid, args.batch_size, args.loss_func, args.task_name, args.task_num, args.dict_name, args.conf_size, args.only_polar, args.weight_path)

    os.system(cmd)

    env = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=8,
            map_size=int(60000e9),
        )
    txn = env.begin()

    predicts = pd.read_pickle(args.result)

    mol_file_list = []
    for epoch in range(len(predicts)):
        predict = predicts[epoch]
        bsz = predicts[epoch]['bsz']
        for index in range(bsz):
            mol_file = {}
            mol_file['id_name'] = predict['id_name'][index]
            value = pickle.loads(txn.get(mol_file['id_name'].encode()))
            mol_file['atoms'] = list(np.array(value['atoms']))
            mol_file['coords'] = list(np.array(value['coordinates'][0]))
            
            mol_file['logit_output'] = predict['logit_output'][index].float()   
            mol_file['target'] = predict['target'][index]
            mol_file['logit_atom_H_output'] = predict['logit_atom_H_output'][index].float()            
            mol_file['atom_H_target'] = predict['atom_H_target'][index]
            mol_file['logit_bond_output'] = predict['logit_bond_output'][index].float()
            mol_file['bond_target'] = predict['bond_target'][index]

            mol_file['pred_atom_charge'] = predict['pred_atom_charge'][index].float()         
            mol_file['pred_atom_H'] = predict['pred_atom_H'][index].float()
            mol_file['pred_atom_bond'] = predict['pred_atom_bond'][index].float()
         
            mol_file_list.append(mol_file)
    
    print(len(mol_file_list))
    
    with Pool() as pool:
        for inner_output in tqdm(pool.imap(check_consist_valid, mol_file_list), total=len(mol_file_list)):
            pass
    
    t2 = time.time()
    data_process_time = t2 - t1    
    
    print(f'input_xyz_file/fold: {args.filename}')
    print(f'output_sdf_fold: {args.outputs_path}')
    print(f'used_time: {data_process_time / 60} min')        
