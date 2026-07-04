import os
from typing import Callable, List, Optional
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import one_hot, scatter
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm

class SDBS(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["sdbsno.csv"]
    
    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        df = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]), dtype=str)
        # types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        atom_vocab = [1,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40,
        41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60,
        62, 63, 64, 65, 66, 67, 68, 70, 72, 73, 74, 75, 78, 79, 80, 81, 82, 83,
        90, 92]
        atom2id = {a:i for i,a in enumerate(atom_vocab)}
        data_list = []
        # for idx, row in df.iterrows():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing SDBS"):
            # print(idx, row['sdbsno'])
            sdbsno = row['sdbsno'].strip()
            mol = Chem.SDMolSupplier(self.raw_dir + f"/sdf/{sdbsno}.sdf", removeHs=False)[0]
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            N = mol.GetNumAtoms()
            # ===== 1. 原子特征 =====
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            
            z = torch.tensor(atomic_number, dtype=torch.long)
            x = torch.tensor([atom2id[a] for a in atomic_number], dtype=torch.long)

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

            # x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            # x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
            #                   dtype=torch.float).t().contiguous()
            # x = torch.cat([x1, x2], dim=-1)
            

            # ===== 3. 原子 3D 坐标 =====
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            
            # atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            # # channels = [self.elements_hash[atom] for atom in atoms]
            # radius = torch.tensor([pt.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()], dtype=torch.float)

            # ===== 6. PyG Data =====
            data = Data(
                x=x,
                z=z,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                smiles=smiles,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
    
    def get(self, idx):
        data = super().get(idx).clone()
        data.data_idx = torch.tensor([idx], dtype=torch.long)
        return data




