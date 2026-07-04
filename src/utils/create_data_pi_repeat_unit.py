import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from sentence_transformers import SentenceTransformer

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT

import pandas as pd
from tqdm import tqdm

def generate_pi_repeat_unit(diamine: Chem.Mol, dianhydride: Chem.Mol) -> Chem.Mol:

    
    # if not diamine or not dianhydride:
    #     raise ValueError("SMILES 解析失败")

    # ==========================================
    # 步骤 1: 二胺一端加 [1*]
    # ==========================================
    rxn1 = AllChem.ReactionFromSmarts('[#6:1]-[N;H2X3] >> [#6:1]-[1*]')
    prods1 = rxn1.RunReactants((diamine,))
    if not prods1:
        raise ValueError("二胺匹配失败")
    half_diamine = prods1[0][0]
    # 关键：这里只做基本的清理，不要让它乱动键级
    half_diamine.UpdatePropertyCache()

    # ==========================================
    # 步骤 2: 二酐一端转酰亚胺加 [2*]
    # ==========================================
    # 先强制 Kekulize，确保 SMARTS 能看清 C=O 双键
    Chem.Kekulize(dianhydride, clearAromaticFlags=True)
    rxn2_smarts = '[CX3:1](=[OX1:2])[OX2:3][CX3:4](=[OX1:5]) >> [CX3:1](=[OX1:2])[N:3](-[2*])[CX3:4](=[OX1:5])'
    rxn2 = AllChem.ReactionFromSmarts(rxn2_smarts)
    prods2 = rxn2.RunReactants((dianhydride,))
    if not prods2:
        raise ValueError("步骤 2 匹配失败")
    
    # PMDA 是对称的，我们要确保拿到的产物里【只有一个】[2*]
    # 有些反应可能会把两端都反应掉
    half_dianhydride = None
    for p_set in prods2:
        p = p_set[0]
        s_tmp = Chem.MolToSmiles(p)
        if s_tmp.count('2*') == 1 and s_tmp.count('=O') >= 4:
            half_dianhydride = p
            break
    
    if half_dianhydride is None:
        raise ValueError("无法生成合适的二酐中间体")

    # ==========================================
    # 步骤 3: 最终缩合 (解决失败的关键点)
    # ==========================================
    # 核心：在缩合前，对两个中间体再次强制 Kekulize！！
    Chem.Kekulize(half_diamine, clearAromaticFlags=True)
    Chem.Kekulize(half_dianhydride, clearAromaticFlags=True)

    # 使用更宽容的缩合 SMARTS
    # 我们匹配酸酐部分和伯胺部分
    rxn3_smarts = '[CX3:1](=[OX1:2])[OX2:3][CX3:4](=[OX1:5]).[NX3H2]-[#6:6] >> [CX3:1](=[OX1:2])[N:3](-[#6:6])[CX3:4](=[OX1:5])'
    rxn3 = AllChem.ReactionFromSmarts(rxn3_smarts)
    
    final_prods = rxn3.RunReactants((half_dianhydride, half_diamine))
    if not final_prods:
        # 如果再次失败，打印中间产物的 SMILES 进行调试
        print(f"Debug - Half Diamine: {Chem.MolToSmiles(half_diamine)}")
        print(f"Debug - Half Dianhydride: {Chem.MolToSmiles(half_dianhydride)}")
        raise RuntimeError("步骤 3 缩合失败：请检查中间体结构。")
    
    repeat_unit = final_prods[0][0]
    
    # 彻底清理并恢复芳香性
    Chem.SanitizeMol(repeat_unit)
    return repeat_unit

class PIRepeatUnit(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['diamine.csv', 'dianhydride.csv']
        

    @property
    def processed_file_names(self):
        return ["data_repeat_unit.pt"]

    def process(self):
        df_diamine = pd.read_csv(os.path.join(self.raw_dir, "monomer", self.raw_file_names[0]), dtype=str)  # 读取二胺
        df_dianhydride = pd.read_csv(os.path.join(self.raw_dir, "monomer", self.raw_file_names[1]), dtype=str)  # 读取二酐

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        atom_vocab = [0, 1,  6,  7,  8,  9, 16]
        atom2id = {a:i for i,a in enumerate(atom_vocab)}
        data_list = []
        polyBERT = SentenceTransformer('xushijie/polyBERT')
        for idx_1, row_1 in tqdm(df_diamine.iterrows()):
            
            id_diamine = row_1["id"]
            diamine_smiles = row_1["smiles"]

            for idx_2, row_2 in df_dianhydride.iterrows():
                id_dianhydride = row_2["id"]
                dianhydride_smiles = row_2["smiles"]

                # 1. 获取重复单元 mol
                diamine = Chem.MolFromSmiles(diamine_smiles)
                dianhydride = Chem.MolFromSmiles(dianhydride_smiles)
                repeat_unit_mol = generate_pi_repeat_unit(diamine, dianhydride)
                repeat_unit_mol = Chem.AddHs(repeat_unit_mol)
                smiles = Chem.MolToSmiles(repeat_unit_mol)
                smiles = smiles.replace("[1*]", "[*]").replace("[2*]", "[*]")
                embeddings = polyBERT.encode([smiles])
                # print(embeddings.shape)
                
                # 2. 原子信息
                atomic_number = []
                for atom in repeat_unit_mol.GetAtoms():
                    atomic_number.append(atom.GetAtomicNum())

                z = torch.tensor(atomic_number, dtype=torch.long)
                x = torch.tensor([atom2id[a] for a in atomic_number], dtype=torch.long)

                # 3. 键信息
                row, col, edge_type = [], [], []
                for bond in repeat_unit_mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_type += 2 * [bonds[bond.GetBondType()]]

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                
                name = f"{id_diamine}_{id_dianhydride}"

                # ===== Data =====
                data = Data(
                    x=x,
                    z=z,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    embedding=embeddings,
                    name=name,
                )

                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    

# dataset = PIRepeatUnit('dataset/pi/data')
# for data in dataset:
#     # print(data)
#     # print(data.z)
#     # print(data.name)
#     print(data.embedding.shape)
#     break

