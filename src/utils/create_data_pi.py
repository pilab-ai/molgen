import os
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected

import MDAnalysis as mda
from MDAnalysis.transformations import unwrap
from tqdm import tqdm

from src.utils.create_data_pi_repeat_unit import PIRepeatUnit

def parse_trajectory(data, out, unwrap_pbc: bool = True):
    mass_to_element = {
        1.008: 1,
        12.011: 6,
        14.0067: 7,
        15.9994: 8,
        18.9984: 9,
        32.06: 16
    }
    atom_vocab = [1, 6, 7, 8, 9, 16]
    atom2id = {a:i for i,a in enumerate(atom_vocab)}

    # 读取 Universe
    u = mda.Universe(data, out, format="LAMMPSDUMP")

    if unwrap_pbc:
        try:
            u.trajectory.add_transformations(unwrap(u.atoms))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to unwrap trajectory {out}. "
                "Check that topology bonds and box dimensions are available."
            ) from exc

    # 提取原子序数 z 和对应编号 x，形状应均为 [num_nodes, 1]
    elements = [mass_to_element[m] for m in u.atoms.masses]
    z = torch.tensor(elements, dtype=torch.long)
    x = torch.tensor([atom2id[a] for a in elements], dtype=torch.long)

    bonds = u.bonds.indices
    edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()
    # 转换为无向图 (i->j 且 j->i)
    edge_index = to_undirected(edge_index)

    n_frames = len(u.trajectory)
    frame_indices = np.linspace(0, n_frames - 1, 10, dtype=int)

    all_pos = []
    for i in frame_indices:
        u.trajectory[i]
        pos = u.atoms.positions.copy()
        all_pos.append(pos)

    return x, z, edge_index, all_pos

class PI(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        conformation_dir = f"{self.raw_dir}/conformation"
        # 所有 mol 文件
        return [f for f in os.listdir(conformation_dir)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        conformation_dir = Path(f"{self.raw_dir}/conformation")
        dataset = PIRepeatUnit('dataset/pi/data')
        data_list = []

        for unit_data in tqdm(dataset):
            name = unit_data.name
            for degree in [16, 32, 64]:
                folder = conformation_dir/f"{name}_{degree}"
                if folder.is_dir():
                    for temperature in range(200, 650, 50):
                        data_file = folder/f"polymer_{name}_{degree}npt.data"
                        out_file = folder/f"{temperature}_{name}_{degree}.out"
                        if data_file.is_file() and out_file.is_file():
                            x, z, edge_index, all_pos = parse_trajectory(data_file, out_file)
                            for pos in all_pos:
                                # ===== Data =====
                                data = Data(
                                    name=name,

                                    unit_embedding = unit_data.embedding,

                                    x=x,
                                    z=z,
                                    edge_index=edge_index,
                                    
                                    degree=torch.tensor([degree], dtype=torch.int),
                                    temperature = torch.tensor([temperature], dtype=torch.float),
                                    pos=torch.tensor(pos, dtype=torch.float)  # [200, N, 3]
                                )

                                if self.pre_filter and not self.pre_filter(data):
                                    continue

                                if self.pre_transform:
                                    data = self.pre_transform(data)

                                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get(self, idx):
        data = super().get(idx).clone()
        data.data_idx = torch.tensor([idx], dtype=torch.long)
        return data
    
dataset = PI('dataset/pi/data')
# for data in dataset:
#     print(data)
#     break
# print(len(dataset))





