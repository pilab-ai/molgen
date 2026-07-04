import os
import torch
from torch_geometric.data import InMemoryDataset, Data

class PILatent(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(
            os.path.join(self.processed_dir, "data_with_latent.pt"),
            weights_only=False,
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data_with_latent.pt"]

    def process(self):
        pass

    def get_z_stats(self):
        stats = torch.load(os.path.join(self.processed_dir, "code_stats.pt"), weights_only=False)
        return stats['mean'], stats['std']

    def get_unit_embedding_dim(self):
        return int(self.get(0).unit_embedding.view(1, -1).size(-1))
    
    def get_condition_stats(self):

        degree = self.data.degree.float()
        temperature = self.data.temperature.float()
        degree = torch.log(degree.clamp_min(1.0))

        return {
            "degree_mean": degree.mean(dim=0, keepdim=True),
            "degree_std": degree.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6),
            "temperature_mean": temperature.mean(dim=0, keepdim=True),
            "temperature_std": temperature.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6),
        }
    
    def get(self, idx):
        data = super().get(idx).clone()
        data.data_idx = torch.tensor([idx], dtype=torch.long)
        return data
    

    