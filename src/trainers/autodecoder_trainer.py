import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, scatter
import lightning as L

from src.model.decoder import TMolDecoder

class TMolAutoDecoderTrainer(L.LightningModule):
    def __init__(
        self,
        n_codes: int,
        n_atom_types: int,
        scale: float, 
        code_dim: int = 256,
        max_atoms: int = 10000,
        hidden_dim: int = 512,
        n_layers: int = 3,
        lr_dec: float = 5e-4,
        lr_code: float = 1e-3,
        weight_decay: float = 0.0,
        code_reg_weight: float = 1e-6,
        val_fixed_seed: int = 12345,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.scale=scale

        self.codes = nn.Embedding(n_codes, code_dim)
        nn.init.normal_(self.codes.weight, mean=0.0, std=0.01)

        self.decoder = TMolDecoder(
            z_dim=code_dim,
            max_atoms=max_atoms,
            n_atom_types=n_atom_types,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        self.max_atoms = max_atoms

        self.lr_dec = lr_dec
        self.lr_code = lr_code
        self.weight_decay = weight_decay
        self.code_reg_weight = code_reg_weight
        self.val_fixed_seed = val_fixed_seed
    
    def bond_length_loss(self, pred_pos, true_pos, edge_index):
        row, col = edge_index
        mask = row < col
        row = row[mask]
        col = col[mask]

        pred_len = torch.norm(pred_pos[row] - pred_pos[col], dim=-1)
        true_len = torch.norm(true_pos[row] - true_pos[col], dim=-1)

        return F.l1_loss(pred_len, true_len)

    def _shared_step(self, batch, batch_idx: int, mode: str):

        # 每个分子单独求中心
        center = scatter(
            batch.pos,
            batch.batch,
            dim=0,
            dim_size=batch.num_graphs,
            reduce="mean",
        )  # [B, 3]

        pos_centered = batch.pos - center[batch.batch]
        pos_norm = pos_centered / self.scale

        atom_type, node_mask = to_dense_batch(x=batch.x, batch=batch.batch, max_num_nodes=self.max_atoms)
        coords, _ = to_dense_batch(x=pos_norm, batch=batch.batch, max_num_nodes=self.max_atoms)

        z = self.codes(batch.data_idx)
        out = self.decoder(z)

        loss_exist = F.binary_cross_entropy_with_logits(
            out["exist_logits"],
            node_mask.float(),
        )

        loss_atom = F.cross_entropy(
            out["atom_logits"][node_mask],
            atom_type[node_mask],
        )

        loss_coord = F.smooth_l1_loss(
            out["coords"][node_mask] * self.scale,
            coords[node_mask] * self.scale,
        )

        # loss_bond = self.bond_length_loss(out["coords"][node_mask], coords[node_mask], batch.edge_index)

        loss_code_reg = z.pow(2).mean()
        # loss = loss_exist + loss_atom + loss_coord + loss_bond + self.code_reg_weight*loss_code_reg
        # coord_weight = 10000.0
        # bond_weight = 100.0
        # loss = (
        #     loss_exist
        #     + loss_atom
        #     + coord_weight * loss_coord
        #     + bond_weight * loss_bond
        #     + self.code_reg_weight * loss_code_reg
        # )
        loss = loss_exist + loss_atom + loss_coord


        self.log(f"{mode}/loss/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f"{mode}/loss/code_reg", loss_code_reg, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)

        self.log(f"{mode}/loss/exist", loss_exist, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f"{mode}/loss/atom", loss_atom, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f"{mode}/loss/coord", loss_coord, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch.num_graphs)
        # self.log(f"{mode}/loss/bond", loss_bond, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch.num_graphs)
        
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, mode="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.decoder.parameters(),
                    "lr": self.lr_dec,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": self.codes.parameters(),
                    "lr": self.lr_code,
                    "weight_decay": 0.0,
                },
            ]
        )
        return optimizer
    

