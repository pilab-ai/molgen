from tqdm import tqdm
import torch

from src.trainers.autodecoder_trainer import TMolAutoDecoderTrainer

def run(dataset, best_ckpt_path, save_dir):
    print(f"Best checkpoint: {best_ckpt_path}")
    best_model = TMolAutoDecoderTrainer.load_from_checkpoint(
        checkpoint_path = best_ckpt_path,
        map_location="cpu",
    )
    best_model.eval()
    codes = best_model.codes.weight.detach().cpu()
    assert codes.size(0) == len(dataset), (
        f"codes 数量 {codes.size(0)} 与 dataset 数量 {len(dataset)} 不一致"
    )
    
    # ===== 统计 latent code =====
    z_mean = codes.mean(dim=0, keepdim=True)  # [1, code_dim]
    z_std = codes.std(dim=0, keepdim=True).clamp_min(1e-6)  # [1, code_dim]
    torch.save(
        {
            "mean": z_mean, 
            "std": z_std
        }, 
        f"{save_dir}/processed/code_stats.pt"
        )

    print("global mean:", codes.mean().item())
    print("global std:", codes.std().item())
    print("global min:", codes.min().item())
    print("global max:", codes.max().item())

    data_list = []
    for idx in tqdm(range(len(dataset))):
        data = dataset.get(idx)
        data.code = codes[idx].clone().unsqueeze(0)  # [1, code_dim]
        data_list.append(data)

    torch.save(dataset.collate(data_list), f"{save_dir}/processed/data_with_latent.pt")
    
