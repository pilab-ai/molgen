import time
import yaml
import torch_geometric.transforms as T
import src.pipelines.train_autodecoder as p1
import src.pipelines.export_dataset_with_latent as p2
import src.pipelines.train_flow as p3

class ScaleTransform:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, data):
        data.pos = data.pos / self.scale
        return data
    
def load_dataset(cfg, transform):
    name = cfg["data"]["name"]
    if name == "qm9":
        from src.utils.create_data_qm9 import QM9
        return QM9(cfg["data"]["dir"], transform=transform)
    if name == "sdbs":
        from src.utils.create_data_sdbs import SDBS
        return SDBS(cfg["data"]["dir"], transform=transform)
    if name == "pi":
        from src.utils.create_data_pi import PI
        return PI(cfg["data"]["dir"], transform=transform)
    raise ValueError(f"Unknown dataset: {name}")

def main():
    
    with open("config/pi.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    transform = T.Compose([
        T.Center(),
        ScaleTransform(cfg["data"]["max_size"]),
    ])

    dataset = load_dataset(cfg, None)

    # train auto-decoder
    start = time.time()
    p1.run(dataset=dataset, cfg=cfg)
    end = time.time()
    print("Total time for training auto-decoder: ", end - start)

    # # export_dataset_with_latent
    # p2.run(
    #     dataset=dataset, 
    #     # best_ckpt_path = checkpoint.best_model_path
    #     best_ckpt_path = "logs/pi/autodecoder_model/version_24/checkpoints/epoch=1183-step=513856.ckpt", 
    #     save_dir = cfg["data"]["dir"]
    #     )

    # # train flow
    # with open("config/flow.yaml", "r") as f:
    #     cfg = yaml.safe_load(f)
    # p3.run(cfg=cfg)


if __name__ == "__main__":
    main()
