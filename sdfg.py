import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path=".", config_name="zz")
def my_app(cfg: DictConfig):
    detected = "B"  # runtime detection
    
    # Merge real_config defaults with the detected server dict
    cfg.real_config = OmegaConf.merge(cfg.real_config, cfg.servers[detected])
    
    print(OmegaConf.to_yaml(cfg.real_config))

if __name__ == "__main__":
    my_app()
