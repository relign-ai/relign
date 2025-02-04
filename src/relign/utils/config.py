import os
import hydra
from omegaconf import OmegaConf

def load_deepspeed_config():
    """
    Load and return the DeepSpeed configuration.
    
    This function:
      - Computes the project root assuming this file is located at:
        /root/relign/src/relign/utils/config.py
      - Derives the absolute path for the configs directory at /root/relign/configs.
      - Converts that absolute path into a relative path (Hydra requirement).
      - Initializes Hydra and composes the configuration.
      - Converts and returns the deepspeed portion of the config as a plain container.
    """
    # The project root is three directories up from this file:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    abs_configs_dir = os.path.join(project_root, "configs")
    # Hydra.initialize() requires a relative path to the config directory;
    # assuming tests or your application run from the project root, this should resolve to "configs"
    configs_dir = os.path.relpath(abs_configs_dir, os.getcwd())
    
    hydra.initialize(config_path=configs_dir, version_base=None)
    cfg = hydra.compose(config_name="config")
    ds_config = cfg.deepspeed
    return OmegaConf.to_container(ds_config, resolve=True)