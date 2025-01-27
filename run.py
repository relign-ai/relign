import hydra
import torch
from omegaconf import OmegaConf
from utils.print import colorful_print

from policies.policy_factory import AgentFactory
from environment.factory import EnvironmentFactory
from train_loop import train_loop

CONFIG_NAME = "archer_city.yaml"


@hydra.main(config_path="config", config_name=CONFIG_NAME)
def main(config):
    colorful_print(">>> Configuration file: " + CONFIG_NAME + "<<<", fg="blue")
    colorful_print(OmegaConf.to_yaml(config), fg="blue")

    cache_dir = config.cache_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the appropriate agent
    try:
        agent = AgentFactory.create_agent(config.agent, config)
        colorful_print(f"Successfully loaded agent '{config.agent}'.", fg="green")
    except ValueError as e:
        colorful_print(str(e), fg="red")
        return

    # Load the appropriate environment
    env_name = config.env_name
    env_load_path = config.env_load_path

    try:
        env = EnvironmentFactory.create_environment(
            env_name, env_load_path, device, cache_dir
        )
        colorful_print(f"Successfully loaded environment '{env_name}'.", fg="green")
    except ValueError as e:
        colorful_print(str(e), fg="red")
        return


if __name__ == "__main__":
    main()
