import torch
from environment.utils import batch_interact_environment
from transformers import AutoTokenizer

import hydra


from environment.factory import EnvironmentFactory
from agent.factory import AgentFactory

tokenizer = AutoTokenizer.from_pretrained("gpt2")

CONFIG_NAME = "archer_city.yaml"


@hydra.main(config_path="config", config_name=CONFIG_NAME)
def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = AgentFactory.create_agent(config.agent, config, device=device)
    env = EnvironmentFactory.create_environment(
        config.env_name, config.env_load_path, device, config.cache_dir
    )

    trajectories = batch_interact_environment(
        agent,
        env,
        config.rollout_size,
    )

    for trajectory in trajectories:
        print("\ntrajectory: ", trajectory)
        print("\n")


if __name__ == "__main__":
    main()
