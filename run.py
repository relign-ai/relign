import hydra
from omegaconf import OmegaConf
from utils.print import colorful_print


from agent.factory import AgentFactory
from environment.factory import EnvironmentFactory


CONFIG_NAME = "archer_city.yaml"


@hydra.main(config_path="config", config_name=CONFIG_NAME)
def main(config):
    colorful_print(">>> Configuration file: " + CONFIG_NAME + "<<<", fg="blue")
    colorful_print(OmegaConf.to_yaml(config), fg="red")

    # load the appropriate agent
    try:
        agent = AgentFactory.create_agent(config.agent, config)
        colorful_print(f"Successfully loaded agent '{config.agent}'.", fg="green")
    except ValueError as e:
        colorful_print(str(e), fg="red")
        return

    # Load the appropriate trainer (based on the agent)

    # Load the appropriate environment

    env_name = config.env.name
    num_envs = config.env.num_envs

    try:
        env = EnvironmentFactory.create_environment(env_name, num_envs)
        colorful_print(f"Successfully loaded environment '{env_name}'.", fg="green")
    except ValueError as e:
        colorful_print(str(e), fg="red")
        return
    # call the training loop


if __name__ == "__main__":
    main()
