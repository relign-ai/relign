from typing import Type, Dict

from episode_generation.environment.base_environment import Env
from utils.print import colorful_print


class EnvironmentFactory:
    """
    Factory class to create environments based on configuration.
    """

    _environments: Dict[str, Type[Env]] = {}

    @classmethod
    def register_environment(cls, name: str, env_class: Type[Env]):
        """
        Register an environment class with the factory.

        Args:
            name (str): The name of the environment.
            env_class (Type[BaseEnvironment]): The environment class to register.
        """
        colorful_print(f"Registering environment {env_class.__name__}", fg="green")
        cls._environments[name] = env_class

    @classmethod
    def create_environment(cls, name: str, *args, **kwargs) -> Env:
        """
        Create an instance of the environment with the given name.

        Args:
            name (str): The name of the environment.

        Returns:
            BaseEnvironment: An instance of the requested environment.
        """
        if name not in cls._environments:
            raise ValueError(f"Environment '{name}' not registered.")
        env_class = cls._environments[name]
        return env_class(*args, **kwargs)
