from typing import Type, Dict
from environment.base import BaseEnvironment


class EnvironmentFactory:
    """
    Factory class to create environments based on configuration.
    """

    _environments: Dict[str, Type[BaseEnvironment]] = {}

    @classmethod
    def register_environment(cls, name: str, env_class: Type[BaseEnvironment]):
        """
        Register an environment class with the factory.

        Args:
            name (str): The name of the environment.
            env_class (Type[BaseEnvironment]): The environment class to register.
        """
        cls._environments[name] = env_class

    @classmethod
    def create_environment(cls, name: str, *args, **kwargs) -> BaseEnvironment:
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
