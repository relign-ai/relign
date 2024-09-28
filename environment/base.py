from abc import ABC, abstractmethod
from typing import Any, Optional, List, Tuple


class BaseEnvironment(ABC):
    """
    Base class for all environment implementations.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the environment.
        """

    @abstractmethod
    def reset(self, *args, **kwargs) -> Any:
        """
        Reset the environment to its initial state.

        Returns:
            The initial observation from the environment.
        """

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool]:
        """
        Take an action in the environment.

        Args:
            action (Any): An action provided by the agent.

        Returns:
            observation (Any): The next observation from the environment.
            reward (float): The reward obtained from taking the action.
            done (bool): Whether the episode has ended.
        """
