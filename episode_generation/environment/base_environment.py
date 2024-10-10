from abc import ABC, abstractmethod
from typing import (
    Optional,
    List,
    Tuple,
    Dict,
    Self,
)

from copy import deepcopy
from common.types import Reward, Done, History, TextHistory


class Env(ABC):
    @abstractmethod
    def step(self, history: History) -> Tuple[History, Reward, Done]:
        pass

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> History:
        pass

    def close(self) -> None:
        pass

    def copy(self) -> Self:
        return deepcopy(self)
    

class TextEnv(Env):
    @abstractmethod
    def step(self, history: TextHistory) -> Tuple[TextHistory, Reward, Done]:
        pass


class BatchedEnv(ABC):
    @abstractmethod
    def step(
        self,
        histories: List[Optional[History]],
        done: Optional[List[Done]] = None,
    ) -> List[Optional[Tuple[History, Reward, Done]]]:
        pass

    @abstractmethod
    def reset(
        self,
        seed: Optional[List[Optional[int]]] = None,
        options: Optional[List[Optional[Dict]]] = None,
    ) -> List[History]:
        pass

    def close(self) -> None:
        pass

    def copy(self) -> Self:
        return deepcopy(self)


# Adapter to convert Env to BatchedEnv
class EnvToBatchedEnv(BatchedEnv):
    def __init__(self, env: Env, batch_size: int = 1):
        self.env = env
        self.batch_size = batch_size

    def step(
        self,
        histories: List[Optional[History]],
        done: Optional[List[Done]] = None,
    ) -> List[Optional[Tuple[History, Reward, Done]]]:
        results = []
        for history, done_flag in zip(histories, done or [False]*self.batch_size):
            if done_flag or history is None:
                results.append(None)
            else:
                step_result = self.env.step(history)
                results.append(step_result)
        return results

    def reset(
        self,
        seed: Optional[List[Optional[int]]] = None,
        options: Optional[List[Optional[Dict]]] = None,
    ) -> List[History]:
        reset_histories = []
        for _ in range(self.batch_size):
            history = self.env.reset()
            reset_histories.append(history)
        return reset_histories
    


