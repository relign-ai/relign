from typing import List, Union
from abc import ABC, abstractmethod

from common.types import Episode
from common.dataset import EpisodeDataset


class Buffer(ABC):
    def __init__():
        pass
        
    @abstractmethod
    def add():
        pass

    @abstractmethod
    def sample():
        pass


class ReplayBuffer(ABC):
    def __init__(self,
                 max_size: int,
                 batch_size: int,
                 ):
        self.max_size = max_size
        self.batch_size = batch_size
        self.size = 0,

    def add(self, episodes: Union[List[Episode], EpisodeDataset]):
        pass

    def sample(self, batch_size: int) -> EpisodeDataset:
        pass

