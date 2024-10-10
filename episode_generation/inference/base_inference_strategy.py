from abc import ABC, abstractmethod
from datasets import Dataset


class InferenceStrategy(ABC):
    @abstractmethod

    #TODO: specify dataset format
    def generate(self, dataset: Dataset) -> Dataset:
        """
        Params:
            dataset: The datasest to genreate candidate answers from. 
        Returns:
            Dataset: A dataset of episodes. 
        """
        pass