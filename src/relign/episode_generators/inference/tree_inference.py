from episode_generation.inference.base_inference_strategy import InferenceStrategy
from datasets import Dataset


class TreeInference(InferenceStrategy):
    """
    Implementation of VinePPO's inference strategy. 
    """
    def generate(self, dataset: Dataset) -> Dataset:
        pass
    
    
