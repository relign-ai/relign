from typing import Optional, Dict, List, Union
from abc import ABC, abstractmethod
from datasets import (
    Dataset,
    DatasetDict,
)

from relign.utils import logging

logger = logging.get_logger(__name__)


class BaseTask(ABC):
    """
    TODO: Add more documentation when the class is stable. For now it looks like it has two main functions:
    
    1. build_dataset: This function fully defines the dataset for this task.
    2. evaluate_predictions: This function evaluates predictions against references and 
       therefore determines the rewards for the task.
    """    

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initializes a new instance of the BaseTask class.

        Args:
            system_prompt: The system prompt to use for the queries of the task
        """
        self._ds_cache: Optional[DatasetDict] = None
        self.system_prompt = system_prompt


    @property
    def name(self) -> str:
        """
        The name of this task
        """
        return f"{self.__class__.__name__}"


    @abstractmethod
    def build_dataset(self) -> DatasetDict:
        """
        Build the datasets for this task. This function is called by get_datasets if the 
        datasets are not already cached. This function fully defines the dataset for this task.
        """
        pass

    
    @abstractmethod
    def evaluate_predictions(
        self,
        *,
        predictions: List[List[str]] = None,
        references: Dataset = None,
    ) -> Dict[str, float]:
        """
        Evaluate predictions against references
        Params:
            predictions: A list of predictions, each prediction is a list of candidate answers
            references: The reference dataset

        Returns:
            A dictionary of metrics
        """
        pass

    
    def get_datasets(
        self, split: Optional[str] = None, no_cache: bool = False
    ) -> Union[DatasetDict, Dataset]:
        """
        Get the datasets for this task

        Params:
            split: The split to get, if None, return the entire dataset
            no_cache: If True, do not use the cached dataset
        """
        if self._ds_cache is None or no_cache:
            self._ds_cache = self.build_dataset()

        if split is None:
            return self._ds_cache
        else:
            return self._ds_cache[split]
        
    
    #----------------- Helper functions -----------------# 

    def map_datasets_fields(data_source: DatasetDict, 
                           field_map: Dict[str, str], 
                           remove_mapped_fields: bool, 
                           hf_num_proc: int) -> DatasetDict:
        """
        Maps the fields of the datasets in the given DatasetDict according to the provided field map.

        Params:
            data_source: The source dataset to be mapped.
            field_map: A dictionary mapping the original field names to the new field names.
            remove_mapped_fields: If True, the original fields will be removed after mapping.
            hf_num_proc: The number of processes to use for parallel processing.

        Raises:
            ValueError: If the data_source is not an instance of DatasetDict.
        """
        
        if not isinstance(data_source, DatasetDict):
            raise ValueError(
                f"The datasource should be a DatasetDict, but got {type(data_source)} instead"
            )

        def map_fields(example):
            return {field_map[k]: v for k, v in example.items()}

        data_source = data_source.map(
            map_fields,
            desc="Mapping fields",
            num_proc=hf_num_proc,
            remove_columns=(
                list(field_map.keys()) if remove_mapped_fields else None
            ),
        )

        data_source = data_source.map(
            lambda idx: {"_treetune__idx": idx},
            with_indices=True,
            num_proc=hf_num_proc,
            desc="Adding idx column",
        )

        return data_source
    