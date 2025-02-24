import json
from pathlib import Path
from typing import Dict, Any

from relign.common.registry import RegistrableBase

class Runtime:
    """
    A runtime manager that:
    1. Reads in a 'config' (dict) which may include nested configuration blocks.
    2. Dynamically instantiates (via the registry) the components specified by 'type'.
    3. Allows for further runtime logic and orchestration (running experiments, etc.).
    """

    def __init__(
        self,
        config: Dict[str, Any],     # The parsed config as a Python dict.
        experiment_name: str, 
        run_name: str,
        wandb_project: str,
    ):
        self.config = config
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.wandb_project = wandb_project

        # We will fill in these after parsing/instantiation:
        self.components = {}

    def setup(self):
        """
        Parse the entire configuration tree, instantiating the appropriate objects
        from the registry. This method sets up all necessary runtime components
        (trainers, inference strategies, etc.) and stores them in `self.components`.
        """
        self.components = self._instantiate_all(self.config)

    def _instantiate_all(self, config_block: Dict[str, Any]) -> Any:
        """
        Recursively walk a (sub-)config. If it has a 'type' field, we treat
        the entire block as a class config to be constructed via `RegistrableBase.from_config`.
        Otherwise, we keep drilling down. 
        """

        # If this is not a dict, just return it as-is (e.g. a string or int).
        if not isinstance(config_block, dict):
            return config_block

        # If it *is* a dict but misses 'type', treat it as a nested structure and keep recursing.
        if "type" not in config_block:
            # Return a dict where each sub-value is also processed recursively.
            return {
                key: self._instantiate_all(value)
                for key, value in config_block.items()
            }

        # Otherwise, this dict has a 'type' -> means it's a registrable component
        component_type = config_block["type"]
        component_params = {**config_block}  # shallow copy
        del component_params["type"]

        # Recursively instantiate each sub-block
        for param_key, param_value in component_params.items():
            # If param_value is itself a dict or list of dicts, parse them
            # so that they become actual objects if they have 'type'.
            component_params[param_key] = self._instantiate_all(param_value)

        # Now that all subfields are resolved, instantiate the actual component.
        # (If your base class is something else for each domain, you'd figure that out here.)
        component_obj = RegistrableBase.from_config(dict(type=component_type, **component_params))

        return component_obj

    def run(self):
        """
        Runs the app. This is just a placeholder. 
        In practice, you might do something like:
          - trainer = self.components["trainer"]
          - results = trainer.train()
        """
        #runner from top level component 
        runner = self.components['runner']

        # start the train run 
        runner.algortihm.run()


    def teardown(self):
        """
        Cleanup or teardown logic, if needed.
        """
        pass

