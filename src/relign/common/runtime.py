from typing import Dict, Any
from pathlib import Path

class Runtime:
    def __init__(
        self, 
        config: Path, 
        experiment_name: str,
        run_name: str,
        wandb_project: str,
    ):
        self.config = config
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.wandb_project = wandb_project

    def setup(self):
        """ 
        Takes an jsonnet experiment and builds out the jsonnet tree. We then 
        pass the appropirate kwargs form the jsonnet to the right class  instances/
        objets
        """

         
        pass

    def run(self):
        """ 
        Runs the app   
        """
        pass

    def teardown(self):
        """ Tears down the class"""
        pass

