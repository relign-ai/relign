from typing import Dict, Any
from pathlib import Path

class Runtime:
    def __init__(
        self, 
        config: Path, 
        classes: Dict[str, Any]
    ):
        self.config = config
        self.classes = classes

    def setup(self):
        """ 
        Takes an jsonnet experiment and builds out the jsonnet tree. We then 
        pass the appropirate kwargs form the jsonnet to the right class  instances/
        objets
        """
        pass

    def teardown(self):
        """ Tears down the classe"""
        pass

