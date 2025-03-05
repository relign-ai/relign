from typing import Dict, Any, Type, List
from pathlib import Path
import json
import _jsonnet

from relign.utils.logging import get_logger
from relign.common.registry import RegistrableBase, LazyConfig

logger = get_logger(__name__)



class Runtime:
    """
    A runtime manager that:
    1. Reads in a 'config' (dict) which may include nested configuration blocks.
    2. Dynamically instantiates (via the registry) the *top-level* component (e.g., a runner).
    3. For any nested components (marked by "type"), returns a lazy instantiator instead of
       constructing them immediately. This allows passing additional runtime parameters before
       instantiation.
    """

    def __init__(
        self,
        config: Dict[str, Any],     # The parsed config as a Python dict.
        experiment_name: str, 
        run_name: str,
        wandb_project: str,
    ):
        self.config_path = config
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.wandb_project = wandb_project
        self.config_data = None
        self.instances = {}
        
        # Store runtime parameters for later use
        self.runtime_params = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "wandb_project": wandb_project,
            "directory": config.parent,  # Use the config directory as project directory
        }
    
    def _load_jsonnet(self, path: Path) -> Dict[str, Any]:
        """Load and parse a jsonnet file."""
        try:
            json_str = _jsonnet.evaluate_file(str(path))
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error loading jsonnet file {path}: {e}")
            if "unterminated string" in str(e):
                logger.error("Check for missing quotes or unclosed string literals in your jsonnet file")
            raise

    def _process_config_node(self, config_node: Dict[str, Any], parent_path: List[str] = None) -> Dict[str, Any]:
        """
        Recursively process a configuration node, preparing it for lazy instantiation
        if it has a 'type' key.
        """
        if parent_path is None:
            parent_path = []
        
        if not isinstance(config_node, dict):
            return config_node
        
        processed_config = {}
        
        # Carry over keys, potentially processing nested dicts
        for key, value in config_node.items():
            if isinstance(value, dict):
                processed_config[key] = self._process_config_node(
                    value,
                    parent_path + [key]
                )
            else:
                processed_config[key] = value
                
        return processed_config

    def _instantiate_component(
        self,
        config: Dict[str, Any],
        base_class: Type[RegistrableBase],
        additional_kwargs: Dict[str, Any] = None
    ) -> Any:
        """
        Instantiate the top-level component using the `RegistrableBase` system.
        For nested components (sub-configs with a 'type'), store them as lazy instantiators
        so we can pass additional runtime parameters before actually constructing them.
        """
        if additional_kwargs is None:
            additional_kwargs = {}
            
        # Make a copy to avoid modifying the original config
        config_copy = config.copy()
        
        # Inject additional runtime parameters if they're not already present
        for key, val in additional_kwargs.items():
            if key not in config_copy:
                config_copy[key] = val

        # For any nested dict that has a "type", convert it into a lazy instantiator
        for key, val in list(config_copy.items()):
            if isinstance(val, dict) and "type" in val:
                # Replace this nested dict with a lazy instantiator
                config_copy[key] = self._create_lazy_instantiator(val)

        # Now fully instantiate the top-level object
        try:
            instance = base_class.from_config(config_copy)
            return instance
        except Exception as e:
            logger.error(f"Error instantiating component with type '{config.get('type', 'unknown')}': {e}")
            raise

    def _create_lazy_instantiator(self, config: Dict[str, Any]) -> "LazyConfig":
        """
        Create a LazyConfig wrapper that can be used to instantiate a component
        later, with optional extra parameters.
        """
        # We can validate that config has a "type"
        if "type" not in config:
            logger.warning("Creating LazyConfig for a dict without 'type'?")
        return LazyConfig(config)

    def setup(self):
        """ 
        Load the config, process it, and instantiate the top-level runner.
        Sub-components will remain lazily instantiated.
        """
        logger.info(f"Loading config from {self.config_path}")
        self.config_data = self._load_jsonnet(self.config_path)
        
        # Process the config to prepare nested structures
        processed_config = self._process_config_node(self.config_data)
        
        # For the top-level, we know it's a runner
        from relign.runners.base_runner import BaseRunner
        
        # Store all processed configs in case we need them
        self.config_store = {"root": processed_config}
        
        # Instantiate the top-level runner. Nested components remain lazy.
        logger.info("Instantiating top-level runner from config")
        self.runner = self._instantiate_component(
            processed_config,
            BaseRunner,
            self.runtime_params
        )
        
        logger.info("Setup complete")
        return self

    def run(self):
        """ 
        Execute the runner.
        """
        if not hasattr(self, 'runner'):
            logger.error("Cannot run before setup. Call setup() first.")
            return
            
        logger.info("Starting run")
        try:
            self.runner.run()
            logger.info("Run completed successfully")
        except Exception as e:
            logger.error(f"Error during run: {e}")
            raise

    def teardown(self):
        """ Teardown the runner and any other resources. """
        if hasattr(self, 'runner') and hasattr(self.runner, 'teardown'):
            logger.info("Tearing down runner")
            self.runner.teardown()
        
        self.instances = {}
        self.config_store = {}
        logger.info("Teardown complete")

