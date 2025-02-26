from typing import Dict, Any, Type, Optional, Union, List
from pathlib import Path
import importlib
import json
import os
import _jsonnet

from relign.utils.logging import get_logger
from relign.common.registry import RegistrableBase

logger = get_logger(__name__)


class Runtime:
    def __init__(
        self, 
        config: Path, 
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
            raise

    def _process_config_node(self, config_node: Dict[str, Any], parent_path: List[str] = None) -> Dict[str, Any]:
        """
        Recursively process a configuration node, identifying class types and preparing
        for instantiation through the RegistrableBase system.
        """
        if parent_path is None:
            parent_path = []
        
        # Handle primitive types
        if not isinstance(config_node, dict):
            return config_node
        
        processed_config = {}
        
        # Check if this node defines a class type
        if "type" in config_node:
            processed_config["type"] = config_node["type"]
            # Store the original type for reference
            processed_config["_original_type"] = config_node["type"]
        
        # Process all other keys recursively
        for key, value in config_node.items():
            if key == "type" or key.startswith("_"):
                continue
                
            if isinstance(value, dict):
                new_path = parent_path + [key]
                processed_config[key] = self._process_config_node(value, new_path)
            else:
                processed_config[key] = value
                
        return processed_config

    def _instantiate_component(self, config: Dict[str, Any], base_class: Type[RegistrableBase], additional_kwargs: Dict[str, Any] = None) -> Any:
        """
        Instantiate a component using the RegistrableBase system.
        
        Args:
            config: The configuration dict for this component
            base_class: The base class to use for instantiation
            additional_kwargs: Additional runtime parameters
            
        Returns:
            An instantiated component
        """
        if additional_kwargs is None:
            additional_kwargs = {}
            
        # Make a copy of the config to avoid modifying the original
        config_copy = config.copy()
        
        # Add runtime parameters
        for key, value in additional_kwargs.items():
            if key not in config_copy:
                config_copy[key] = value
                
        # Handle nested components
        for key, value in config_copy.items():
            if isinstance(value, dict) and "_original_type" in value:
                # This is a nested component, keep it as a config dict for lazy instantiation
                continue
                
        # Use the RegistrableBase system to instantiate
        try:
            instance = base_class.from_config(config_copy)
            return instance
        except Exception as e:
            logger.error(f"Error instantiating component with type '{config.get('type', 'unknown')}': {e}")
            raise
            
    def _create_lazy_instantiator(self, config: Dict[str, Any], base_class: Type[RegistrableBase]):
        """
        Creates a function that will instantiate a component when called.
        This enables lazy instantiation of nested components.
        
        Args:
            config: The component configuration
            base_class: The base class to use for instantiation
            
        Returns:
            A function that when called will instantiate the component
        """
        def instantiate(**runtime_kwargs):
            merged_kwargs = {**self.runtime_params, **runtime_kwargs}
            return self._instantiate_component(config, base_class, merged_kwargs)
            
        return instantiate

    def setup(self):
        """ 
        Takes a jsonnet experiment and builds out the jsonnet tree. We then 
        pass the appropriate kwargs from the jsonnet to the right class instances/
        objects
        """
        # Load the config
        logger.info(f"Loading config from {self.config_path}")
        self.config_data = self._load_jsonnet(self.config_path)
        
        # Process the config
        processed_config = self._process_config_node(self.config_data)
        
        # For the top level, we know it's a runner
        from relign.runners.base_runner import BaseRunner
        
        # Store all configurations in the instances dict for potential reuse
        self.config_store = {"root": processed_config}
        
        # Instantiate the top-level runner
        logger.info("Instantiating components from config")
        self.runner = self._instantiate_component(processed_config, BaseRunner, self.runtime_params)
        
        logger.info("Setup complete")
        return self

    def run(self):
        """ 
        Runs the application
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
        """ Tears down the class"""
        if hasattr(self, 'runner') and hasattr(self.runner, 'teardown'):
            logger.info("Tearing down runner")
            self.runner.teardown()
        
        # Clean up any other resources
        self.instances = {}
        self.config_store = {}
        logger.info("Teardown complete")

