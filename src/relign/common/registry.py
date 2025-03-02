# myproject/registrable.py
from typing import Type, Dict, TypeVar, List, Any, get_type_hints, get_origin, get_args, Optional, Union, Generic
import inspect
from functools import lru_cache

T = TypeVar("T", bound="RegistrableBase")

# Define a Lazy generic type for type annotations
class Lazy(Generic[T]):
    """
    Generic type hint for indicating a parameter should receive a lazy component.
    This is used purely for type annotations and has no runtime behavior.
    """
    pass


class LazyConfig:
    """
    A small class that holds a config dict for a subcomponent (with a "type"),
    and offers an instance method `from_config(...)` which instantiates it.
    """

    def __init__(self, config: Dict[str, Any]):
        # e.g. {"type":"math_reward_function", "arg1":..., ...}
        self._config = config.copy()
        if "type" not in self._config:
            raise ValueError("LazyConfig must have a 'type' field in config.")
        
        # Add method to check if this is a LazyConfig to use in isinstance checks
        self.__lazy_config__ = True

    def from_config(self, **runtime_kwargs) -> Any:
        """
        Merges runtime kwargs, finds the correct subclass from the registry, 
        and calls its .from_config(...) method to produce an *instantiated* component.
        """
        merged_config = self._config.copy()
        # Merge extra params
        for k, v in runtime_kwargs.items():
            if k not in merged_config:
                merged_config[k] = v
        
        # We'll look up all classes that subclass RegistrableBase
        from relign.common.registry import RegistrableBase
        type_name = merged_config["type"]

        for subclass in RegistrableBase.__subclasses__():
            if hasattr(subclass, '_registry') and type_name in subclass._registry:
                return subclass.from_config(merged_config)
        
        # If not found, fallback
        import logging
        logging.warning(
            f"No registry class found for type '{type_name}', returning config dict."
        )
        return merged_config


class RegistrableBase:
    """
    A lightweight base class offering a named registry for subclasses.
    Inherit from this base, then decorate your subclasses with @register("name").
    """
    # Don't define _registry here

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a subclass under a given name.
        
        Usage:
            @MyBaseClass.register("my_subclass")
            class MySubclass(MyBaseClass):
                ...

        Raises:
            ValueError if the name is already registered.
        """
        # Create registry for this class if it doesn't exist yet
        if not hasattr(cls, '_registry'):
            cls._registry = {}
            
        def decorator(subclass: Type[T]) -> Type[T]:
            if name in cls._registry:
                raise ValueError(
                    f"Cannot register '{name}' under {cls.__name__}; "
                    f"already used by {cls._registry[name].__name__}."
                )
            cls._registry[name] = subclass
            return subclass
        return decorator
    
    @classmethod
    @lru_cache(maxsize=128)
    def _get_constructor_type_hints(cls, subclass: Type[T]) -> Dict[str, Any]:
        """
        Get and cache the type hints for a class constructor.
        Using lru_cache to avoid repeatedly parsing the same class.
        """
        return get_type_hints(subclass.__init__)
    
    @classmethod
    def _should_be_instantiated(cls, param_name: str, param_type: Any) -> bool:
        """
        Determine if a parameter should be instantiated based on its type annotation.
        
        Returns True if:
        - The type is NOT wrapped in Lazy[...]
        - The type is not Optional[Lazy[...]]
        - The type is not Union[..., Lazy[...], ...]
        """
        # Handle Optional[...] which is implemented as Union[..., None]
        origin = get_origin(param_type)
        if origin is Union:
            # If it's a Union or Optional, check each argument
            args = get_args(param_type)
            for arg in args:
                # If any argument is Lazy, then don't instantiate
                arg_origin = get_origin(arg)
                if arg_origin is Lazy:
                    return False
                # If None type, skip (part of Optional)
                if arg is type(None):
                    continue
                # Recursive check for nested types
                if not cls._should_be_instantiated(param_name, arg):
                    return False
            # If no Lazy found in the Union, then instantiate
            return True
        
        # Handle direct Lazy[...]
        if origin is Lazy:
            return False
        
        # Default to eager instantiation
        return True

    @classmethod
    def from_config(cls: Type[T], config: Dict[str, Any]) -> Type[T]:
        """
        Construct an instance of a subclass from a configuration dictionary.
        The configuration must have a "type" key corresponding to the subclass name.
        
        This enhanced version will:
        1. Look up the subclass based on the 'type' field
        2. Examine type hints of the subclass constructor
        3. For each parameter with a dict with 'type', decide whether to:
           - Pass a LazyConfig (if param is annotated as Lazy[...])
           - Instantiate the component (if param is annotated as a concrete type)
        """
        if 'type' not in config:
            raise ValueError("No 'type' field provided in config.")
        
        # Get a copy of the config to avoid modifying the original
        config_copy = config.copy()
        subclass_type = config_copy.pop("type")  # Remove type to avoid recursion
        
        # Remove any internal metadata keys (those starting with _)
        for key in list(config_copy.keys()):
            if key.startswith("_"):
                config_copy.pop(key)
        
        # Get the appropriate subclass
        subclass = cls.by_name(subclass_type)
        
        # Get constructor type hints
        try:
            type_hints = cls._get_constructor_type_hints(subclass)
        except (TypeError, ValueError):
            # If we can't get type hints (e.g., for a built-in), proceed with default behavior
            type_hints = {}
        
        # Process nested config dictionaries with 'type' key
        processed_args = {}
        for key, value in config_copy.items():
            if isinstance(value, dict) and "type" in value:
                # Check if we should instantiate or keep lazy
                if key in type_hints and cls._should_be_instantiated(key, type_hints[key]):
                    # Instantiate the component
                    for subcls in RegistrableBase.__subclasses__():
                        if hasattr(subcls, '_registry') and value["type"] in subcls._registry:
                            processed_args[key] = subcls.from_config(value)
                            break
                    else:
                        # If no registry found, keep as LazyConfig
                        processed_args[key] = LazyConfig(value)
                else:
                    # Keep as LazyConfig
                    processed_args[key] = LazyConfig(value)
            elif hasattr(value, '__lazy_config__'):
                # This is already a LazyConfig instance
                if key in type_hints and cls._should_be_instantiated(key, type_hints[key]):
                    # We need to instantiate it
                    processed_args[key] = value.from_config()
                else:
                    # Keep it lazy
                    processed_args[key] = value
            else:
                # Pass through non-dict values or dicts without 'type'
                processed_args[key] = value
        
        # Instantiate using the processed arguments
        return subclass(**processed_args)

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        """
        Return the subclass associated with `name`.

        Raises:
            ValueError if no subclass is found under that name.
        """
        # Make sure we have a registry
        if not hasattr(cls, '_registry'):
            cls._registry = {}
            
        if name not in cls._registry:
            raise ValueError(
                f"No '{cls.__name__}' registered under the name: '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all registered names for this base class.
        """
        # Make sure we have a registry
        if not hasattr(cls, '_registry'):
            cls._registry = {}
            
        return list(cls._registry.keys())