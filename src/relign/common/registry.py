# myproject/registrable.py
from typing import Type, Dict, TypeVar, List, Any

T = TypeVar("T", bound="RegistrableBase")

class RegistrableBase:
    """
    A lightweight base class offering a named registry for subclasses.
    Inherit from this base, then decorate your subclasses with @register("name").
    """
    _registry: Dict[str, Type["RegistrableBase"]] = {}

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
    def from_config(cls: Type[T], config: Dict[str, Any]) -> Type[T]:
        """
        Construct an instance of a subclass from a configuration dictionary.
        The configuration must have a "type" key corresponding to the subclass name.
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
        
        # Instantiate using the config without the type key
        return subclass(**config_copy)

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        """
        Return the subclass associated with `name`.

        Raises:
            ValueError if no subclass is found under that name.
        """
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
        return list(cls._registry.keys())