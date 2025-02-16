# myproject/registrable.py
from typing import Type, Dict, TypeVar, List

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
        return list(cls._registry.keys())C