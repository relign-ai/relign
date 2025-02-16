from collections import defaultdict
from typing import Any, Dict

# A single global dictionary that maps:
#   Category -> (Name -> Class)
REGISTRIES: Dict[str, Dict[str, Any]] = defaultdict(dict)

def register(category: str, name: str):
    """
    A generic decorator for registering a class under a given category and name.

    Usage Example:
        @register("trainer", "ppo")
        class PPOTrainer:
            ...

        @register("inference", "chain_of_thought")
        class ChainOfThoughtInference:
            ...
    """
    def decorator(cls):
        if name in REGISTRIES[category]:
            raise ValueError(
                f"Duplicate registration: '{name}' already in category '{category}'."
            )
        REGISTRIES[category][name] = cls
        return cls
    return decorator


def get_registered(category: str, name: str):
    """
    Retrieve the class from the registry by category and name.
    """
    cat_dict = REGISTRIES.get(category)
    if not cat_dict:
        raise KeyError(f"No category '{category}' found in registry.")
    cls = cat_dict.get(name)
    if not cls:
        raise KeyError(f"No item '{name}' found in category '{category}'.")
    return cls