"""
Util
"""

import click
import warnings


def colorful_print(string: str, *args, **kwargs) -> None:
    print(click.style(string, *args, **kwargs))


def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs))
