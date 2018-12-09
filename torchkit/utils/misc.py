import os

from typing import Container


def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.
    
    Args:
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
