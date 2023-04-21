import os

import appdirs
from pubnet import __name__ as pkg_name

__all__ = ["default_cache_dir", "default_data_dir", "delete", "list"]

_APPAUTHOR = "net_synergy"


def default_cache_dir() -> str:
    """Find the default location to save cache files.

    If the directory does not exist it is created.

    Cache files are specifically files that can be easily reproduced,
    i.e. those that can be downloaded from the internet.
    """

    cache_dir = appdirs.user_cache_dir(pkg_name, _APPAUTHOR)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir, mode=0o755)

    return cache_dir


def default_data_dir() -> str:
    """Find the default location to save data files.

    If the directory does not exist it is created.

    Data files are files created by a user. It's possible they can be
    reproduced by rerunning the script that produced them but there is
    no gurentee they can be perfectly reproduced.
    """

    data_dir = appdirs.user_data_dir(pkg_name, _APPAUTHOR)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir, mode=0o755)

    return data_dir


def list(data_dir=default_data_dir()):
    """List all graphs saved in `data_dir`"""

    return os.listdir(data_dir)


def delete(graph_name, data_dir=default_data_dir()):
    """Delete the graph from `data_dir`"""

    def delete_directory(path):
        for f in os.listdir(path):
            os.unlink(os.path.join(path, f))
        os.rmdir(path)

    path = os.path.join(data_dir, graph_name)
    if os.path.isdir(path):
        delete_directory(path)
    else:
        raise NotADirectoryError(f"{graph_name} not found in {data_dir}")
