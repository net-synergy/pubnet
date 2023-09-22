"""
Methods for storing and generating publication networks.

Graphs are stored as directories with files for each node and edge. By default,
graphs are stored in the `default_data_dir`. Files can be saved in multiple
forms including plain text--which is easy to modify by hand or with tools
outside of this package, but can be slow in inefficiently stored--or as
compressed binary types--that can't be easily modified by are significantly
faster for larger data and take up less storage space. Additionally, in the
future, graphs DBs may be supported.
"""

import os
from typing import List, Optional

import appdirs

from pubnet import __name__ as pkg_name

__all__ = [
    "set_default_cache_dir",
    "set_default_data_dir",
    "default_cache_dir",
    "default_data_dir",
    "delete_graph",
    "list_graphs",
    "clear_cache",
    "clear_data",
]

_APPAUTHOR = "net_synergy"

pkg_cache_dir = ""
pkg_data_dir = ""


def set_default_cache_dir(new_path: str = "") -> None:
    global pkg_cache_dir
    pkg_cache_dir = new_path


def set_default_data_dir(new_path: str = "") -> None:
    global pkg_data_dir
    pkg_data_dir = new_path


def default_cache_dir(path: str = "") -> str:
    """Find the default location to save cache files.

    The location of the default cache is dependent on the OS and environment
    variables. It can also be modified with `set_cache_dir`.

    If the directory does not exist it is created.

    Cache files are specifically files that can be easily reproduced,
    i.e. those that can be downloaded from the internet.

    Arguments
    ---------
    path : str, optional
        If empty return the parent cache directory, otherwise return the
        directory `path` under the parent cache directory. If the path does not
        exist it will be created. Behavior is modified by `abs_path`.

    Returns
    -------
    cache_dir : str
        The path to the directory to store data.

    See also
    --------
    `set_cache_dir`, `default_data_dir`
    """

    if pkg_cache_dir:
        cache_dir = pkg_cache_dir
    else:
        cache_dir = appdirs.user_cache_dir(pkg_name, _APPAUTHOR)

    cache_dir = os.path.join(cache_dir, path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, mode=0o755)

    return cache_dir


def default_data_dir(path: str = "") -> str:
    """Find the default location to save data files.

    The location of the default data directory is dependent on the OS and
    environment variables. It can also be modified with `set_data_dir`.

    If the directory does not exist it is created.

    Data files are files created by a user. It's possible they can be
    reproduced by rerunning the script that produced them but there is
    no gurentee they can be perfectly reproduced.

    Arguments
    ---------
    path : str, optional
        If empty return the parent cache directory, otherwise return the
        directory `path` under the parent cache directory. If the path does not
        exist it will be created. Behavior is modified by `abs_path`.

    Returns
    -------
    cache_dir : str
        The path to the directory to store data.

    See also
    --------
    `set_cache_dir`, `default_data_dir`
    """

    if pkg_data_dir:
        data_dir = pkg_data_dir
    else:
        data_dir = appdirs.user_data_dir(pkg_name, _APPAUTHOR)

    data_dir = os.path.join(data_dir, path)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir, mode=0o755)

    return data_dir


def _dir_exists(path: str) -> bool:
    """default directory commands create a directory so test if the directory
    both exists and is not empty. If empty delete and return non-existent."""

    try:
        os.rmdir(path)
    except OSError:
        return True

    return False


def _clear_dir(path: str) -> None:
    if not _dir_exists(path):
        raise NotADirectoryError("Path does not exist")

    for f_name in os.listdir(path):
        f_path = os.path.join(path, f_name)
        if os.path.isdir(f_path):
            _clear_dir(f_path)
        else:
            os.unlink(f_path)

    os.rmdir(path)


def clear_cache(path: str = "") -> None:
    """Clear a cache directory

    By default clears all data cached by this package.

    Parameters
    ----------
    path : str
        If not an empty string (default), clear only the directory PATH
        relative to the default cache.
    """

    _clear_dir(default_cache_dir(path))


def clear_data(path: str = "") -> None:
    """Clear a data directory

    By default clears all data saved by this package.

    Parameters
    ----------
    path : str
        If not an empty string (default), clear only the directory PATH
        relative to the default data directory.
    """

    _clear_dir(default_data_dir(path))


def list_graphs(data_dir: Optional[str] = None) -> List[str]:
    """List all graphs saved in `data_dir`"""

    if not data_dir:
        data_dir = default_data_dir()

    return os.listdir(data_dir)


def delete_graph(graph_name: str, data_dir: Optional[str] = None) -> None:
    """Delete the graph from `data_dir`"""

    def delete_directory(path):
        for f in os.listdir(path):
            os.unlink(os.path.join(path, f))
        os.rmdir(path)

    if data_dir:
        path = os.path.join(data_dir, graph_name)
    else:
        path = default_data_dir(graph_name)

    if os.path.isdir(path):
        delete_directory(path)
    else:
        raise NotADirectoryError(
            f"{graph_name} not found in {default_data_dir()}"
        )
