import os
import sys

from pubnet import __name__ as pkg_name

__all__ = ["default_cache_dir", "default_data_dir", "delete", "list"]


def default_cache_dir():
    """Find the default location to save cache files.

    Cache files are specifically files that can be easily reproduced,
    i.e. those that can be downloaded from the internet.
    """

    if sys.platform.startswith("win"):
        try:
            cache_home = os.environ["LOCALAPPDATA"]
        except KeyError as err:
            raise EnvironmentError(
                "Location for local app data is not set.",
                "Explicitely set cache directory to get around error.",
            ) from err
    else:
        try:
            cache_home = os.environ["XDG_CACHE_HOME"]
        except KeyError:
            home = os.environ["HOME"]
            return os.path.join(home, pkg_name, "cache")

    return os.path.join(cache_home, pkg_name)


def default_data_dir():
    """Find the default location to save data files.

    Data files are files created by a user. It's possible they can be
    reproduced by rerunning the script that produced them but there is
    no gurentee they can be perfectly reproduced.
    """

    if sys.platform.startswith("win"):
        try:
            data_home = os.environ["APPDATA"]
        except KeyError as err:
            raise EnvironmentError(
                "Location for app data is not set.",
                "Explicitely set cache directory to get around error.",
            ) from err
    else:
        try:
            data_home = os.environ["XDG_DATA_HOME"]
        except KeyError:
            home = os.environ["HOME"]
            return os.path.join(home, pkg_name, "share")

    return os.path.join(data_home, pkg_name)


def list(data_dir=default_data_dir()):
    """List all graphs saved in `data_dir`"""

    return os.listdir(data_dir)


def delete(graph_name, data_dir=default_data_dir()):
    """Delete the graph in data_dir"""

    def delete_directory(path):
        for f in os.listdir(path):
            os.unlink(os.path.join(path, f))
        os.rmdir(path)

    path = os.path.join(data_dir, graph_name)
    if os.path.isdir(path):
        delete_directory(path)
    else:
        raise NotADirectoryError(f"{graph_name} not found in {data_dir}")
