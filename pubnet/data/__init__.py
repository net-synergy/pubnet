"""
Methods for storing and generating publication networks.

Storage
-------
Graphs are stored as directories with files for each node and edge. By default,
graphs are stored in the `default_data_dir`. Files can be saved in multiple
forms including plain text--which is easy to modify by hand or with tools
outside of this package, but can be slow in inefficiently stored--or as
compressed binary types--that can't be easily modified by are significantly
faster for larger data and take up less storage space. Additionally, in the
future, graphs DBs may be supported.

Generation
----------
Publication networks can be generated from common data sources, including
pubmed and crossref. New data gets downloaded to the `default_cache_dir`.
"""

from ._utils import default_cache_dir, default_data_dir, delete, list

__all__ = ["default_cache_dir", "default_data_dir", "delete", "list"]
