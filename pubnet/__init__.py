"""Publication Network data structure.

This package provides the PubNet data structure for storing
publication data as a set of graphs. Graphs are represented as nodes
and edges (adjacency matrices).
"""

from pubnet.network import from_data, from_dir

__all__ = ["from_dir", "from_data"]
