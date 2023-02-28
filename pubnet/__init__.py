"""Publication Network data structure.

This package provides the PubNet data structure for storing publication data as
a set of graphs. Graphs are represented as a list of nodes and edges.

New publication networks can be generated using the data module to interact
with publication data sources such as crossref and pubmed.

The PubNet object provides tools for reading graphs into memory, saving graphs
to disc, filtering graphs, modifying graph content (including creating new node
and edge types), selecting a node or edge from a graph, and running graph
algorithms on selected edges or nodes.
"""

from pubnet.network import from_data, from_dir

__all__ = ["from_dir", "from_data"]
