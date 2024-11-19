# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Ability to save and restore numpy edge features.
- Generate edge name from node type if name not explicitly passed.
- Method to reset all node indices in the network.
- Edge attribute for `isbipartite`.

### Changed

- Don't count empty directories when listing saved graphs.
- Rename specter argument.
- JIT compile specter model for performance improvement.
- Default to resetting the indices before saving the network.
- Add option to explicitly set whether edge is directed or not when creating it.

### Fixed

- Test for if graph exists when loading was using `os.path.exists` which would always succeed since `graph_path` creates an empty directory.
- Set the directed flag when creating igraph graphs.
- Was assuming edges are undirected when start id differs from end id when that guarantees they are directed.
- Compose should only require mode if node type's are the same not if the graph is directed.
- Prevent unmatched quotation marks from messing up reading in tables.

## [0.9.0] - 2024-11-05

### Added

- `from_pubnet` method.
- Ability to select graph components with edge wildcard.
- Mutate node methods to create new nodes from existing nodes.
- Sanitize methods.
- Index repacking methods.
- Language model methods.
- Overwrite option to `from_pubmed`.
- To pandas method for nodes.

### Changed

- Align sizes in PubNet representation.
- Don't error if expected node file is missing, instead create empty node.
- Ensure all nodes have name and ID.
- Add methods for getting and setting various names associated with a node.
- Default date field in pubmed XML.
- Create publication indices instead of using PMIDs.

### Fixed

- When slicing a network, only slice edges that are connected to the network's root.
- Handle nodes with no edges after PubNet indexing (return empty node set).
- Order of the last edges when printing was backward.
- Missing names in call to `from_data` in `add_edge`.
- Handle empty edges when calculating overlap.
- Typo in pubmed path for Grant IDs.
- Protect against overwriting graphs when creating graphs from pubmed.
- Prevent duplicate edges when creating pubmed graphs.

## [0.8.1] - 2023-10-06

### Added

- A Change log
- Reduction methods for combining edge sets.
- Limited directed graph functionality.
- Re-rooting using composing edges so edges starting with the old root now use the new root.
- Composing edges (A--C from A--B + B--C).
- Network level overlap based off the root node.
- Network-wide conversion between edge backends.
- Edge features.

### Changed

- Edges and nodes are know tracked as sets.
- Linters and formatters and reformat files.

### Fixed

- Distribution plot (broken by changes to how indices are tracked).
- Printing edge set IDs (when the set had a float type feature it).
- Issue when trying to use anything dependent on `ids_where` with the root
node.
- Issue when trying to use a node index for filtering.

## [0.8.0] - 2023-10-04

### Added

- Get node and get edge methods to reduce the load of `__getitem__`
- Select root method to change the network's root.
- Storage module handles more decisions about where data is saved to ensure consistency throughout package and simplify function parameters.

### Changed

- Have edge overlap return an edge instead of an array.
- Give Network, Edge, and Node types name attribute to reduce manually passing names.
- Set use the node IDs for Indices in the node DataFrames instead of duplicating IDs and slight differences between the two.
- Node loaders are now class methods of the Node type instead of free functions.
- Edge representation are manually aligned instead of relying on tabs.
- Edge types set method -> set_data to make it clearer.
- `to_dir` and `from_dir` renamed to `save_graph` and `load_graph`
- `data` module -> `storage` module.

### Deprecated

- `to_dir` and `from_dir`.
- `data` module.
