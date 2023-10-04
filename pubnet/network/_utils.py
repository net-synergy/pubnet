"""Helper functions for writing publication network functions."""

import os
import re
from typing import cast

__all__ = [
    "node_file_parts",
    "node_gen_file_name",
    "node_find_file",
    "node_list_files",
    "node_files_containing",
    "node_gen_id_label",
    "node_id_label_parts",
    "edge_gen_file_name",
    "edge_key",
    "edge_parts",
    "edge_file_parts",
    "edge_find_file",
    "edge_files_containing",
    "edge_gen_header",
    "edge_header_parts",
]

NODE_PATH_REGEX = re.compile(r"(?P<node>\w+)_nodes.(?P<ext>[\w\.]+)")
EDGE_PATH_REGEX = re.compile(r"(?P<n1>\w+)_(?P<n2>\w+)_edges.(?P<ext>[\w\.]+)")
EDGE_KEY_DELIM = "-"


def edge_key(node_1: str, node_2: str) -> str:
    """Generate a dictionary key for the given pair of nodes.

    Known future issue:
        If we need directed edges, the order of nodes in the file name
        may be important. Add in a directed keyword argument, if true
        look for files only with the nodes in the order they were
        provided otherwise look for both. Another option is to not
        only check the file name but check the header for the START_ID
        and END_ID node types.

    See also
    --------
    `edge_parts` for going in the other direction.
    """
    return EDGE_KEY_DELIM.join(sorted((node_1, node_2)))


def edge_parts(key: str | tuple[str, str]) -> tuple[str, str]:
    """Break an edge key into its nodes

    See also
    --------
    `edge_key` for going in the other direction.
    """

    # Often gets called in places where an edge could be described as a tuple
    # of nodes or a key, so if already in tuple form, harmlessly pass back.
    if isinstance(key, tuple):
        return key

    parts = key.split(EDGE_KEY_DELIM)
    if len(parts) != 2:
        raise ValueError(
            f"{parts} has wrong number of parts. Key should have exactly one"
            f' "{EDGE_KEY_DELIM}".'
        )

    return cast(tuple[str, str], tuple(sorted(parts)))


def edge_file_parts(file_name: str) -> tuple[str, str]:
    """Return the edge name and its file extension.

    Assumes the convention f\"{node}_{node_or_type}_edges.{ext}\".

    Parameters
    ----------
    file_name : str
        The name of the file.
    """

    name_matches = re.search(EDGE_PATH_REGEX, file_name)

    if name_matches is None:
        raise NameError("File name does not match naming conventions.")

    name_parts = name_matches.groupdict()

    return (edge_key(name_parts["n1"], name_parts["n2"]), name_parts["ext"])


def node_file_parts(file_name: str):
    """Return the edge name and its file extension.

    Assumes the convention f\"{node}_nodes.{ext}\".

    Parameters
    ----------
    file_name : str
        The name of the file.
    """

    name_parts = re.search(NODE_PATH_REGEX, file_name)

    if name_parts is None:
        raise NameError("File name does not match naming conventions.")

    return name_parts.groups()


def node_gen_file_name(node: str, ext: str, data_dir: str) -> str:
    """Create the path to a file the given node can be saved to."""
    return os.path.join(data_dir, f"{node}_nodes.{ext}")


def edge_gen_file_name(edge: str, ext: str, data_dir: str) -> tuple[str, str]:
    """Create the path to a file the given edge can be saved to."""

    n1, n2 = edge_parts(edge)
    data_path = os.path.join(data_dir, f"{n1}_{n2}_edges.{ext}")
    header_path = os.path.join(data_dir, f"{n1}_{n2}_edge_header.tsv")
    return (data_path, header_path)


def node_list_files(data_dir: str) -> dict[str, dict[str, str]]:
    """Return all node files in the data_dir

    Returns a dictionary with nodes as keys and a dictionary with extension as
    keys and file paths as values as values.

    example: path_dict['Author']['tsv'] = file_path
    """

    files = os.listdir(data_dir)
    node_files = [
        (m.groupdict(), os.path.join(data_dir, m.group()))
        for m in (re.search(NODE_PATH_REGEX, f) for f in files)
        if m is not None
    ]
    nodes = {n[0]["node"] for n in node_files}
    out: dict[str, dict[str, str]] = {}
    for n in nodes:
        out[n] = {f[0]["ext"]: f[1] for f in node_files if f[0]["node"] == n}

    return out


def node_find_file(node: str, path_dict: dict[str, dict[str, str]]) -> str:
    """Return the file path for a node."""

    try:
        available_files = path_dict[node]
    except KeyError:
        raise FileNotFoundError(f"No file found for node {node}.")

    ext_preference = ["feather", "tsv", "tsv.gz"]
    for ext in ext_preference:
        try:
            return available_files[ext]
        except KeyError:
            continue

    raise FileNotFoundError(
        f"No file found for node {node} with a supported file extension."
    )


def edge_list_files(data_dir: str) -> dict[str, dict[str, str]]:
    """Return a dictionary of dictionaries of files paths for each edge.

    Outer dictionary has edge keys as keys and inner dictionary has extension
    as keys.

    Example: path_dict['Author-Publication']['tsv'] = file_path
    """

    files = os.listdir(data_dir)
    edge_files = [
        (m.groupdict(), os.path.join(data_dir, m.group()))
        for m in (re.search(EDGE_PATH_REGEX, f) for f in files)
        if m is not None
    ]
    edges = {edge_key(e[0]["n1"], e[0]["n2"]) for e in edge_files}
    out: dict[str, dict[str, str]] = {}
    for e in edges:
        out[e] = {
            f[0]["ext"]: f[1]
            for f in edge_files
            if edge_key(f[0]["n1"], f[0]["n2"]) == e
        }

    return out


def edge_find_file(
    n1: str, n2: str, path_dict: dict[str, dict[str, str]]
) -> str:
    """Find the edge file in data_dir for the provided node types."""

    edge = edge_key(n1, n2)
    try:
        available_files = path_dict[edge]
    except KeyError:
        raise FileNotFoundError(f"No edge file found for nodes {n1}, {n2}.")

    ext_preference = ["npy", "tsv", "tsv.gz", "pickle"]
    for ext in ext_preference:
        try:
            return available_files[ext]
        except KeyError:
            continue

    raise FileNotFoundError(
        f"No file found for nodes {n1}, {n2} with a supported file extension."
    )


def node_files_containing(
    nodes: str | tuple[str, ...], data_dir: str
) -> dict[str, str]:
    """Find the preferred node file for the provided nodes in data_dir

    If nodes is \"all\" find a file for all nodes in the data_dir, otherwise
    only find files for nodes in the requested list.

    If a node is requested but no file is found, an error will be raised.

    Preferred file is based on the extension. Extension preference can be seen
    in `node_find_file`.
    """

    all_node_files = node_list_files(data_dir)
    if nodes == "all":
        nodes = tuple(all_node_files.keys())

    return {n: node_find_file(n, all_node_files) for n in nodes}


def edge_files_containing(
    nodes: str | tuple[str, ...], data_dir: str
) -> dict[str, str]:
    """Find the preferred edge file for the provided nodes in data_dir

    If nodes is \"all\" find a file for all nodes in the data_dir, otherwise
    only find files for nodes in the requested list. This means all edge files
    linking pairs of node types, where both node types are in the supplied
    list.

    Preferred file is based on the extension. Extension preference can be seen
    in `edge_find_file`.
    """

    all_edge_files = edge_list_files(data_dir)
    if isinstance(nodes, str):
        assert nodes == "all", ValueError(nodes)
        edges = tuple(all_edge_files.keys())
    else:
        edges = tuple(
            edge_key(n1, n2)
            for i, n1 in enumerate(nodes)
            for n2 in nodes[i:]
            if edge_key(cast(str, n1), cast(str, n2)) in all_edge_files.keys()
        )

    return {e: edge_find_file(*edge_parts(e), all_edge_files) for e in edges}


def node_gen_id_label(id: str, namespace: str) -> str:
    return f"{id}:ID({namespace})"


def node_id_label_parts(label: str) -> tuple[str, str]:
    pattern = r"(?P<id>\w+):ID\((?P<namespace>\w+)\)"
    match = re.search(pattern, label)

    if match is None:
        raise ValueError(f"{label} does not match label naming convention.")

    id = match.groupdict()["id"]
    namespace = match.groupdict()["namespace"]

    return (id, namespace)


def edge_gen_header(start_id: str, end_id: str, features: list[str]) -> str:
    feat_header = "\t" + "\t".join(features) if features else ""

    return f":START_ID({start_id})\t:END_ID({end_id}){feat_header}"


def edge_header_parts(header: str) -> tuple[str, str, list[str], bool]:
    """Parse the header for column names.

    Returns
    -------
    start_id, end_id : str
        The node namespace for the start and end of the edges.
    features : list[str]
        A (possibly empty) list of feature names.
    reverse : bool
        Whether the start and end ids have been reversed. That is if the first
        column is end and the second column is start.
    """
    ids = re.findall(r":((?:START)|(?:END))_ID\((\w+)\)", header)
    for id, node in ids:
        if id == "START":
            start_id = node
        elif id == "END":
            end_id = node

    reverse = ids[0][0] == "END"

    features = [
        feat
        for feat in re.findall(r"([\w:()]+)", header)
        if not (feat.startswith(":START") or feat.startswith(":END"))
    ]
    return (start_id, end_id, features, reverse)
