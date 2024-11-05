"""Functions for working with pubmed data.

Find, download, and parse pubmed files, to generate PubNet objects.
"""

__all__ = ["from_pubmed", "list_pubmed_files", "available_paths"]

import os
import re
import shutil
from contextlib import suppress
from typing import Optional

import pubmedparser
import pubmedparser.ftp

import pubnet.storage as storage
from pubnet import PubNet
from pubnet.network._utils import (
    edge_gen_file_name,
    edge_gen_header,
    edge_key,
    node_gen_file_name,
    node_gen_id_label,
)
from pubnet.storage import graph_path

from ._pubmed_paths import (
    available_paths,
    expand_structure_dict,
    is_node_list,
    node_list_to_file_names,
    sterilize_node_list,
)

list_pubmed_files = pubmedparser.ftp.list_files


def _exists_locally(
    graph_path: str,
    node_list,
    file_numbers: str | int | list[int] | range,
) -> bool:
    """Determine if a network has already been downloaded.

    Tests if a graph exists at the path that meets the specifications based on
    the desired pubmed files and path structure.
    """
    # TODO: Intend to change how pubmedparser returns results to make it easier
    # to distinguish which pubmed xml files data came from to allow reusing
    # network data for creating pubnets with different subsets of the original
    # data. Should also be able to determine if the correct data was collected.
    # And be able to parse only files that haven't yet been parsed and then
    # update the given network with that new data.
    #
    # Until then use `from_pubnet` to generate the graph then
    # `PubNet.load_graph` to read that graph from disk later.
    return False

    if not os.path.exists(graph_path):
        return False

    previous_source_list = os.path.join(graph_path, "source_files.txt")
    if not os.path.exists(previous_source_list):
        return False

    with open(previous_source_list, "r") as f:
        previous_source_files = "".join(f.readlines())

    pubmed_file_regex = re.compile(r"pubmed\d{2}n(\d{4})\.xml\.gz")
    saved_file_numbers = {
        int(n) for n in re.findall(pubmed_file_regex, previous_source_files)
    }

    if (
        # Consider adding update logic later instead recreating the graph from
        # scratch.
        isinstance(file_numbers, str)
        or (
            isinstance(file_numbers, int)
            and file_numbers not in saved_file_numbers
        )
        or (
            not isinstance(file_numbers, int)
            and set(saved_file_numbers) != set(file_numbers)
        )
    ):
        return False

    # FIXME: node_list_to_file_names is broken.
    # expceted_files = node_list_to_file_names(node_list, "Publication", "")
    expected_files = {}

    return set(expected_files) != set(os.listdir(graph_path))


class _Index:
    def __init__(self):
        self.count = 0
        self.ids = {}

    def add(self, value):
        self.ids[value] = self.count
        self.count += 1

    def __contains__(self, value):
        return value in self.ids

    def __getitem__(self, value):
        return self.ids[value]


def _convert_key(
    key: str,
    raw_data_dir: str,
    graph_dir: str,
    clean_cache: bool,
) -> _Index:
    key_index = _Index()

    original_file = os.path.join(raw_data_dir, key + ".tsv")
    node_file = node_gen_file_name(key, "tsv", graph_dir)
    with open(original_file, "r") as raw_ptr, open(node_file, "w") as node_ptr:
        header = raw_ptr.readline()
        node_ptr.write(node_gen_id_label(key + "ID", key) + "\t" + header)
        for line in raw_ptr:
            parts = line.strip().split("\t")

            if parts[0] not in key_index:
                node_ptr.write(
                    str(key_index.count)
                    + "\t"
                    + "\t".join(parts).lower()
                    + "\n"
                )
                key_index.add(parts[0])

    if clean_cache:
        os.unlink(original_file)

    return key_index


def _convert_relational_group(
    nodes: list[str],
    net_key: str,
    group_key: str,
    key_index: _Index,
    raw_data_dir: str,
    graph_dir: str,
    clean_cache: bool,
) -> None:
    group_index = _Index()
    group_file = edge_gen_file_name(
        edge_key(net_key, group_key), "tsv", graph_dir
    )[0]

    with open(group_file, "w") as group_ptr:
        group_ptr.write(edge_gen_header(net_key, group_key, []) + "\n")
        for n in nodes:
            node_index = _Index()
            edge_index = _Index()
            name = group_key + "_" + n
            original_file = os.path.join(raw_data_dir, name + ".tsv")
            node_file = node_gen_file_name(name, "tsv", graph_dir)
            edge_file = edge_gen_file_name(
                edge_key(n, group_key), "tsv", graph_dir
            )[0]
            with open(original_file, "r") as raw_ptr, open(
                node_file, "w"
            ) as node_ptr, open(edge_file, "w") as edge_ptr:
                header = raw_ptr.readline().split("\t")
                node_ptr.write(
                    node_gen_id_label(n + "ID", n)
                    + "\t"
                    + "\t".join(header[2:])
                )
                edge_ptr.write(edge_gen_header(group_key, n, []) + "\n")

                for line in raw_ptr:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue

                    group_label = "-".join(parts[:2])
                    if group_label not in group_index:
                        group_ptr.write(
                            str(key_index[parts[0]])
                            + "\t"
                            + str(group_index.count)
                            + "\n"
                        )
                        group_index.add(group_label)

                    if parts[2] not in node_index:
                        node_ptr.write(
                            str(node_index.count)
                            + "\t"
                            + "\t".join(parts[2:]).lower()
                            + "\n"
                        )
                        node_index.add(parts[2])

                    # A given author can somehow have multiple last names and
                    # other fields that there should only be one of.
                    if group_label not in edge_index:
                        edge_ptr.write(
                            f"{group_index[group_label]}\t"
                            + f"{node_index[parts[2]]}\n"
                        )
                        edge_index.add(group_label)

            if clean_cache:
                os.unlink(original_file)


def _convert_file(
    node: str,
    key: str,
    key_index: _Index,
    raw_data_dir: str,
    graph_dir: str,
    clean_cache: bool,
) -> None:
    node_index = _Index()

    original_file = os.path.join(raw_data_dir, node + ".tsv")
    node_file = node_gen_file_name(node, "tsv", graph_dir)
    edge_file = edge_gen_file_name(edge_key(node, key), "tsv", graph_dir)[0]
    with open(original_file, "r") as raw_ptr, open(
        node_file, "w"
    ) as node_ptr, open(edge_file, "w") as edge_ptr:
        header = raw_ptr.readline().split("\t")
        node_ptr.write(
            node_gen_id_label(node + "ID", node) + "\t" + "\t".join(header[1:])
        )
        edge_ptr.write(edge_gen_header(key, node, []) + "\n")
        for line in raw_ptr:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            if parts[1] not in node_index:
                node_ptr.write(
                    str(node_index.count)
                    + "\t"
                    + "\t".join(parts[1:]).lower()
                    + "\n"
                )
                node_index.add(parts[1])

            edge_ptr.write(f"{key_index[parts[0]]}\t{node_index[parts[1]]}\n")

    if clean_cache:
        os.unlink(original_file)


def _to_graph(
    key_node: str,
    node_list,
    raw_data_dir: str,
    graph_dir: str,
    clean_cache: bool,
) -> None:
    key_index = _convert_key(key_node, raw_data_dir, graph_dir, clean_cache)

    for node in node_list:
        if (isinstance(node, dict) and node["name"] == key_node) or (
            isinstance(node, str) and node == key_node
        ):
            continue

        if (
            isinstance(node, dict)
            and "grouping" in node
            and node["grouping"] == "relational"
        ):
            _convert_relational_group(
                node["value"],
                key_node,
                node["name"],
                key_index,
                raw_data_dir,
                graph_dir,
                clean_cache,
            )
        else:
            _convert_file(
                node["name"] if isinstance(node, dict) else node,
                key_node,
                key_index,
                raw_data_dir,
                graph_dir,
                clean_cache,
            )


def from_pubmed(
    file_numbers: str | int | list[int] | range,
    node_list,
    graph_name: str,
    data_dir: Optional[str] = None,
    load_graph: bool = True,
    clean_cache: bool = True,
    overwrite: bool = False,
) -> PubNet | None:
    """Create a PubNet object from pubmed data.

    Parameters
    ----------
    file_numbers : str, int, list[int], range
       The numbers of the files to download from pubmed. Values are passed
       directly to `pubmedparser.ftp.download`. If "all", processes all
       files---this is a long process that will download a upwards of 10sGB of
       data.
    node_list : list
       A list of the nodes to grab from the downloaded pubmed XML files. For a
       list of available nodes, see `download.pubmed.available_paths`. For
       nodes not in the predefined available paths, a dictionary can be used
       to specify the path. A dictionary can also be used to rename a node and
       to create a relational or condensed group. Relational groups act as a
       subgraph, where the nodes within the group will have links to each
       other and the group has links to the publication IDs.

       example:

       node_list = [
           # Rename the date value publication.
           {"name": "publication", "value": "date"},
           {   # Collect several author attributes as a subgraph.
               "name": "author",
               "value": ["last_name", "fore_name", "affiliation"],
               "grouping": "relational",
           },
           # Get Publication chemicals and keywords.
           "chemical",
           "keyword"
       ]
    graph_name : str
       The name to give the graph, used for future loading and saving actions.
    data_dir : str, optional
       Where to save the graph. Defaults to the `default_data_dir`.
    load_graph : bool, default True
       Whether to load the graph as a PubNet object or just save the files
       to disk.
    clean_cache : bool, default True
       Whether to clear the raw pubmedparser files after creating the graph.
       The cleared files are not required for reading the graph later. Should
       leave this True unless there's a good reason to turn it off, left over
       files could mess up future calls to the function.
    overwrite : bool, default False
       Whether to write over a preexisting graph with the same name or not. If
       True, a new graph will always be created. If False, will check if a
       graph with the requested name already exists, if there is a graph it
       will check if the graph on disk is the same as that being requested and
       will return that if so. If there is a graph at the given location and it
       is not the same as requested, an error will be raised.

    Returns
    -------
    network : PubNet, None
        If `load_graph` returns a PubNet network containing the pubmed data. If
        not `load_graph` does not return anything.

    """
    if not is_node_list(node_list):
        raise TypeError("Node list does not match expected format.")

    node_list = sterilize_node_list(node_list)
    publication_struct = expand_structure_dict(node_list)
    save_dir = graph_path(graph_name)

    if graph_name in storage.list_graphs():
        if overwrite:
            storage.delete_graph(graph_name)
        elif _exists_locally(save_dir, node_list, file_numbers):
            return PubNet.load_graph(save_dir) if load_graph else None
        else:
            raise FileExistsError(
                "A graph with the requested name already exists."
            )

    files = pubmedparser.ftp.download(file_numbers)
    raw_data = pubmedparser.read_xml(files, publication_struct, "pubnet")

    shutil.copy(
        os.path.join(raw_data, "processed.txt"),
        os.path.join(save_dir, "source_files.txt"),
    )

    _to_graph(
        "Publication", node_list, raw_data, save_dir, clean_cache=clean_cache
    )
    if clean_cache:
        os.unlink(os.path.join(raw_data, "processed.txt"))
        with suppress(OSError):
            os.rmdir(raw_data)

    if load_graph:
        return PubNet.load_graph(graph_name)

    return None
