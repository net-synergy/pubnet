import os

import numpy as np

import pubnet

from pubnet import from_dir

from ._utils import simple_pubnet, mktmpdir

LOG_NUM_NODE_RANGE = (1, 5)
PARAMS = (("numpy","igraph"), 10 ** np.arange(*LOG_NUM_NODE_RANGE))
PARAM_NAMES = ("Representation", "n_nodes")
IO_PARAMS = PARAMS + (("node", "edge", "graph"), ("tsv","gzip"))
IO_PARAM_NAMES = PARAM_NAMES + ("scope", "formats")


class TimeEdges:
    params = PARAMS
    param_names = PARAM_NAMES
    timeout = 480

    def setup(self, representation, n_nodes):
        self.simple_pubnet = simple_pubnet(representation, n_nodes)

    def time_finds_start_id(self, *args):
        for e in self.simple_pubnet.edges:
            self.simple_pubnet[e].start_id

    def time_finds_end_id(self, *args):
        for e in self.simple_pubnet.edges:
            self.simple_pubnet[e].end_id

    def time_overlap(self, *args):
        self.simple_pubnet["Author", "Publication"].overlap


class TimeNodes:
    params = PARAMS
    param_names = PARAM_NAMES

    def setup(self, representation, n_nodes):
        self.simple_pubnet = simple_pubnet(representation, n_nodes)

    def time_finds_namespace(self, *args):
        self.simple_pubnet["Author"].id == "AuthorId"

    def time_slice_column(self, *args):
        self.simple_pubnet["Author"]["LastName"]

    def time_slice_columns(self, *args):
        features = ["LastName", "ForeName"]
        self.simple_pubnet["Author"][features].columns.values

    def time_slice_column_by_index(self, *args):
        self.simple_pubnet["Author"][0] is self.simple_pubnet["Author"][
            self.simple_pubnet["Author"].features[0]
        ]
        self.simple_pubnet["Author"][1] is self.simple_pubnet["Author"][
            self.simple_pubnet["Author"].features[1]
        ]

    def time_slice_rows_by_index(self, *args):
        actual = self.simple_pubnet["Author"][0:2]

    def time_slice_rows_by_mask(self, *args):
        self.simple_pubnet["Author"][
            self.simple_pubnet["Author"]["LastName"] == "Smith"
        ]

    def time_slice_rows_and_columns(self, *args):
        actual = {
            "Slices": self.simple_pubnet["Author"][0:2, 0:2],
            "Slice + List": self.simple_pubnet["Author"][
                0:2, ["AuthorId", "LastName"]
            ],
            "Mask + Slice": self.simple_pubnet["Author"][
                self.simple_pubnet["Author"]["ForeName"] == "John", 0:2
            ],
        }


class TimeNetwork:
    params = PARAMS
    param_names = PARAM_NAMES

    def setup(self, representation, n_nodes):
        self.simple_pubnet = simple_pubnet(representation, n_nodes)

        working_dir = os.path.dirname(__file__)
        data_dir = os.path.join(
            working_dir[0 : working_dir.rindex("/")], "tests/data"
        )
        self.other_pubnet = pubnet.from_dir(
            graph_name="simple_pubnet",
            nodes=("Author", "Publication"),
            edges=(("Publication", "Author"),),
            data_dir=data_dir,
            representation=representation,
        )

    def time_creates_empty_nodes_for_missing_edge_nodes(self, *args):
        len(self.simple_pubnet["Chemical"]) == 0

    def time_filter_to_single_publication_id(self, *args):
        publication_id = 1
        subnet = self.simple_pubnet[publication_id]

        subnet["Author"][subnet["Author"].id]

        subnet["Publication"][subnet["Publication"].id]

    def time_filter_to_publicaiton_ids(self, *args):
        publication_ids = np.asarray([1, 2], dtype=self.simple_pubnet.id_dtype)
        subnet = self.simple_pubnet[publication_ids]

        subnet["Author"][subnet["Author"].id]

        subnet["Publication"][subnet["Publication"].id]

    def time_filter_twice(self, *args):
        publication_ids_1 = np.asarray(
            [4, 6], dtype=self.simple_pubnet.id_dtype
        )
        publication_ids_2 = 4

        subnet_1 = self.simple_pubnet[publication_ids_1]
        subsubnet = subnet_1[publication_ids_2]
        subnet_2 = self.simple_pubnet[publication_ids_2]

        subsubnet.isequal(subnet_2)

    def time_filter_to_author(self, *args):
        random_author = self.simple_pubnet["Author"].get_random(1).LastName
        subnet = self.simple_pubnet.containing(
            "Author", "LastName", random_author
        )

    def time_filter_to_author_multiple_steps(self, *args):
        random_author = self.simple_pubnet["Author"].get_random(1).LastName
        publication_ids = self.simple_pubnet.ids_containing(
            "Author", "LastName", random_author, steps=2
        )
        subnet = self.simple_pubnet[publication_ids]

        subnet["Author", "Publication"]["Publication"]
        subnet["Chemical", "Publication"]["Publication"]

    def time_update(self, *args):
        expected_nodes = set(self.simple_pubnet.nodes).union(
            set(self.other_pubnet.nodes)
        )

        expected_edges = set(self.simple_pubnet.edges).union(
            set(self.other_pubnet.edges)
        )

        self.simple_pubnet.update(self.other_pubnet)

class TimeReadIO:
    params = IO_PARAMS
    param_names = IO_PARAM_NAMES

    def setup(self, representation, n_nodes, scope,formats):
        self.graph_name = f"{representation}_{n_nodes}"
        self._data_dir_obj = mktmpdir()
        self.data_dir = self._data_dir_obj.name
        self.simple_pubnet = simple_pubnet(representation, n_nodes)

        nodes = None
        edges = None
        if scope in ("node", "graph"):
            nodes = ("Publication", "Author")
        if scope in ("edge", "graph"):
            edges = (("Author", "Publication"),)

        self.simple_pubnet.to_dir(
            self.graph_name,
            nodes=nodes,
            edges=edges,
            data_dir=self.data_dir,
            format=formats,
        )

    def teardown(self, *args):
        self._data_dir_obj.cleanup()

    def time_read_graph(self, representation, *args):
        new = from_dir(
            graph_name=self.graph_name,
            data_dir=self.data_dir,
            representation=representation,
        )


class TimeWriteIO:
    params = IO_PARAMS
    param_names = IO_PARAM_NAMES

    def setup(self, representation, n_nodes, scope,formats):
        self.graph_name = f"{representation}_{n_nodes}"
        self._data_dir_obj = mktmpdir()
        self.data_dir = self._data_dir_obj.name
        self.simple_pubnet = simple_pubnet("igraph", n_nodes)

        nodes = None
        edges = None
        if scope in ("node", "graph"):
            nodes = ("Publication", "Author")
        if scope in ("edge", "graph"):
            edges = (("Author", "Publication"),)

        self.nodes = nodes
        self.edges = edges

    def teardown(self, *args):
        self._data_dir_obj.cleanup()

    def time_write_graph(self, representation, n_nodes, scope, formats):
        self.simple_pubnet.to_dir(
            self.graph_name,
            nodes=self.nodes,
            edges=self.edges,
            data_dir=self.data_dir,
            format=formats,
        )

class MemEdges:
    params = PARAMS
    param_names = PARAM_NAMES

    def setup(self, representation, n_nodes):
        self.simple_pubnet = simple_pubnet(representation, n_nodes)

    def peakmem_finds_start_id(self, *args):
        for e in self.simple_pubnet.edges:
            self.simple_pubnet[e].start_id

    def peakmem_finds_end_id(self, *args):
        for e in self.simple_pubnet.edges:
            self.simple_pubnet[e].end_id

    def peakmem_overlap(self, *args):
        self.simple_pubnet["Author", "Publication"].overlap


class MemNodes:
    params = PARAMS
    param_names = PARAM_NAMES
    timeout = 480

    def setup(self, representation, n_nodes):
        self.simple_pubnet = simple_pubnet(representation, n_nodes)

    def peakmem_finds_namespace(self, *args):
        self.simple_pubnet["Author"].id == "AuthorId"

    def peakmem_slice_column(self, *args):
        self.simple_pubnet["Author"]["LastName"]

    def peakmem_slice_columns(self, *args):
        features = ["LastName", "ForeName"]
        self.simple_pubnet["Author"][features].columns.values

    def peakmem_slice_column_by_index(self, *args):
        self.simple_pubnet["Author"][0] is self.simple_pubnet["Author"][
            self.simple_pubnet["Author"].features[0]
        ]
        self.simple_pubnet["Author"][1] is self.simple_pubnet["Author"][
            self.simple_pubnet["Author"].features[1]
        ]

    def peakmem_slice_rows_by_index(self, *args):
        actual = self.simple_pubnet["Author"][0:2]

    def peakmem_slice_rows_by_mask(self, *args):
        self.simple_pubnet["Author"][
            self.simple_pubnet["Author"]["LastName"] == "Smith"
        ]

    def peakmem_slice_rows_and_columns(self, *args):
        actual = {
            "Slices": self.simple_pubnet["Author"][0:2, 0:2],
            "Slice + List": self.simple_pubnet["Author"][
                0:2, ["AuthorId", "LastName"]
            ],
            "Mask + Slice": self.simple_pubnet["Author"][
                self.simple_pubnet["Author"]["ForeName"] == "John", 0:2
            ],
        }


class MemNetwork:
    params = PARAMS
    param_names = PARAM_NAMES

    def setup(self, representation, n_nodes):
        self.simple_pubnet = simple_pubnet(representation, n_nodes)

        working_dir = os.path.dirname(__file__)
        data_dir = os.path.join(
            working_dir[0 : working_dir.rindex("/")], "tests/data"
        )
        self.other_pubnet = pubnet.from_dir(
            graph_name="simple_pubnet",
            nodes=("Author", "Publication"),
            edges=(("Publication", "Author"),),
            data_dir=data_dir,
            representation=representation,
        )

    def peakmem_creates_empty_nodes_for_missing_edge_nodes(self, *args):
        len(self.simple_pubnet["Chemical"]) == 0

    def peakmem_filter_to_single_publication_id(self, *args):
        publication_id = 1
        subnet = self.simple_pubnet[publication_id]

        subnet["Author"][subnet["Author"].id]

        subnet["Publication"][subnet["Publication"].id]

    def peakmem_filter_to_publicaiton_ids(self, *args):
        publication_ids = np.asarray([1, 2], dtype=self.simple_pubnet.id_dtype)
        subnet = self.simple_pubnet[publication_ids]

        subnet["Author"][subnet["Author"].id]

        subnet["Publication"][subnet["Publication"].id]

    def peakmem_filter_twice(self, *args):
        publication_ids_1 = np.asarray(
            [4, 6], dtype=self.simple_pubnet.id_dtype
        )
        publication_ids_2 = 4

        subnet_1 = self.simple_pubnet[publication_ids_1]
        subsubnet = subnet_1[publication_ids_2]
        subnet_2 = self.simple_pubnet[publication_ids_2]

        subsubnet.isequal(subnet_2)

    def peakmem_filter_to_author(self, *args):
        random_author = self.simple_pubnet["Author"].get_random(1).LastName
        subnet = self.simple_pubnet.containing(
            "Author", "LastName", random_author
        )

    def peakmem_filter_to_author_multiple_steps(self, *args):
        random_author = self.simple_pubnet["Author"].get_random(1).LastName
        publication_ids = self.simple_pubnet.ids_containing(
            "Author", "LastName", random_author, steps=2
        )
        subnet = self.simple_pubnet[publication_ids]

        subnet["Author", "Publication"]["Publication"]
        subnet["Chemical", "Publication"]["Publication"]

      
    def  peakmem_update(self, *args):
        expected_nodes = set(self.simple_pubnet.nodes).union(
            set(self.other_pubnet.nodes)
        )

        expected_edges = set(self.simple_pubnet.edges).union(
            set(self.other_pubnet.edges)
        )

        self.simple_pubnet.update(self.other_pubnet)
      
           
class MemReadIO:
    params = IO_PARAMS
    param_names = IO_PARAM_NAMES

    def setup(self, representation, n_nodes, scope,formats):
        self.graph_name = f"{representation}_{n_nodes}"
        self._data_dir_obj = mktmpdir()
        self.data_dir = self._data_dir_obj.name
        self.simple_pubnet = simple_pubnet(representation, n_nodes)

        nodes = None
        edges = None
        if scope in ("node", "graph"):
            nodes = ("Publication", "Author")
        if scope in ("edge", "graph"):
            edges = (("Author", "Publication"),)

        self.simple_pubnet.to_dir(
            self.graph_name,
            nodes=nodes,
            edges=edges,
            data_dir=self.data_dir,
            format=formats,
        )

    def teardown(self, *args):
        self._data_dir_obj.cleanup()

    def peakmem_read_graph(self, representation, *args):
        new = from_dir(
            graph_name=self.graph_name,
            data_dir=self.data_dir,
            representation=representation,
        )


class MemWriteIO:
    params = IO_PARAMS
    param_names = IO_PARAM_NAMES

    def setup(self, representation, n_nodes, scope,formats):
        self.graph_name = f"{representation}_{n_nodes}"
        self._data_dir_obj = mktmpdir()
        self.data_dir = self._data_dir_obj.name
        self.simple_pubnet = simple_pubnet("igraph", n_nodes)

        nodes = None
        edges = None
        if scope in ("node", "graph"):
            nodes = ("Publication", "Author")
        if scope in ("edge", "graph"):
            edges = (("Author", "Publication"),)

        self.nodes = nodes
        self.edges = edges

    def teardown(self, *args):
        self._data_dir_obj.cleanup()

    def peakmem_write_graph(self, representation, n_nodes, scope, formats):
        self.simple_pubnet.to_dir(
            self.graph_name,
            nodes=self.nodes,
            edges=self.edges,
            data_dir=self.data_dir,
            format=formats,
        )
        
