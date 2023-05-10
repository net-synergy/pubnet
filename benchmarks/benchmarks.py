import os

import numpy as np

import pubnet

from ._utils import simple_pubnet, mktmpdir

LOG_NUM_NODE_RANGE = (2, 5)
PARAMS = (("numpy", "igraph"), 10 ** np.arange(*LOG_NUM_NODE_RANGE))
PARAM_NAMES = ("Representation", "n_nodes")
IO_PARAMS = PARAMS + (("node", "edge", "graph"), ("tsv", "gzip", "binary"))
IO_PARAM_NAMES = PARAM_NAMES + ("scope", "format")


class TimeEdges:
    params = PARAMS
    param_names = PARAM_NAMES

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

    time_overlap.timeout = 480


class TimeNodes:
    params = [
        [
            ["igraph", 100],
            ["igraph", 1000],
            ["igraph", 10000],
            ["numpy", 100],
            ["numpy", 1000],
            ["numpy", 10000],
        ]
    ]

    def setup(self, n):
        data_dir = os.path.dirname(__file__)
        simple_pubnet = pubnet.from_dir(
            graph_name="graphs",
            nodes=("Author", "Publication", "Descriptor", "Chemical"),
            edges=(
                ("Author", "Publication"),
                ("Descriptor", "Publication"),
                ("Chemical", "Publication"),
            ),
            data_dir=data_dir,
            root="Publication",
            representation=n[0],
        )
        random_nodes = simple_pubnet["Author"].get_random(n[1])
        self.simple_pubnet = simple_pubnet.containing(
            "Author", "AuthorId", random_nodes["AuthorId"]
        )


def time_finds_namespace(self, n):
    self.simple_pubnet["Author"].id == "AuthorId"


def time_slice_column(self, n):
    self.simple_pubnet["Author"]["LastName"]


def time_slice_columns(self, n):
    features = ["LastName", "ForeName"]
    self.simple_pubnet["Author"][features].columns.values


def time_slice_column_by_index(self, n):
    self.simple_pubnet["Author"][0] is self.simple_pubnet["Author"][
        self.simple_pubnet["Author"].features[0]
    ]
    self.simple_pubnet["Author"][1] is self.simple_pubnet["Author"][
        self.simple_pubnet["Author"].features[1]
    ]


def time_slice_rows_by_index(self, n):
    actual = self.simple_pubnet["Author"][0:2]


def time_slice_rows_by_mask(self, n):
    self.simple_pubnet["Author"][
        self.simple_pubnet["Author"]["LastName"] == "Smith"
    ]


def time_slice_rows_and_columns(self, n):
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
    params = [
        [
            ["igraph", 100],
            ["igraph", 1000],
            ["igraph", 10000],
            ["numpy", 100],
            ["numpy", 1000],
            ["numpy", 10000],
        ]
    ]

    def setup(self, n):
        data_dir = os.path.dirname(__file__)
        simple_pubnet = pubnet.from_dir(
            graph_name="graphs",
            nodes=("Author", "Publication", "Descriptor", "Chemical"),
            edges=(
                ("Author", "Publication"),
                ("Descriptor", "Publication"),
                ("Chemical", "Publication"),
            ),
            data_dir=data_dir,
            root="Publication",
            representation=n[0],
        )

        random_nodes = simple_pubnet["Author"].get_random(n[1])["AuthorId"]

        self.simple_pubnet = simple_pubnet.containing(
            "Author", "AuthorId", random_nodes
        )

    def update_setup(self):
        working_dir = os.path.dirname(__file__)
        data_dir = os.path.join(
            working_dir[0 : working_dir.rindex("/")], "tests/data"
        )
        global other_pubnet
        other_pubnet = pubnet.from_dir(
            graph_name="simple_pubnet",
            nodes=("Author", "Publication"),
            edges=(("Publication", "Author"),),
            data_dir=data_dir,
            representation="numpy",
        )


def time_creates_empty_nodes_for_missing_edge_nodes(self, n):
    len(self.simple_pubnet["Chemical"]) == 0


def time_filter_to_single_publication_id(self, n):
    publication_id = 1
    subnet = self.simple_pubnet[publication_id]

    subnet["Author"][subnet["Author"].id]

    subnet["Publication"][subnet["Publication"].id]


def time_filter_to_publicaiton_ids(self, n):
    publication_ids = np.asarray([1, 2], dtype=self.simple_pubnet.id_dtype)
    subnet = self.simple_pubnet[publication_ids]

    subnet["Author"][subnet["Author"].id]

    subnet["Publication"][subnet["Publication"].id]


def time_filter_twice(self, n):
    publication_ids_1 = np.asarray([4, 6], dtype=self.simple_pubnet.id_dtype)
    publication_ids_2 = 4

    subnet_1 = self.simple_pubnet[publication_ids_1]
    subsubnet = subnet_1[publication_ids_2]
    subnet_2 = self.simple_pubnet[publication_ids_2]

    subsubnet.isequal(subnet_2)


# not working
def time_filter_to_author(self, n):
    subnet = self.simple_pubnet.containing("Author", "LastName", "Smith")


# not working
def time_filter_to_author_multiple_steps(self, n):
    publication_ids = self.simple_pubnet.ids_containing(
        "Author", "LastName", "Smith", steps=2
    )
    subnet = self.simple_pubnet[publication_ids]

    subnet["Author", "Publication"]["Publication"]
    subnet["Chemical", "Publication"]["Publication"]


def time_update(self, n):
    expected_nodes = set(self.simple_pubnet.nodes).union(
        set(other_pubnet.nodes)
    )

    expected_edges = set(self.simple_pubnet.edges).union(
        set(other_pubnet.edges)
    )

    self.simple_pubnet.update(other_pubnet)
    set(self.simple_pubnet.nodes) == expected_nodes
    set(self.simple_pubnet.edges) == expected_edges
    self.simple_pubnet["Author", "Publication"].isequal(
        other_pubnet["Author", "Publication"]
    )


time_update.setup = update_setup


    # not working
    def time_filter_to_author(self, n):
        subnet = self.simple_pubnet.containing("Author", "LastName", "Smith")

    # not working
    def time_filter_to_author_multiple_steps(self, n):
        publication_ids = self.simple_pubnet.ids_containing(
            "Author", "LastName", "Smith", steps=2
        )
        subnet = self.simple_pubnet[publication_ids]

        subnet["Author", "Publication"]["Publication"]
        subnet["Chemical", "Publication"]["Publication"]

    def time_update(self, n):
        expected_nodes = set(self.simple_pubnet.nodes).union(
            set(other_pubnet.nodes)
        )

        expected_edges = set(self.simple_pubnet.edges).union(
            set(other_pubnet.edges)
        )

        self.simple_pubnet.update(other_pubnet)
        set(self.simple_pubnet.nodes) == expected_nodes
        set(self.simple_pubnet.edges) == expected_edges
        self.simple_pubnet["Author", "Publication"].isequal(
            other_pubnet["Author", "Publication"]
        )

    time_update.setup = update_setup


class TimeReadIO:
    params = IO_PARAMS
    param_names = IO_PARAM_NAMES

    def setup(self, representation, n_nodes, scope):
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
            format=format,
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

    def setup(self, representation, n_nodes, scope):
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

        self.nodes = nodes
        self.edges = edges

    def teardown(self, *args):
        self._data_dir_obj.cleanup()

    def time_write_graph(self, representation, n_nodes, scope, format):
        self.simple_pubnet.to_dir(
            self.graph_name,
            nodes=self.nodes,
            edges=self.edges,
            data_dir=self.data_dir,
            format=format,
        )
