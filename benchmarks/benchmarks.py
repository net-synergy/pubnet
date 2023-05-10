import os
import shutil
import time

import numpy as np
import pandas as pd
import pytest

import pubnet
from pubnet import from_dir


class TimeEdges:
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

    def time_finds_start_id(self, n):
        for e in self.simple_pubnet.edges:
            self.simple_pubnet[e].start_id

    def time_finds_end_id(self, n):
        for e in self.simple_pubnet.edges:
            self.simple_pubnet[e].end_id

    def time_overlap(self, n):
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


class TimeIO:
    #  params = [[['igraph',100],['igraph',1000],['igraph',10000],['numpy',100],['numpy',1000],['numpy',10000]]]
    params = [[["igraph", 100]]]

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

    def write_edge_setup(self):
        params = [
            ["igraph", 100],
            ["igraph", 1000],
            ["igraph", 10000],
            ["numpy", 100],
            ["numpy", 1000],
            ["numpy", 10000],
        ]

        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/write_edge")

        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for n in params:
            subdirname = os.path.join(dirname, n[0] + str(n[1]))
            if not os.path.exists(subdirname):
                os.mkdir(subdirname)

    def read_edge_setup(self):
        params = [
            ["igraph", 100],
            ["igraph", 1000],
            ["igraph", 10000],
            ["numpy", 100],
            ["numpy", 1000],
            ["numpy", 10000],
        ]

        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/read_edge")

        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for n in params:
            subdirname = os.path.join(dirname, n[0] + str(n[1]))
            if not os.path.exists(subdirname):
                os.mkdir(subdirname)
            self.simple_pubnet.to_dir(
                "edge",
                nodes=None,
                edges=(("Author", "Publication"),),
                data_dir=subdirname,
                format="tsv",
            )

    def write_node_setup(self):
        params = [
            ["igraph", 100],
            ["igraph", 1000],
            ["igraph", 10000],
            ["numpy", 100],
            ["numpy", 1000],
            ["numpy", 10000],
        ]

        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/write_node")

        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for n in params:
            subdirname = os.path.join(dirname, n[0] + str(n[1]))
            if not os.path.exists(subdirname):
                os.mkdir(subdirname)

    def read_node_setup(self):
        params = [
            ["igraph", 100],
            ["igraph", 1000],
            ["igraph", 10000],
            ["numpy", 100],
            ["numpy", 1000],
            ["numpy", 10000],
        ]

        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/read_node")

        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for n in params:
            subdirname = os.path.join(dirname, n[0] + str(n[1]))
            if not os.path.exists(subdirname):
                os.mkdir(subdirname)
            self.simple_pubnet.to_dir(
                "node",
                nodes=("Publication", "Author"),
                edges=None,
                data_dir=subdirname,
                format="tsv",
            )

    def write_graph_setup(self):
        params = [
            ["igraph", 100],
            ["igraph", 1000],
            ["igraph", 10000],
            ["numpy", 100],
            ["numpy", 1000],
            ["numpy", 10000],
        ]

        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/write_graph")

        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for n in params:
            subdirname = os.path.join(dirname, n[0] + str(n[1]))
            if not os.path.exists(subdirname):
                os.mkdir(subdirname)

    def read_graph_setup(self):
        params = [
            ["igraph", 100],
            ["igraph", 1000],
            ["igraph", 10000],
            ["numpy", 100],
            ["numpy", 1000],
            ["numpy", 10000],
        ]

        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/read_graph")

        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for n in params:
            subdirname = os.path.join(dirname, n[0] + str(n[1]))
            if not os.path.exists(subdirname):
                os.mkdir(subdirname)
            self.simple_pubnet.to_dir(
                "graph", data_dir=subdirname, format="tsv"
            )

    def write_edge_teardown(self):
        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/write_edge")
        shutil.rmtree(dirname)

    def read_edge_teardown(self):
        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/read_edge")
        shutil.rmtree(dirname)

    def write_node_teardown(self):
        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/write_node")
        shutil.rmtree(dirname)

    def read_node_teardown(self):
        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/read_node")
        shutil.rmtree(dirname)

    def write_graph_teardown(self):
        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/write_graph")
        shutil.rmtree(dirname)

    def read_graph_teardown(self):
        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/read_graph")
        shutil.rmtree(dirname)

    # def time_write_edge_io(self,n):
    #     working_dir = os.path.dirname(__file__)
    #     dirname = os.path.join(working_dir, 'io/write_edge')
    #     subdirname = os.path.join(dirname, n[0]+str(n[1]))

    #     self.simple_pubnet.to_dir(
    #         "edge",
    #         nodes=None,
    #         edges=(("Author", "Publication"),),
    #         data_dir=subdirname,
    #         format="tsv",
    #     )
    # time_write_edge_io.setup = write_edge_setup
    # time_write_edge_io.teardown = write_edge_teardown

    def time_read_edge_io(self, n):
        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/read_edge")
        subdirname = os.path.join(dirname, n[0] + str(n[1]))

        new = from_dir(
            graph_name="edge",
            data_dir=subdirname,
            representation=n[0],
        )

    time_read_edge_io.setup = read_edge_setup
    time_read_edge_io.teardown = read_edge_teardown

    # def time_write_node_io(self,n):
    #     working_dir = os.path.dirname(__file__)
    #     dirname = os.path.join(working_dir, 'io/write_node')
    #     subdirname = os.path.join(dirname, n[0]+str(n[1]))

    #     self.simple_pubnet.to_dir(
    #         "node",
    #         nodes=("Publication", "Author"),
    #         edges=None,
    #         data_dir=subdirname,
    #         format="tsv",
    #     )
    # time_write_node_io.setup = write_node_setup
    # time_write_node_io.teardown = write_node_teardown

    def time_read_node_io(self, n):
        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/read_node")
        subdirname = os.path.join(dirname, n[0] + str(n[1]))

        new = from_dir(graph_name="node", data_dir=subdirname)

    time_read_node_io.setup = read_node_setup
    time_read_node_io.teardown = read_node_teardown

    # def time_write_graph_io(self,n):
    #     working_dir = os.path.dirname(__file__)
    #     dirname = os.path.join(working_dir, 'io/write_graph')
    #     subdirname = os.path.join(dirname, n[0]+str(n[1]))

    #     self.simple_pubnet.to_dir("graph", data_dir=subdirname, format="tsv")
    # time_write_graph_io.setup = write_graph_setup
    # time_write_graph_io.teardown = write_graph_teardown

    def time_read_graph_io(self, n):
        working_dir = os.path.dirname(__file__)
        dirname = os.path.join(working_dir, "io/read_graph")
        subdirname = os.path.join(dirname, n[0] + str(n[1]))

        new = from_dir(
            graph_name="graph",
            data_dir=subdirname,
            representation=n[0],
        )

    time_read_graph_io.setup = read_graph_setup
    time_read_graph_io.teardown = read_graph_teardown
