import time

import numpy as np
import pandas as pd
import pytest

import pubnet
from pubnet import PubNet

from ._test_fixtures import author_node, other_pubnet, simple_pubnet


class TestEdges:
    def test_finds_start_id(self, simple_pubnet):
        for e in simple_pubnet.edges:
            assert simple_pubnet[e].start_id == "Publication"

    def test_finds_end_id(self, simple_pubnet):
        expected = ["Author", "Chemical"]
        for e in simple_pubnet.edges:
            assert simple_pubnet[e].end_id in expected

    def test_handles_swapped_start_and_close_id(self):
        net = PubNet.load_graph(
            "simple_pubnet",
            ("Publication",),
            (("Publication", "Flippedheaders"),),
            data_dir="tests/data",
        )
        edges = net["Publication", "Flippedheaders"]
        assert edges[edges.start_id][0] == 2
        assert edges[edges.end_id][0] == 1

    def test_shape(self, simple_pubnet):
        assert len(simple_pubnet["Author", "Publication"]) == 12
        assert len(simple_pubnet["Chemical", "Publication"]) == 10

    @pytest.mark.skip("Modifying overlap, expect could fail for now.")
    def test_overlap(self, simple_pubnet):
        expected = np.array(
            [
                [1, 2, 2],
                [1, 3, 2],
                [1, 4, 1],
                [1, 5, 1],
                [2, 3, 1],
                [2, 4, 1],
                [2, 5, 1],
                [3, 5, 1],
                [4, 5, 1],
                [4, 6, 1],
                [5, 6, 1],
            ]
        )
        assert np.array_equal(
            simple_pubnet["Author", "Publication"].overlap, expected
        )


class TestNodes:
    def test_finds_namespace(self, author_node):
        assert author_node.id == "AuthorId"

    def test_shape(self, author_node):
        assert author_node.shape == (4, 2)

    def test_slice_column(self, author_node):
        assert author_node.feature_vector("LastName")[0] == "Smith"

    def test_slice_columns(self, author_node):
        features = ["LastName", "ForeName"]
        assert (author_node[features].columns.values == features).all()

    def test_slice_column_by_index(self, author_node):
        assert author_node[0].isequal(author_node[author_node.features[0]])
        assert author_node[1].isequal(author_node[author_node.features[1]])

    def test_slice_rows_by_index(self, author_node):
        expected = pd.DataFrame(
            {
                "AuthorId": [1, 2],
                "LastName": ["Smith", "Kim"],
                "ForeName": ["John", "John"],
            }
        )
        actual = author_node[0:2]

        for feature in author_node.features:
            assert (
                actual.feature_vector(feature) == expected[feature].values
            ).all()

    def test_slice_rows_by_mask(self, author_node):
        actual = author_node[author_node.feature_vector("LastName") == "Smith"]
        expected = pd.DataFrame(
            {
                "AuthorId": [1, 3],
                "LastName": ["Smith", "Smith"],
                "ForeName": ["John", "Jane"],
            }
        )

        for feature in author_node.features:
            assert (
                actual.feature_vector(feature) == expected[feature].values
            ).all()

    def test_slice_rows_and_columns(self, author_node):
        actual = {
            "Slices": author_node[0:2, 0:2],
            "Slice + List": author_node[0:2, ["AuthorId", "LastName"]],
            "Mask + Slice": author_node[
                author_node.feature_vector("ForeName") == "John", 0:2
            ],
        }
        expected = pd.DataFrame(
            {"AuthorId": [1, 2], "LastName": ["Smith", "Kim"]}
        )

        for node in actual.values():
            assert (
                node.feature_vector("LastName") == expected["LastName"].values
            ).all()
            assert (node.index == expected["AuthorId"].values).all()


class TestNetwork:
    @pytest.mark.filterwarnings(
        "ignore:Constructing PubNet object without Publication nodes."
    )
    def test_handles_no_nodes(self):
        net = PubNet.load_graph(
            "simple_pubnet",
            None,
            (("Publication", "Author"),),
            data_dir="tests/data",
        )
        assert len(net["Publication"]) == 0
        assert len(net["Author"]) == 0

    def test_creates_empty_nodes_for_missing_edge_nodes(self, simple_pubnet):
        assert len(simple_pubnet["Chemical"]) == 0

    def test_handles_no_edges(self):
        net = PubNet.load_graph(
            "simple_pubnet",
            ("Publication",),
            None,
            data_dir="tests/data",
        )
        assert len(net.edges) == 0

    def test_filter_to_single_publication_id(self, simple_pubnet):
        publication_id = 1
        subnet = simple_pubnet[publication_id]

        expected_authors = np.asarray([1, 2, 3])

        assert len(subnet["Author", "Publication"]) == 3
        assert len(subnet["Chemical", "Publication"]) == 2
        assert np.array_equal(
            np.unique(subnet.get_node("Author").index), expected_authors
        )
        assert np.array_equal(
            np.unique(subnet.get_node("Publication").index),
            np.asarray([publication_id]),
        )

    def test_filter_to_publicaiton_ids(self, simple_pubnet):
        publication_ids = np.asarray([1, 2], dtype=simple_pubnet.id_dtype)
        subnet = simple_pubnet[publication_ids]

        expected_authors = np.asarray([1, 2, 3])

        assert len(subnet["Author", "Publication"]) == 5
        assert len(subnet["Chemical", "Publication"]) == 4
        assert np.array_equal(
            np.unique(subnet.get_node("Author").index), expected_authors
        )
        assert np.array_equal(
            np.unique(subnet.get_node("Publication").index),
            publication_ids,
        )

    # Issue where filtering a network created by filtering another
    # network is not working. This should ensure the network returned
    # by slicing doesn't have any unexpected traits.
    #
    # Caused by DataFrame using row IDs for numeric indexing but row
    # IDs aren't regenerated after slicing. If the first row is
    # filtered out, there is no longer a row 0, and indexing with 0 is
    # a key error.
    def test_filter_twice(self, simple_pubnet):
        publication_ids_1 = np.asarray([4, 6], dtype=simple_pubnet.id_dtype)
        publication_ids_2 = 4

        subnet_1 = simple_pubnet[publication_ids_1]
        subsubnet = subnet_1[publication_ids_2]
        subnet_2 = simple_pubnet[publication_ids_2]

        assert subsubnet.isequal(subnet_2)

    def test_filter_to_author(self, simple_pubnet):
        subnet = simple_pubnet.containing("Author", "LastName", "Smith")
        expected_publication_ids = np.asarray([1, 2, 3, 5])

        assert np.array_equal(
            np.unique(subnet["Author", "Publication"]["Publication"]),
            expected_publication_ids,
        )
        assert np.array_equal(
            np.unique(subnet["Chemical", "Publication"]["Publication"]),
            expected_publication_ids,
        )

        assert np.array_equal(
            np.unique(subnet.get_node("Publication").index),
            expected_publication_ids,
        )

    def test_filter_to_author_multiple_steps(self, simple_pubnet):
        # Currently doesn't remove any publications since all
        # publications within two steps.
        publication_ids = simple_pubnet.ids_containing(
            "Author", "LastName", "Smith", steps=2
        )
        subnet = simple_pubnet[publication_ids]
        expected_publication_ids = np.asarray([1, 2, 3, 4, 5, 6])

        assert np.array_equal(
            np.unique(subnet["Author", "Publication"]["Publication"]),
            expected_publication_ids,
        )
        assert np.array_equal(
            np.unique(subnet["Chemical", "Publication"]["Publication"]),
            expected_publication_ids,
        )

    def test_drops_node(self, simple_pubnet):
        node = "Author"
        simple_pubnet.drop(node)

        assert node not in simple_pubnet.nodes
        assert node not in simple_pubnet._node_data.keys()

    def test_drops_nodes(self, simple_pubnet):
        nodes = ["Author", "Chemical"]
        simple_pubnet.drop(nodes)

        assert np.isin(nodes, simple_pubnet.nodes, invert=True).all()
        assert np.isin(
            nodes, simple_pubnet._node_data.keys(), invert=True
        ).all()

    def test_drops_edge(self, simple_pubnet):
        edge = ("Author", "Publication")
        simple_pubnet.drop(edges=edge)

        edge = pubnet.network.edge_key(*edge)
        assert edge not in simple_pubnet.edges
        assert edge not in simple_pubnet._edge_data.keys()

    def test_drops_edges(self, simple_pubnet):
        edges = (("Author", "Publication"), ("Chemical", "Publication"))
        simple_pubnet.drop(edges=edges)

        edges = simple_pubnet._as_keys(edges)
        assert np.isin(
            edges, simple_pubnet._as_keys(simple_pubnet.edges), invert=True
        ).all()
        assert np.isin(
            edges, simple_pubnet._edge_data.keys(), invert=True
        ).all()

    @pytest.mark.filterwarnings(
        "ignore:Constructing PubNet object without Publication nodes."
    )
    def test_update(self, simple_pubnet, other_pubnet):
        expected_nodes = set(simple_pubnet.nodes).union(
            set(other_pubnet.nodes)
        )

        expected_edges = set(simple_pubnet.edges).union(
            set(other_pubnet.edges)
        )
        print(expected_edges)

        other_pubnet.drop("Publication")
        simple_pubnet.update(other_pubnet)
        assert set(simple_pubnet.nodes) == expected_nodes
        assert set(simple_pubnet.edges) == expected_edges
        # Assert other's Chemical-Publication edge shadowed simple_pubnet's.
        assert simple_pubnet["Chemical", "Publication"].isequal(
            other_pubnet["Chemical", "Publication"]
        )


@pytest.mark.skip("Modifying simialrity and overlap methods.")
class TestSnapshots:
    """Ensure consistency between edge representations."""

    @pytest.mark.parametrize("method", ["shortest_path"])
    def test_similarity(self, simple_pubnet, method, snapshot):
        snapshot.snapshot_dir = "tests/snapshots"
        publication_ids = simple_pubnet.ids_containing(
            "Author", "LastName", "Smith"
        )
        snapshot.assert_match(
            str(
                simple_pubnet["Author", "Publication"].similarity(
                    publication_ids, method
                )
            ),
            f"similarity_{method}_output.txt",
        )

    @pytest.mark.parametrize("method", ["shortest_path"])
    def test_repeated_overlap_calculations(self, simple_pubnet, method):
        """Overlap is stored in a variable so subsequent runs should be quicker
        than original run."""

        publication_ids = simple_pubnet.ids_containing(
            "Author", "LastName", "Smith"
        )

        t_start = time.time()
        sim_1 = simple_pubnet["Author", "Publication"].similarity(
            publication_ids, method
        )
        t_1 = time.time() - t_start

        t_start = time.time()
        sim_2 = simple_pubnet["Author", "Publication"].similarity(
            publication_ids, method
        )
        t_2 = time.time() - t_start

        assert t_2 < t_1
        assert np.all(sim_1 == sim_2)
