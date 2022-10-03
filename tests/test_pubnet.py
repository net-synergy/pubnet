import numpy as np
import pandas as pd
import pytest
from pubnet import PubNet


# If true uses CompressedEdge otherwise NumpyEdge.  If more edge types
# are added in the future this sholud be extended to `edge_type` and
# return the str representing the desired type.
@pytest.fixture(params=[True, False])
def simple_pubnet(request):
    try:
        return PubNet(
            ("Author", "Publication"),
            (("Publication", "Author"), ("Publication", "Chemical")),
            data_dir="tests/data/simple_pubnet",
            compressed=request.param,
        )
    except NotImplementedError:
        pytest.skip("Not implemented")


@pytest.fixture
def author_node(simple_pubnet):
    return simple_pubnet["Author"]


class TestEdges:
    def test_finds_start_id(self, simple_pubnet):
        for e in simple_pubnet.edges:
            assert simple_pubnet[e].start_id == e[0]

    def test_finds_close_id(self, simple_pubnet):
        for e in simple_pubnet.edges:
            assert simple_pubnet[e].end_id == e[1]

    @pytest.mark.xfail
    def test_handles_swapped_start_and_close_id(self):
        net = PubNet(
            ("Publication",),
            (("Publication", "Flippedheaders"),),
            data_dir="tests/data/simple_pubnet",
            compressed=False,
        )
        edges = net["Publication", "Flippedheaders"]
        assert edges[edges.start_id][0] == 2
        assert edges[edges.end_id][0] == 1

    def test_shape(self, simple_pubnet):
        assert simple_pubnet["Author", "Publication"].shape == 12
        assert simple_pubnet["Chemical", "Publication"].shape == 10

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
        assert author_node.shape == (4, 3)

    def test_slice_column(self, author_node):
        assert author_node["LastName"][0] == "Smith"

    def test_slice_columns(self, author_node):
        features = ["LastName", "ForeName"]
        assert (author_node[features].columns.values == features).all()

    def test_slice_column_by_index(self, author_node):
        assert author_node[0] is author_node["AuthorId"]
        assert author_node[1] is author_node["LastName"]

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
            assert (actual[feature].values == expected[feature].values).all()

    def test_slice_rows_by_mask(self, author_node):
        actual = author_node[author_node["LastName"] == "Smith"]
        expected = pd.DataFrame(
            {
                "AuthorId": [1, 3],
                "LastName": ["Smith", "Smith"],
                "ForeName": ["John", "Jane"],
            }
        )

        for feature in author_node.features:
            assert (actual[feature].values == expected[feature].values).all()

    def test_slice_rows_and_columns(self, author_node):
        actual = {
            "Slices": author_node[0:2, 0:2],
            "Slice + List": author_node[0:2, ["AuthorId", "LastName"]],
            "Mask + Slice": author_node[
                author_node["ForeName"] == "John", 0:2
            ],
        }
        expected = pd.DataFrame(
            {"AuthorId": [1, 2], "LastName": ["Smith", "Kim"]}
        )

        for node in actual.values():
            for feature in expected.columns:
                assert (node[feature].values == expected[feature].values).all()


class TestNetwork:
    @pytest.mark.filterwarnings(
        "ignore:Constructing PubNet object without Publication nodes."
    )
    def test_handles_no_nodes(self):
        net = PubNet(
            None,
            (("Publication", "Author"),),
            data_dir="tests/data/simple_pubnet",
            compressed=False,
        )
        assert len(net["Publication"]) == 0
        assert len(net["Author"]) == 0

    def test_creates_empty_nodes_for_missing_edge_nodes(self, simple_pubnet):
        assert len(simple_pubnet["Chemical"]) == 0

    def test_handles_no_edges(self):
        net = PubNet(
            ("Publication",),
            None,
            data_dir="tests/data/simple_pubnet",
            compressed=False,
        )
        assert len(net.edges) == 0

    def test_filter_to_single_publication_id(self, simple_pubnet):
        publication_id = 1
        subnet = simple_pubnet[publication_id]

        expected_authors = np.asarray([1, 2, 3])

        assert subnet["Author", "Publication"].shape == 3
        assert subnet["Chemical", "Publication"].shape == 2
        assert np.array_equal(
            np.unique(subnet["Author"][subnet["Author"].id]), expected_authors
        )

    def test_filter_to_publicaiton_ids(self, simple_pubnet):
        publication_ids = np.asarray([1, 2], dtype=simple_pubnet.id_datatype)
        subnet = simple_pubnet[publication_ids]

        expected_authors = np.asarray([1, 2, 3])

        assert subnet["Author", "Publication"].shape == 5
        assert subnet["Chemical", "Publication"].shape == 4
        assert np.array_equal(
            np.unique(subnet["Author"][subnet["Author"].id]), expected_authors
        )

    def test_filter_to_author(self, simple_pubnet):
        publication_ids = simple_pubnet.publications_containing(
            "Author", "LastName", "Smith"
        )
        subnet = simple_pubnet[publication_ids]
        expected_publication_ids = np.asarray([1, 2, 3, 5])

        assert np.array_equal(
            np.unique(subnet["Author", "Publication"]["Publication"]),
            expected_publication_ids,
        )
        assert np.array_equal(
            np.unique(subnet["Chemical", "Publication"]["Publication"]),
            expected_publication_ids,
        )

    def test_filter_to_author_multiple_steps(self, simple_pubnet):
        # Currently doesn't remove any publications since all
        # publications within two steps.
        publication_ids = simple_pubnet.publications_containing(
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


class TestSnapshots:
    """Ensure consistency between edge representations."""

    @pytest.mark.parametrize("method", ["shortest_path"])
    def test_similarity(self, simple_pubnet, method, snapshot):
        snapshot.snapshot_dir = "tests/snapshots"
        publication_ids = simple_pubnet.publications_containing(
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