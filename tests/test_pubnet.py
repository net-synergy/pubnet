import numpy as np
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
    def test_finds_namespace(self, simple_pubnet):
        assert simple_pubnet["Author"].id == "AuthorId"

    def test_shape(self, simple_pubnet):
        assert simple_pubnet["Author"].shape == (4, 3)


class TestNetwork:
    def test_handles_no_nodes(self):
        net = PubNet(
            None,
            (("Publication", "Author"),),
            data_dir="tests/data/simple_pubnet",
            compressed=False,
        )
        assert net.nodes is None

    def test_handles_no_edges(self):
        net = PubNet(
            ("Publication",),
            None,
            data_dir="tests/data/simple_pubnet",
            compressed=False,
        )
        assert net.edges is None

    def test_filter_to_publicaiton_ids(self, simple_pubnet):
        publication_ids = np.asarray([1, 2], dtype=np.int64)
        subnet = simple_pubnet[publication_ids]

        assert subnet["Author", "Publication"].shape == 5
        assert subnet["Chemical", "Publication"].shape == 4

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
