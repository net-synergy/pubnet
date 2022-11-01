import pubnet
import pytest

__all__ = ["simple_pubnet", "other_pubnet", "author_node"]

# If true uses CompressedEdge otherwise NumpyEdge.  If more edge types
# are added in the future this sholud be extended to `edge_type` and
# return the str representing the desired type.
@pytest.fixture(params=["numpy", "igraph"])
def simple_pubnet(request):
    try:
        return pubnet.from_dir(
            "Publication",
            ("Author", "Publication"),
            (("Publication", "Author"), ("Publication", "Chemical")),
            data_dir="tests/data",
            graph_name="simple_pubnet",
            representation=request.param,
        )
    except NotImplementedError:
        pytest.skip("Not implemented")


@pytest.fixture
def other_pubnet():
    return pubnet.from_dir(
        "Publication",
        ("Chemical",),
        (("Publication", "Chemical"),),
        data_dir="tests/data",
        graph_name="other_pubnet",
    )


@pytest.fixture
def author_node(simple_pubnet):
    return simple_pubnet["Author"]
