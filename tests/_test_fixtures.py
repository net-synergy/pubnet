import pubnet
import pytest

__all__ = ["simple_pubnet", "other_pubnet", "author_node"]


@pytest.fixture(params=["numpy", "igraph"])
def simple_pubnet(request):
    try:
        return pubnet.from_dir(
            "simple_pubnet",
            ("Author", "Publication"),
            (("Publication", "Author"), ("Publication", "Chemical")),
            data_dir="tests/data",
            representation=request.param,
        )
    except NotImplementedError:
        pytest.skip("Not implemented")


@pytest.fixture
def other_pubnet():
    return pubnet.from_dir(
        "other_pubnet",
        ("Chemical",),
        (("Publication", "Chemical"),),
        data_dir="tests/data",
    )


@pytest.fixture
def author_node(simple_pubnet):
    return simple_pubnet["Author"]
