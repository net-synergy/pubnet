from pubnet import PubNet

data_dir = "example/graphs"
nodes = ("Author", "Publication", "Descriptor", "Chemical")
edges = (
    ("Author", "Publication"),
    ("Descriptor", "Publication"),
    ("Chemical", "Publication"),
)

publications = PubNet(nodes, edges, data_dir)
authors = publications["Author"].get_random(n=4, seed=1)
publication_ids = publications.publications_containing(
    "Author", "LastName", list(authors["LastName"]), steps=2
)

subnet = publications[publication_ids]
sim = subnet["Author", "Publication"].similarity(
    publications.publications_containing(
        "Author", "LastName", list(authors["LastName"])
    ),
)
