from pubnet import from_dir

data_dir = "example/graphs"
nodes = ("Author", "Publication", "Descriptor", "Chemical")
edges = (
    ("Author", "Publication"),
    ("Descriptor", "Publication"),
    ("Chemical", "Publication"),
)

publications = from_dir(
    "Publication",
    nodes=nodes,
    edges=edges,
    data_dir=data_dir,
    representation="igraph",
)
last_names = list(publications["Author"].get_random(n=4, seed=1)["LastName"])
publication_ids = publications.containing(
    "Author", "LastName", last_names, steps=2
)

subnet = publications[publication_ids]
sim = subnet["Author", "Publication"].similarity(
    publications.containing("Author", "LastName", last_names)
)
