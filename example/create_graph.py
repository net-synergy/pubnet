from pubnet import PubNet

nodes = ("Author", "Publication", "Descriptor", "Chemical")
edges = (
    ("Author", "Publication"),
    ("Descriptor", "Publication"),
    ("Chemical", "Publication"),
)

publications_np = PubNet.load_graph(
    "pubmed",
    nodes=nodes,
    edges=edges,
    representation="numpy",
)

publications_ig = PubNet.load_graph(
    "pubmed",
    nodes=nodes,
    edges=edges,
    representation="igraph",
)

publications_ig[10627536]
publications_np[10627536]

eig = publications_ig["Author", "Publication"]
enp = publications_np["Author", "Publication"]

list(publications_ig["Author", "Publication"][1:5, "Author"])
publications_np["Author", "Publication"].isin(
    "Publication", range(10627536, 10627600)
)
publications_ig["Author", "Publication"].isin(
    "Publication", range(10627536, 10627600)
)

publications_ig[range(10627536, 11000000)]
publications_np[range(10627536, 11000000)]

last_names = list(
    publications_np["Author"].get_random(n=4, seed=1)["LastName"]
)
subnet = publications_np.containing("Author", "LastName", last_names, steps=2)
subnet.re_root("Author", counts="normalize")
ovr = subnet.overlap({"Chemical", "Publication"}, mutate=False)
ovr.reduce_edges(lambda x, acc: x + acc, "overlap", normalize=True).as_igraph()

author_ovr_np = subnet["Author", "Publication"].overlap("Author")

subnet = publications_ig.containing("Author", "LastName", last_names, steps=2)
author_ovr_ig = subnet["Author", "Publication"].overlap("Author")

# sim = subnet["Author", "Publication"].similarity(
#     publications_np.containing("Author", "LastName", last_names)
# )
