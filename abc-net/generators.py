# Commonly used graph generators

import numpy as np
import networkx as nx
import graph_tool as gt
import graph_tool.stats as gts
rng = np.random.default_rng()
from sklearn.neighbors import NearestNeighbors


def build_erdosh_reny(n: int, p: float) -> gt.Graph:
    """
    Build an Erdosh-Reny graph with given amount of vertices and connection probability
    Args:
        n: number of nodes/vertices
        p: connection probability
    Returns:
        G: graph-tool network
    """
    G = gt.generation.random_graph(n, lambda: rng.poisson((n-1) * p), directed=False, model="erdos")
    
    # clean the graph
    gts.remove_parallel_edges(G)
    gts.remove_self_loops(G)
        
    return G


def build_extended_barabasi_albert(n: int, m: int = 1, p: float = 0.25, q: float = 0.25) -> gt.Graph:
    """
    Build an extended Barabasi Albert
    Args:
        n: number of nodes/vertices
        m: number of edges added for each node
        p: probability according to which, new m edges are added to the graph, starting from randomly chosen existing \
        nodes and attached preferentially at the other end.
        q: probability according to which existing m edges are rewired by randomly choosing an edge and \
        rewiring one end to a preferentially chosen node.
    Returns:
        G: graph-tool network
    """
    G = nx.extended_barabasi_albert_graph(n, m, p, q)
    G = nx2gt(G)
    gts.remove_parallel_edges(G)
    gts.remove_self_loops(G)
    return G


def build_watts_strogatz(n: int, k: int, p: float) -> gt.Graph:
    """
    Build a Watts-Strogatz network
    Args:
        n: number of nodes/vertices
        k: number of linked neighbours
        p: rewiring probability
    Returns:
        G: graph-tool network
    """
    # build circular network
    G = gt.generation.circular_graph(n, k)
    # find which edges need to be rewired
    edge_index = np.where( rng.uniform(0, 1, size=G.num_edges()) < p )[0]
    edges_to_rewire = G.get_edges()[edge_index]
    # select new vertices
    new_vertices = rng.integers(n,size=len(edge_index))
    # rewire
    for i,e in enumerate(edges_to_rewire):
        # add new edge
        G.add_edge(G.vertex(e[0]), G.vertex(new_vertices[i]))
        # remove old one
        er = G.edge(G.vertex(e[0]), G.vertex(e[1]))
        G.remove_edge(er)
    gts.remove_self_loops(G)
    gts.remove_parallel_edges(G)
    return G


def build_barabasi_albert(n: int, m: int = 1, gamma: float = 1., c: float = 0.) -> gt.Graph:
    """
    Build a Barabasi Albert network
    Args:
        n: number of nodes/vertices
        m: number of links for each new node
        gamma: power-law preferential attachment parameter
        c: additive constant to preferential attachment
    Returns:
        G: graph-tool network
    """
    G = gt.generation.price_network(n, m=m, gamma=gamma, c=c, directed=False, seed_graph=None)
    return G


def build_planted_partition(l: int = 20, k: int = 30, p_in: float = 0.25, p_out: float = 0.0025) -> gt.Graph:
    """
    Build a planted partition network
    Args:
        l: number of clusters
        k: number of nodes/vertices per cluster
        p_in: probability of wiring within cluster
        p_out: probability of wiring between different clusters
    Returns:
        G: graph-tool network
    """
    G = nx.planted_partition_graph(l, k, p_in, p_out)
    G = nx2gt(G)
    gts.remove_parallel_edges(G)
    gts.remove_self_loops(G)
    return G


def build_geom_net(X = None, n: int = None, d: int = None, D: int = None, noise: float = 0.,
                       threshold: float = None, nn: int = None, lambda_poisson: float = None) -> gt.Graph():
    """
    Build graph from given geometrical points X or create them with given d embedded into D
    Args:
        X: datapoints
        n: number of points/nodes
        d: intrinsic dimension
        D: embedding dimension
        noise_size: noise in embedding dimensions
        threshold (float): radius within neighbours are taken
        nn (int): number of neighbours considered (uniform for all vertices)
        lambda_poisson (float): the number of neighbours to be considered for each vertex is extracted from
            a Poisson distribution (with expected value lambda), instead of being uniform
    Returns:
        G: graph
    """

    # assert threshold is not None or nn is not None, "set the threshold or the number of neighbours"

    if X is None:
        assert ((n is not None) and (d is not None)), "if datapoints are not given, set n and d at least"
        X = rng.uniform(0,1,size=(n,d))
        if noise > 0 and D > d:
            X = np.c_[X, np.zeros((n, D - d))] + rng.normal(scale=noise, size=(n, D))

    if nn is not None: 
        maxk = nn
    else:
        maxk = n if n < 5000 else 1000
        
    nbrs = NearestNeighbors(n_neighbors=maxk).fit(X)
    distances, dist_indices = nbrs.kneighbors(X)

    if lambda_poisson is not None:
        k_n = rng.poisson(lambda_poisson, size=n)
        edge_list = [ (i[0], target) for i in dist_indices for target in i[1:k_n[i[0]]+1] ]
    elif nn is not None:
        edge_list = [ (i[0], target) for i in dist_indices for target in i[1:nn+1] ]
    else:
        edge_list = [ (i[0], target) for i in dist_indices for target in i[i < threshold] ]

    G = gt.Graph(directed=False)
    G.add_edge_list(edge_list)
    gts.remove_parallel_edges(G)
    gts.remove_self_loops(G)
    
    return G


# conversion from nx to gt in order to speed up further operations (like shortest paths)
def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    if isinstance(key, str):
        # Encode the key as ASCII
        key = str(key)#.encode('ascii', errors='replace'))

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, str):
        tname = 'string'
        value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    Adapted to Python3 from https://gist.github.com/bbengfort/a430d460966d64edc6cad71c502d7005#file-nx2gt-py
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set() # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key  = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname) # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set() # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname) # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {} # vertex mapping for tracking edges later
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value # ep is short for edge_properties

    # Done, finally!
    return gtG