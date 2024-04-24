# Functions to operate over existing networks to exacts observables

import numpy as np
import os

from dadapy import IdDiscrete
from scipy.stats import entropy

import graph_tool as gt
from graph_tool.topology import shortest_distance as sd
import graph_tool.topology as gtt
import graph_tool.clustering as gtc
import graph_tool.inference as gti
import graph_tool.centrality as gty

from typing import Callable

rng = np.random.default_rng()


######################### Extract observables ###################################

def extract_observables(G: gt.Graph, compute_id=True, d_max=None):
    # save summary statistics into dictionary
    obs = dict()

    # number of vertices
    obs["n_vertices"] = G.num_vertices()
    
    # number of edges
    obs["n_edges"] = G.num_edges()
    
    # degree distribution
    obs["degree"] = G.get_out_degrees(G.get_vertices())

    # diameter
    obs["diameter"] = int(max([gtt.pseudo_diameter(G)[0], gtt.pseudo_diameter(G, source=G.num_vertices() - 1)[0],
                               gtt.pseudo_diameter(G, source=G.num_vertices()/2)[0]]))

    if d_max is None:
        d_max = obs["diameter"]
    
    # local clustering
    obs["local_clustering"] = np.array(gtc.local_clustering(G).a)

    # average clustering coefficient
    obs["clustering_coefficient"] = gtc.global_clustering(G)[0]
    
    # page rank
    obs["page_rank"] = np.array(gty.pagerank(G).a)
    
    # betweennes
    obs["betweennes"] = np.array((gty.betweenness(G)[0]).a)

    # closeness
    obs["closeness"] = np.array(gty.closeness(G).a)

    # eigenvector centrality
    obs["eigenvector"] = np.array((gty.eigenvector(G,max_iter=1000)[1]).a)
    # obs["eigenvector"] = eigs( gts.adjacency(G), k=50, which='LR')  

    # modularity
    obs["modularity"] = gti.modularity(G, gti.minimize_blockmodel_dl(G).get_blocks())

    # intrinsic dimension
    if compute_id:
        I3D, _ = gt_to_ide(G, elems=None, d_max=d_max + 1)
        obs["ID"] = I3D.return_id_scaling(np.arange(1, d_max + 1), plot=False, r=0.9)[0]

    # network portrait
    # obs["portrait"] = portrait_from_distances(I3D.distances)

    # moments of degree distribution
    # obs["Ek"] = np.mean(I3D.distances[:, 1] - 1)
    # obs["Ek2"] = np.mean((I3D.distances[:, 1] - 1) ** 2)

    return obs


########################### ID related operations ###################################


def gt_to_ide(G: gt.Graph, elems = 1000, d_max: int = 50, dense: bool = False):
    """
    Starting from a graph-tool network, build the cumulative number of neighbours at each distance for given nodes
    
    The function returns an IdDiscrete object and a list of the used nodes (if subsampling is performed).
    Since all-to-all distances calculation in a network with N vertices can be of order O(N^3), 
    it is possible to use just some vertices to compute the ID, passing a number n or a list
    with keyword elems. Typically 10-20% of the toal number of nodes are enough to obtain an ID 
    very close to the complete one.
        
    Args:
        G: input graph
        elems: (int or list/np.ndarray(int)): specific nodes or number of nodes to be used to build the IdDiscrete object
        d_max: furthest distance considered
        dense: change short-path distance algorithm depending on density of network

    """

    # select subset of vertices used for the ID estimation
    if elems is None:               # use all vertices
        indexes = G.get_vertices()

    elif isinstance(elems, int):    # use random nodes
        if elems >= G.num_vertices():
            indexes = G.get_vertices()
        else:                       # use the nodes provided (as a list of integers)
            indexes = rng.choice(G.get_vertices(), elems, replace=False)

    elif isinstance(elems, np.ndarray):
        assert max(elems) < G.num_vertices(), "elems would select out of range vertices"
        indexes = elems
        elems = len(indexes)

    else:
        print('use a proper format for elems')

    # initialise I3D object
    ID = IdDiscrete(np.zeros(shape=(len(indexes), 1)), condensed=True, is_network=True)
    ID.distances = np.zeros(shape=(len(indexes), d_max + 1), dtype=int)

    # compute shortest-path distances and store them into the dadapy IdDiscrete object
    for i, el in enumerate(indexes):
        # if i%100==0:
        # print(i,end='\r')
        dist = sd(G, source=G.vertex(el), max_dist=d_max, dense=dense)
        
        uniq, counts = _my_counter(dist.a, d_max)
        ID.distances[i, uniq] = counts
        ID.distances[i] = np.cumsum(ID.distances[i])

        """
        # alternative (slightly slower) version using numpy.unique and possibly returning the indices of neighbours

        idx = np.argsort(dist.a).astype(int)
        dist = dist.a[idx].astype(int)
        if condensed:
            uniq, counts = np.unique(dist, return_counts=True)
            mask = uniq<=d_max
            ID.distances[i, uniq[mask]] = np.copy(counts[mask])
            ID.distances[i] = np.cumsum(ID.distances[i])
        elif len(dist)<maxk:
            ID.distances[i] = np.copy( np.r_[ dist,np.full(maxk-len(dist),np.inf) ] )
            ID.dist_indeces[i] = np.copy( np.r_[ idx,np.full(maxk-len(idx),np.nan) ] )
        else:
            ID.distances[i] = np.copy( dist[:maxk] )
            ID.dist_indeces[i] = np.copy( idx[:maxk]) 
        """

    return ID, indexes


################ Graph portrait related operations ################################

def portrait_from_distances(distances):
    """Portrait from single vertex distances
    
    Extract graph portrait representation  (see Citation J. P. Bagrow et al 2008 EPL 81 68004
    DOI 10.1209/0295-5075/81/68004) from matrix of cumulative number of neighbours for each vertex
    i.e. IdDiscrete.distances

    Args:
        distances (np.ndarray(int,int)): matrix of cumulative number of neighbours for each vertex
    Returns:
        Portrait_representation
    """
    N, d = distances.shape
    temp = distances[:, 1:] - distances[:, :-1]
    temp = np.c_[np.ones(N, dtype=int).T, temp]
    m = temp.max()
    return np.array([np.histogram(temp[:, j], range=(-0.5, m + 0.5), bins=m + 1)[0] for j in range(d)])


def moments_from_portrait(GP, moment_order=1):
    """
    Extract moments of degree distributions at different scales
    
    Args:
        GP: graph portrait
        moment_order (int or list(int)): moments to be extracted from degree distribution at different scales
    Returns:
        M (np.ndarray(floats)): moments of deg distribution at any scale
    """
    if isinstance(moment_order, (int, float)):
        moment_order = [moment_order]

    return np.array([np.sum(np.arange(GP.shape[1]) ** m * GP, axis=1) / GP[0, 1] for m in moment_order])


def pad_portraits_to_same_size(B1, B2):
    """Make sure that two matrices are padded with zeros and/or trimmed of
    zeros to be the same dimensions.
    
    Args:
        B1 (np.ndarray(float,float)): portrait 1
        B2 (np.ndarray(float,float)): portrait 2
    Returns:
        BigB1 (np.ndarray(float,float)): extended portrait 1
        BigB2 (np.ndarray(float,float)): extended portrait 2
    """
    ns, ms = B1.shape
    nl, ml = B2.shape

    # Bmats have N columns, find last *occupied* column and trim both down:
    lastcol1 = max(np.nonzero(B1)[1])
    lastcol2 = max(np.nonzero(B2)[1])
    lastcol = max(lastcol1, lastcol2)
    B1 = B1[:, :lastcol + 1]
    B2 = B2[:, :lastcol + 1]

    BigB1 = np.zeros((max(ns, nl), lastcol + 1))
    BigB2 = np.zeros((max(ns, nl), lastcol + 1))

    BigB1[:B1.shape[0], :B1.shape[1]] = B1
    BigB2[:B2.shape[0], :B2.shape[1]] = B2

    return BigB1, BigB2


def my_portrait_divergence(BG, BH):
    """
    Compute the portrait divergence between graphs portraits BG and BH.

    Args:
        BG, BH (np.ndarray(float,float)): Two graphs to compare.

    Returns:
        JSDpq (float): the Jensen-Shannon divergence between the portraits BG and BH

    """
    if BG.shape != BH.shape:
        BG, BH = pad_portraits_to_same_size(BG, BH)

    L, K = BG.shape
    V = np.tile(np.arange(K), (L, 1))

    XG = BG * V / (BG * V).sum()
    XH = BH * V / (BH * V).sum()

    # flatten distribution matrices as arrays:
    P = XG.ravel()
    Q = XH.ravel()

    # lastly, get JSD:
    M = 0.5 * (P + Q)
    KLDpm = entropy(P, M, base=2)
    KLDqm = entropy(Q, M, base=2)
    JSDpq = 0.5 * (KLDpm + KLDqm)

    return JSDpq


######################## SAVE GRAPHS ############################


def save_graphs(model: Callable[..., gt.Graph], samples: int, path: str, params, *args, **kwargs):
    """Save realizations of given models for further analysis
    Args:
        model: a function generating a gt.Graph()
        samples: number of graphs to be saved with the same generating parameters
        path: directory where to save the graphs
        params (list(str)): name of parameters of generating model passed in args or kwargs
    Returns:
    
    Example of usage:
        save_graphs(build_barabasi_albert, 10, 'my_directory', ['N','m','gamma','c'], n, m, g, c)
        
        Parallel(n_jobs=10)( delayed(save_graphs)(build_barabasi_albert, 10,\
        target_dir+'prior_'+str(i),['N','m','gamma','c'],n, m, *prior_[i]) for i in trange(num_priors))
    """
    path.rstrip('/')
    path = path + '/'
    try:
        os.makedirs(path, exist_ok=True)
        # print("Directory '%s' created successfully" %directory)
    except OSError as error:
        print("Directory '%s' can not be created")

    with open(path + 'notes.txt', 'w') as file:
        for j, el in enumerate(params):
            file.write(el + ' = ' + format(args[j], '.5f') + '\n')
        file.write('trials = ' + str(samples))

    for i in range(samples):
        G = model(*args, **kwargs)
        G.save(path + str(i) + '.gt.xz')


##################### HELPER FUNCTIONS #########################


def _my_counter(x, dmax: int):
    """
    Fast way to count unique elements in array with few different values
    Args:
        x (list or np.ndarray): starting vector
        dmax: highest element considered
    Returns:
        ux[0] np.ndarray: vector of unique elements
        counts: np.ndarray(int): number of repetitions for each element
    """
    x = x[x <= dmax]
    z = dmax * x
    counts = np.bincount(z)
    pos = counts.nonzero()[0]
    ux = np.divmod(pos, dmax)
    return ux[0], counts[pos]


def GT_to_IDE(G,elems=1000,condensed=True,maxk=1000,d_max=100,dense=False):
    
    from graph_tool.topology import shortest_distance as sd
    
    n = G.num_vertices()
    maxk = min(n-1,maxk)
    
    # select subset of vertex used for the ID estimation
    # all points will still be used to compute observables
    if elems is None:
        indexes = G.get_vertices()
    elif isinstance(elems,int):
        if elems>=n:
            indexes = G.get_vertices()
        else:
            indexes = rng.choice(G.get_vertices(),elems,replace=False)
    elif isinstance(elems,np.ndarray):
        indexes = elems
        elems = len(indexes)
    else:
        print('use a proper format for elems')
    
    # initialise I3D object
    ID = IdDiscrete(np.zeros(shape=(len(indexes),1)),maxk=maxk,condensed=condensed,is_network=True)
    
    if condensed:
        ID.distances = np.zeros(shape=(len(indexes),d_max+1),dtype=int)#,dtype=int)
    else:
        ID.distances = np.zeros(shape=(len(indexes),maxk))#,dtype=int)
        ID.dist_indeces = np.zeros(shape=(len(indexes),maxk))#,dtype=int)
    
    # if there are few elements
    # if n*elems<10:#5e8:
    #     print('all together')
    #     v_list = [G1.vertex(i) for i in indexes]
    #     dist = sd(G,target=v_list,max_dist=d_max,dense=dense)      
    #     for i,el in enumerate(indexes):
    #         idx = np.argsort(dist[G.vertex(el)].a)
    #         tmp = tmp.a[idx].astype(int)
    #         uniq, counts = np.unique(tmp, return_counts=True,axis=1)
    #         ID.distances[i, uniq[mask]] = np.copy(counts[mask])
    #         ID.distances[i] = np.cumsum(ID.distances[i])
    # else:        
    for i,el in enumerate(indexes):
#        if i%100==0:
#            print(i,end='\r')
        dist = sd(G,source=G.vertex(el),max_dist=d_max,dense=dense)
        idx = np.argsort(dist.a).astype(int)
        dist = dist.a[idx].astype(int)
        if condensed:
            uniq, counts = np.unique(dist, return_counts=True)
            mask = uniq<=d_max
            ID.distances[i, uniq[mask]] = np.copy(counts[mask])
            ID.distances[i] = np.cumsum(ID.distances[i])
        elif len(dist)<maxk:
            ID.distances[i] = np.copy( np.r_[ dist,np.full(maxk-len(dist),np.inf) ] )
            ID.dist_indeces[i] = np.copy( np.r_[ idx,np.full(maxk-len(idx),np.nan) ] )
        else:
            ID.distances[i] = np.copy( dist[:maxk] )
            ID.dist_indeces[i] = np.copy( idx[:maxk]) 
            
    return ID, indexes