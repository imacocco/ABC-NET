from .graph_operations import *
import graph_tool.stats as gtt
import graph_tool.inference as gti
import graph_tool.clustering as gtc
from glob import glob


def extract_observables_from_path(source_dir: str, d_max: int, r: float = 0.5, ext: str = 'gt.xz'):
    """Extract and average features/observables/summary statistics from multiple graph realizations
    Args:
        source_dir: directory containing graphs
        d_max: largest distance to look at
        r: ratio between radii to be used in the computation of intrinsic dimension
        ext: extension used to save graphs
    Returns:
    """
    # ID, portrait, <k>, <k**2>, diameter, average clustering coeff, modularity
    source_dir.rstrip('/')
    source_dir = source_dir + '/'
    gr_files = glob(source_dir + '*' + ext)
    n = len(gr_files)
    if n == 0:
        print("no graph found in " + source_dir + " directory, returning")
        return
    # prepare observables
    ids = []
    port = []
    kmax = []
    diam = []
    cc = []
    mod = []
    ek = []
    # cycle over networks
    for g in gr_files:
        G = gt.load_graph(g)
        # ID and portrait
        I3D,_ = gt_to_ide(G, elems=None, d_max=d_max+1)
        ids.append(I3D.return_id_scaling(np.arange(1, d_max+1), plot=False)[0])
        portrait = portrait_from_distances(I3D.distances)
        port.append(portrait)
        kmax.append(portrait.shape[1])
        # <k>, <k**2>
        ek.append(moments_from_portrait(portrait, [1, 2])[:, 1])
        # Diameter
        diam.append(max([gtt.pseudo_diameter(G)[0], gtt.pseudo_diameter(G, source=G.num_vertices()-1)[0],
                         gtt.pseudo_diameter(G, source=G.num_vertices()/2)[0]]))
        # ave clus coeff
        cc.append(gtc.global_clustering(G)[0])
        # modularity 
        state = gti.minimize_blockmodel_dl(G)
        mod.append(gti.modularity(G,state.get_blocks()))
        
    idx = np.argmax(kmax)
    
    for i in range(n):
        if i == idx:
            continue
        port[i], port[idx] = pad_portraits_to_same_size(port[i], port[idx])

    np.savetxt(source_dir+'ID.dat', np.c_[np.mean(ids, axis=0), np.std(ids, axis=0)], header='ID\tstd(ID)', fmt='%.3f')
    np.savetxt(source_dir+'portrait.dat', np.mean(port, axis=0), fmt='%.1f')
    np.savetxt(source_dir+'scalar_obs.dat',
               np.c_[np.mean(ek,axis=0)[0], np.mean(ek,axis=0)[1], np.mean(diam), np.mean(cc), np.mean(mod)],
               header='<k>\t<k**2>\tdiam\tcc\tmod', fmt='%.5f')

        
"""
Once you have saved the observables you can retrieve them like this: 

ID = np.zeros((num_priors,d_max+1))
ID_err = np.zeros((num_priors,d_max+1))
Scalar_obbs = np.zeros((num_priors,5))
for i in range(len(priors)):
    ID[i], ID_err[i] = np.loadtxt('my_dir/prior_'+str(i)+'/ID.dat',unpack=True)
    Scalar_obbs[i] = np.loadtxt('my_dir/prior_'+str(i)+'/scalar_obs.dat')
"""