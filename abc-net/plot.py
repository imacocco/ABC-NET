import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


def plot_sequential_posterior(priors, distances, eps: float = None, quantile=None,
                              d_min: int = 0, d_max: int = None, means: bool = True, ref=None):
    """Plot posterior for scale dependent quantities, passed as a list
    Args:
        priors (list or np.ndarray): list of priors
        distances (list or np.ndarray): list of distances
        eps (float): radius for acceptance
        quantile (float): fraction of closest priors accepted to build the postrior
        d_min: smaller scale to plot
        d_max: largest scale to plot
        means: whether to plot means of posteriors
        ref: if provided, plot also ground truth values
    Returns:    
    """
    assert len(priors) == len(distances)
    N, dims = priors.shape
    
    sup = len(distances[0])
    if d_max > sup:
        d_max = sup        
    
    scales = np.arange(d_min+1, d_max+1)
        
    idxs = np.argsort(distances, axis=0)

    # fix radius of acceptance (max tolerance distance)
    if eps is not None:
        posteriors = np.array([priors[distances[:, i] < eps] for i in range(d_min, d_max)])
        acc_rates = [len(pd_i)/N for pd_i in posteriors]
        print('acceptance rate: ', acc_rates)
    # fix fraction of closest points to take into account
    elif quantile is not None:
        posteriors = np.array([priors[idxs[:int(quantile*N), i]] for i in range(d_min, d_max)])
    else:
        raise ValueError("give a value for the tolerance or the fraction \
                        of closest points to be taken into consideration")
        
    if ref is not None or means:
        inf_x = min([min(pi[:, 0]) for pi in posteriors])
        sup_x = max([max(pi[:, 0]) for pi in posteriors])
        inf_y = min([min(pi[:, 1]) for pi in posteriors])
        sup_y = max([max(pi[:, 1]) for pi in posteriors])

    # 2-dimensional scatter plot
    if dims == 2:
        plt.figure()
        plt.title("Posterior at growing scale")
        for i in range(d_max-d_min):
            plt.scatter(posteriors[i][:, 0], posteriors[i][:, 1], alpha=0.3, label='scale='+str(scales[i]))
            
        if means and sup == 1:
            post_means = np.mean(posteriors[0], axis=0)
            plt.plot(post_means[0]*np.ones(2), (inf_y, sup_y), 'k--', label='post mean')
            plt.plot((inf_x, sup_x), post_means[1]*np.ones(2), 'k--')
                
        if ref is not None:
            plt.plot(ref[0]*np.ones(2), (inf_y, sup_y), 'r--', label='ground truth')
            plt.plot((inf_x, sup_x), ref[1]*np.ones(2), 'r--')

        plt.legend()
        
    # posteriors at growing scale
    for j in range(dims):
        plt.figure()
        appo = [plt.hist(posteriors[i][:, j], alpha=0.3, label='scale='+str(scales[i]),
                         bins=50, density=True) for i in range(d_max-d_min)]
        sup_y = max([max(n[0]) for n in appo])
        
        if means and sup == 1:
            plt.plot(post_means[j]*np.ones(2), (0, sup_y), 'k--', label='posterior mean')
            
        if ref is not None:
            plt.plot(ref[j]*np.ones(2), (0, sup_y), 'r--', label='ground truth')
            
        plt.legend()
        

def plot_joint(priors, distances, eps: float = None, quantile: float = None, ref=None,
               plot_priors: bool = False, fileout: str = None):
    """Plot posterior for scale dependent quantities, passed as a list
    Args:
        priors (list or np.ndarray): list of priors
        distances (list or np.ndarray): list of distances
        eps (float): radius for acceptance
        quantile (float): fraction of closest priors accepted to build the posterior
        ref: if provided, plot also ground truth values
        plot_priors: whether to plot priors
        fileout: if provided, saves the plot
    Returns:    
    """
    # initial checks
    assert len(priors) == len(distances)
    N, dims = priors.shape

    # fix radius of acceptance (max tolerance distance)
    if eps is not None:
        posteriors = priors[distances<eps]
        acc_rates = len(posteriors)/N 
        print('acceptance rate: ', acc_rates)
    # fix fraction of closest points to take into account
    elif quantile is not None:
        idxs = np.argsort(distances, axis=0)[:int(quantile*N)]
        posteriors = priors[idxs]
    else:
        raise ValueError("give a value for the tolerance or the fraction of"
                         " closest points to be taken into consideration")
        
    x = posteriors[:, 0]
    y = posteriors[:, 1]
    
    # df = pd.DataFrame(posteriors, columns=['x', 'y'])

    xa = np.mean(x)
    xm = np.median(x)

    ya = np.mean(y)
    ym = np.median(y)
    
    if plot_priors:
        yi = np.min(priors[:, 1])
        ys = np.max(priors[:, 1])
        xi = np.min(priors[:, 0])
        xs = np.max(priors[:, 0])
    else:
        yi = np.min(posteriors[:, 1])
        ys = np.max(posteriors[:, 1])
        xi = np.min(posteriors[:, 0])
        xs = np.max(posteriors[:, 0])

    # JOINTPLOT OBJECT
    g = sns.JointGrid(x=x, y=y, space=0.2, ratio=3, xlim=(xi-0.01, xs+0.01),
                      ylim=(yi-0.01, ys+0.01)) # ,joint_kws={},marginal_kws={'bins':40})
    # g = sns.JointGrid(df,space=0.2,ratio=3)# ,joint_kws={},marginal_kws={'bins':40})

    # JOINTPLOT
    g.plot_joint(sns.kdeplot, fill=True, cbar=True, levels=100)
    # sns.kdeplot(df,x='x',y='y',fill=True,cbar=True,levels=100,ax=g.ax_joint)
    
    if plot_priors:
        g.ax_joint.scatter(priors[:, 0], priors[:, 1], s=10, alpha=0.2)
        
    g.ax_joint.scatter(x, y, alpha=0.5, s=10)
    ## means and medians
    g.ax_joint.axvline(x=xa, linestyle='--', label='post mean', c='orangered', lw=2, alpha=0.6)
    g.ax_joint.axhline(y=ya, linestyle='--', c='orangered', lw=2, alpha=0.6)
    g.ax_joint.axvline(x=xm, linestyle='-.', label='post median', c='seagreen', lw=2, alpha=0.6)
    g.ax_joint.axhline(y=ym, linestyle='-.', c='seagreen', lw=2, alpha=0.6)
    #g.ax_joint.set_xlim(-0.01,0.7)
    #g.ax_joint.set_ylim(-0.01,0.8)

    if ref is not None:
         g.ax_joint.scatter(ref[0], ref[1], c='red', label='ground truth')
            
    g.ax_joint.legend(frameon=False, bbox_to_anchor=(1.5, 1.3), loc="upper right")
    g.ax_joint.set_xlabel('param 1')
    g.ax_joint.set_ylabel('param 2')

    # MARGINALS
    g.plot_marginals(sns.histplot, alpha=0.4, kde=True, bins=50, binrange=(0, 0.8))
    #sns.histplot(df, x='x',alpha=0.4,kde=True,bins=50,binrange=(0,0.7),ax=g.ax_marg_x)
    #sns.histplot(df, y='y',alpha=0.4,kde=True,bins=50,binrange=(0,0.8),ax=g.ax_marg_y)
    
    ## means and medians
    g.ax_marg_x.axvline(x=xa, linestyle='--', c='orangered', lw=2, alpha=0.6)
    g.ax_marg_y.axhline(y=ya, linestyle='--', c='orangered', lw=2, alpha=0.6)
    g.ax_marg_x.axvline(x=xm, linestyle='-.', c='seagreen', lw=2, alpha=0.6)
    g.ax_marg_y.axhline(y=ym, linestyle='-.', c='seagreen', lw=2, alpha=0.6)

    # COLORBAR
    ## below
    # g.fig.colorbar(g.ax_joint.collections[0], ax=[g.ax_joint, g.ax_marg_y, g.ax_marg_x], use_gridspec=True, orientation='horizontal')
    ## at the right
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    pos_joint_ax = g.ax_joint.get_position()
    pos_marg_x_ax = g.ax_marg_x.get_position()
    g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    g.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    
    if fileout is not None:
        plt.tight_layout()
        plt.savefig(fileout)
    else:
        plt.show()