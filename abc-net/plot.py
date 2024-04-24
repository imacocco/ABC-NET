# Plotting functions to visualize the SMC-ABC results

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pyabc.visualization as pv


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


def plot_smc_evo(h, m=0, xmin=None, xmax=None, averages=True, p_true=True, t0=0, t1=None, fileout=None):
    """Plot the evolution of parameters in subsequent SMCABC simulations
    Args:
        h (pyabc.history): data structure with saved data
        m (int): model to show
        xmin (list/np.array(float)): left xlim
        xmax (list/np.array(float)): right xlim
        averages (bool): whether to print averages and stds of posteriors
        p_true (bool/dict): if given, print the ground truth value of the generative model
        t0 (int): first SMC generation to be plotted
        t1 (int): last SMC to be plotted
        fileout (str): filename to save the plot
    """

    prior_ids = list(h.get_distribution(m=0,t=0)[0].columns)
    n = len(prior_ids)

    # if isinstance(xmin, (list, np.ndarray, tuple)):
    #     assert len(xmin)==n, "if providing array of boundaries, it has to match the number of parameters"
    # elif xmin is None:
    #     x_min = [None for _ in n]
    # else:   # single number, make it equal for 
    #     xmin = np.ones(n)*xmin

    max_cols = 5
    
    rows = n//max_cols+1
    cols = min(n,max_cols)
    
    fig, axs = plt.subplots(rows,cols,figsize=(cols*5,rows*5))
           
    if t1 is None:
        t1 = h.max_t + 1
    else:
        t1 = min(h.max_t + 1, t1)
    assert t1 > t0
    
    if averages:
        last_gen = h.get_population(t1-1)
        particles = np.array([list(ai.parameter.values()) for ai in last_gen.particles])
        w = np.array(list(last_gen.get_for_keys(['weight']).values())).reshape(len(particles))
        avg = np.average(particles,weights=w,axis=0)
        std = np.sqrt(np.average((particles-avg)**2, weights=w,axis=0))
        avg_dict = dict(zip(prior_ids,avg))
        std_dict = dict(zip(prior_ids,std))

    alpha = np.linspace(0.3,0.7,t1-t0-1)
    alpha = np.append(alpha,1.)
    # cycle over parameters
    for axi, pi in zip(fig.axes,prior_ids):
    	# cycle over generations
        for t in range(t0,t1):
            df, w = h.get_distribution(m=m, t=t)

            if len(w) == 0:  # model has been dropped
            	break

            pv.plot_kde_1d(
                df,
                w,
                xmin=xmin,
                xmax=xmax,
                x=pi,
                xname=pi,
                ax=axi,
                label=f"PDF @ t={t}",
                alpha=alpha[t-t0]
            )
        axi.set_ylabel('Posterior probability density',fontsize=15)
        axi.set_xlabel(pi,fontsize=15)
        axi.tick_params(axis='x', labelsize=13)
        axi.tick_params(axis='y', labelsize=13)
                      
        # plot ground truth values if available
        if p_true and len(h.get_ground_truth_parameter())>0:
            axi.axvline(h.get_ground_truth_parameter()[pi], #0.3,
                color="darkgreen", linestyle="dashed", label="ground truth")
        elif type(p_true) is dict:
            axi.axvline(p_true[pi], color="darkgreen", linestyle="dashed", label="ground truth")
                      
        # plot posterior averages
        if averages:
            axi.axvline(avg_dict[pi], color="darkorange", linestyle="dashed", label="post mean")
            axi.axvspan(avg_dict[pi]-std_dict[pi], avg_dict[pi]+std_dict[pi], color='orange', alpha=0.3)
        if axi == fig.axes[0]:
            axi.legend(frameon = False, fontsize=13);

    [fig.delaxes(fig.axes[-1]) for i in range(n,cols*rows)]

    if fileout:
        plt.tight_layout()
        plt.savefig(fileout)

def plot_smc_joint_evo(h, m=0, p1='p1', p2='p2', xmin=None, xmax=None, ymin=None, ymax=None, 
        averages=True, p_true=True, t0=0, t1=None, simplex=False, fileout=None):

    prior_ids = list(h.get_distribution(m=0,t=0)[0].columns)
    
    assert p1 in prior_ids, "par " +p1+ " is not present in given history"
    assert p2 in prior_ids, "par " +p2+ " is not present in given history"
    
    if t1 is None:
        t1 = h.max_t + 1
    else:
        t1 = min(h.max_t + 1, t1)
    assert t1 > t0
    
    gt_dic = None
    if p_true and len(h.get_ground_truth_parameter())>0:
        gt_dic = dict(h.get_ground_truth_parameter())
    elif type(p_true) is dict:
        gt_dic = p_true

    max_cols = 4
    n = t1 - t0
    rows = n//max_cols+1
    cols = min(n,max_cols)
    
    fig, axs = plt.subplots(rows,cols,figsize=(cols*3,rows*2.5))
    
    for i,t in enumerate(range(t0,t1)):
        ax = fig.axes[i]
        
        if simplex:
            particles = np.array([list(ai.parameter.values())[1:] for ai in h.get_population(t=t).particles])
            plot_simplex(particles, gt=gt_dic, ax=ax)
            
        else:
            pv.plot_kde_2d(
                *h.get_distribution(m=0, t=t),
                p1,p2,
                xmin=xmin, xmax=xmax, numx=100,
                ymin=ymin, ymax=ymax, numy=100,
                ax=ax,
            )

            # plot reference (ground truth values)
            #if p_true and len(h.get_ground_truth_parameter())>0:
            #    ax.scatter([h.get_ground_truth_parameter()[p1]],[h.get_ground_truth_parameter()[p2]], color="r", label="gt")
            #elif type(p_true) is dict:
            #    ax.scatter([p_true[p1]],[p_true[p2]], color="r", label="gt")
            if p_true or type(p_true) is dict:
                #ax.scatter([gt_dic[p1]],[gt_dic[p2]], color="r", label="gt")
                ax.scatter(0.3,0.3, color="r", label="gt")
                
                
            # plot posterior averages
            if averages:

                gen = h.get_population(t=t)
                particles = np.array([list(ai.parameter.values()) for ai in gen.particles])
                w = np.array(list(gen.get_for_keys(['weight']).values())).reshape(len(particles))
                avg = np.average(particles,weights=w,axis=0)
                std = np.sqrt(np.average((particles-avg)**2, weights=w,axis=0))
                avg_dict = dict(zip(prior_ids,avg))
                std_dict = dict(zip(prior_ids,std))

                ax.axvline(avg_dict[p1], color="darkorange", linestyle="dashed", label="post mean",alpha=0.4)
                ax.axvspan(avg_dict[p1]-std_dict[p1], avg_dict[p1]+std_dict[p1], color='orange', alpha=0.2)

                ax.axhline(avg_dict[p2], color="darkorange", linestyle="dashed",alpha=0.4)
                ax.axhspan(avg_dict[p2]-std_dict[p2], avg_dict[p2]+std_dict[p2], color='orange', alpha=0.2)

        if ax == fig.axes[0]:
            ax.legend(frameon = False);
        elif simplex and gt_dic is not None:
            ax.get_legend().remove()

        ax.set_title(f"Generation t={t}")
    
    [fig.delaxes(fig.axes[-1]) for i in range(n,cols*rows)]
    fig.tight_layout()

    if fileout is not None:
        if fileout.endswith(".png"):
            dpi = 300
        else:
            dpi = None
        plt.savefig(fileout,dpi=dpi)
    
def plot_simplex(X, gt=None, edges=True, fileout=None, ax=None,**kwargs):
    
    # find vertices for simplex projection
    dims = X.shape[1]
    vertices = []
    offset = np.pi/dims*(1 - 0.5*dims%2)
    for i in range(0,dims+1):
        ang = 2*np.pi*float(i)/dims + offset
        vertices.append([np.cos(ang),np.sin(ang)])
    vertices = np.array(vertices)
    xlim = (min(vertices[:,0]),max(vertices[:,0]))
    ylim = (min(vertices[:,1]),max(vertices[:,1]))
    
    # project points on simplex
    X_proj = np.dot(X,vertices[:-1])
    # plot kde smoothing
    g = sns.kdeplot(x=X_proj[:,0], y=X_proj[:,1],
        fill=True, levels=100, cmap="viridis", ax=ax, **kwargs)
    #plot points in simplex
    sns.scatterplot(x=X_proj[:,0], y=X_proj[:,1],alpha=0.2, ax=ax, **kwargs)
    
    if gt is not None: 
        if type(gt) is dict:
            gt = np.array(list(gt.values()))
        X_gt_proj = np.dot(gt,vertices[:-1])
        sns.scatterplot(x=[X_gt_proj[0]],y=[X_gt_proj[1]],c='r',label='ground truth', ax=ax, **kwargs)
    
    #plt.legend()
    if edges and ax:
        ax.plot(vertices[:,0],vertices[:,1],c='k',lw=0.5, alpha=1)#,**kwargs)
        ax.axis('equal')
    elif edges:
        plt.plot(vertices[:,0],vertices[:,1],c='k',lw=0.5, alpha=1)#,**kwargs)
        plt.axis('equal')
    
    
    g.set(xticklabels=[])
    g.set(xticks=[])
    g.set(yticklabels=[])
    g.set(yticks=[])
    sns.despine(left=True,bottom=True)
    # plt.axis('equal', **kwargs)
    # plt.xlim(xlim, **kwargs)
    # plt.ylim(ylim, **kwargs)
    # plt.axis('off', **kwargs)
    
    if fileout:
        plt.savefig(fileout)
    
    

def plot_ID(h,ss='ID',t0=0, t1=None, type='avg', t_gen=None, fileout=None):

    # check the selected summary statistics has been recorded
    assert ss in list(h.get_population(0).get_accepted_sum_stats()[0].keys()), "select proper summary statistics"

    assert type in ['avg','evo','all'], "type has to be 'avg', 'evo' or 'all'"

    obs = np.array(list(h.observed_sum_stat()[ss]))
    R = range(1,len(obs)+1)
    R_int = range(1,len(list(h.get_population(0).get_accepted_sum_stats()[0][ss]))+1)

    if t1 is None:
        t1 = h.max_t + 1
    else:
        t1 = min(h.max_t + 1, t1)
    assert t1 > t0

    if type == 'evo':   # plot avg of selected ss at different generations
        plt.plot(R,obs,label='ground truth', color='orange', linewidth=3, alpha=0.8)
        for t in range(t0,t1):
            gen = h.get_population(t)
            w = np.array(list(gen.get_for_keys(['weight']).values())).reshape(len(gen.particles))
            # plot the ID associated with the particle which is closest to the average
            # particles = np.array([list(ai.parameter.values()) for ai in gen.particles])
            # avg = np.average(particles,weights=w,axis=0)
            # idx = np.argmin(np.sum( (particles-avg)**2, axis=1 ) )
            # samp = np.array(list(gen.get_accepted_sum_stats()[idx][ss]))
            # plt.plot(range(1,len(samp)),samp,label=f't={t}')
            # plot average of IDs
            samp = np.array([i[ss] for i in gen.get_accepted_sum_stats()])
            avg = np.average(samp,weights=w,axis=0)
            plt.plot(range(1,len(avg)+1),avg,label=f't={t}')

    else:               # type is 'avg', plot avg and std of ss at last gen
        
        if t_gen is None:
            t_gen = t1 - 1
        else:
            assert 0<=t_gen<=h.max_t, "select proper generation between 0 and " +str(t1-1)
            
        gen = h.get_population(t_gen)
        particles = np.array([list(ai.parameter.values()) for ai in gen.particles])
        w = np.array(list(gen.get_for_keys(['weight']).values())).reshape(len(particles))

        # take points closest to the posterior mean
        # tot_avg = np.average(particles,weights=w,axis=0)
        # idxs = np.argsort(np.sum( (particles-tot_avg)**2, axis=1 ) )[:50]

        # take points with lowest epsilon
        idxs = np.argsort( list(gen.get_for_keys(['distance']).values()) )[0]#[:50]

        # extract subset of weights and summary statistics
        samp = np.array([gen.get_accepted_sum_stats()[idx][ss] for idx in idxs])
        w = w[idxs]
        # and compute its average
        avg = np.average(samp,weights=w,axis=0)

        if type == 'all':   # plot ensemble of ss
            [plt.plot(range(1,len(smp)+1),smp,color='grey',alpha=0.4, linewidth=1) for smp in samp]
    
        else:   # type == 'avg'
            # find standard deviation
            std = np.sqrt(np.average((samp-avg)**2, weights=w,axis=0))
            plt.fill_between(R_int,avg-std,avg+std,color='orange',alpha=0.5)
        
        plt.plot(R,obs,label='ground truth', color='orange', linewidth=3, alpha=0.8)

        plt.plot(range(1,len(avg)+1),avg,label='weighted avgerage', linewidth=2.5, alpha=0.8)
        
        

    plt.xlabel("Scale",fontsize=16)
    plt.ylabel("ID estimated",fontsize=16)
    plt.legend(frameon=False)
    if fileout is not None:
        plt.tight_layout()
        plt.savefig(fileout)

        
def plot_KS_boxes(h, t=None,keys=None, fileout=None):
    from scipy.stats import ks_2samp as KS
    if t is None:
        t = h.max_t
    else:
        t = min(h.max_t, t)
        
    pop = h.get_population(t)
    obs_ = h.observed_sum_stat()
    if keys is None:
        keys = list(obs_.keys())
        
    particles = np.array([list(ai.parameter.values()) for ai in pop.particles])
    
    ks_dists = dict()
    for k in keys:
        #if k == 'ID' or k =='eigen_centr':
        #    continue
        ks_dists[k] = np.array([KS(obs_[k],el[k])[0] for el in pop.get_accepted_sum_stats()])
        
    plt.figure(figsize=(6,6))    
    plt.boxplot(ks_dists.values(), labels=ks_dists.keys(), patch_artist=True);
    plt.xticks(rotation=45,fontsize=16)
    plt.yticks(fontsize=14)
    plt.ylabel("KS distances",fontsize=16)
    
    if fileout is not None:
        plt.tight_layout()
        plt.savefig(fileout)
        
