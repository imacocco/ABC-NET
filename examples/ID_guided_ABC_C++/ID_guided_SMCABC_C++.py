#!/usr/bin/env python

import numpy as np
rng = np.random.default_rng()
import pyabc
import os
import sys
sys.path.append('../ABC-NET/abc-net/')

import custom_pyabc as cp

def ID_max(x, x0):
    sup = min( len(x["ID"]), len(x0["ID"]) )
    return np.max(abs(x["ID"][:sup] - x0["ID"][:sup]))

def ID_sq(x, x0):
    sup = min(len(x["ID"]),len(x0["ID"]))
    return np.sum((x["ID"][:sup] - x0["ID"][:sup])**2)

def ID_strana(x, x0):
    if x['edges'] < x0['edges']:
        return 1000
    else:
        return x['id_dist']


def main():
    # handle input
    if len(sys.argv) == 1:
        print("You need to specify the directory location where to perform the simulation")
        return 0
    else:
        path = os.path.abspath(sys.argv[1])+'/'
        assert isinstance(path,str), "the directory need to be a string"

    origin = os.path.abspath('./')
    id_path = os.path.join(path, 'id_ref.dat')
    obs = {"ID": np.loadtxt(id_path), "seed": -1, "steps": 0, "vertices": 500, "edges": 1036, "id_dist": 0}

    exe_dir = os.path.abspath('../C++/')
    exe = './MCMC.x'
    # save list of directory in order of creation so that we can rebuild history
    open(os.path.join(path, 'num_list.txt'),'w')

    def model_ext(par):
        # convert params
        par_ext = list(par.values())
        # par_ext.append(1.)
        rng = np.random.default_rng(seed=int(par_ext[0]*1e12))
        seed = rng.integers(2000000000)
        
        with open(os.path.join(path, 'num_list.txt'),'a') as ff:
            ff.write("%d\n"%seed)
               
        path_i = os.path.join(path,'sim_' + str(seed))
        os.mkdir(path_i)
        
        with open(os.path.join(path_i, 'input.dat'), 'w') as fileout:
            fileout.write(id_path+'\n')
            fileout.write("%d\n"%seed)
            for i in par_ext:
                fileout.write("%.5f\n"%i)

        # perform simulation
        os.chdir(exe_dir)
        os.system(exe+' '+path_i+'/')
        # load last step of output
        out = np.array(os.popen('tail -1 '+os.path.join(path_i,'output.dat')).readline().split(' ')[:-1]).astype(float)
        # save into dictionary for pyabc
        res = dict()
        res['seed'] = seed
        res['steps'] = out[0].astype(int)
        res['vertices'] = out[2].astype(int)
        res['edges'] = out[3].astype(int)
        res['id_dist'] = out[4]
        res["ID"] = out[5:]
        os.chdir(origin)
        return res

    # keep only beta if you want only random moves (slower convergence)
    # if you want to add other moves 
    	# - uncomment the other parameters
    	# - select the Dirichlet_plus_one prior
    	# - put the right transition: cp.my_second_MultivariateNormalTransition()
    	# - comment the par_ext.append(1.) line in model_ext
    	# - uncomment eps=pyabc.epsilon.QuantileEpsilon(initial_epsilon=10.0), as it allows to have a good initial round where
    	#   only graphs with the right number of edges is kept

    keys = ["beta","p1","p2","p3","p4"]
    prior = cp.Dirichlet_plus_one((1,1,1,1),1.4,(5000,25000))
    # prior = pyabc.Distribution(beta=pyabc.RV("uniform", 1000, 25000))

    sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=4)
    #sampler = pyabc.sampler.SingleCoreSampler()
    sampler.show_progress=True
    abc = pyabc.ABCSMC(model_ext, prior, ID_strana, population_size=50, 
        sampler=sampler, 
        transitions=cp.my_second_MultivariateNormalTransition(),
        eps=pyabc.epsilon.QuantileEpsilon(initial_epsilon=10.0)
        )
    
    db_path = os.path.join(path, "data.db")
    db = "sqlite:///" + db_path

    abc.new(db, observed_sum_stat=obs)
    #abc.load(db)

    history = abc.run(minimum_epsilon=0.05, max_nr_populations=10)

    return 0

if __name__ == "__main__":
    sys.exit(main())
