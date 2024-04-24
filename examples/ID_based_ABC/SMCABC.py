#!/usr/bin/env python
import sys
import os
import json
sys.path.append(os.getcwd()) 
import graph_tool as gt

import numpy as np
rng = np.random.default_rng()

sys.path.append('../ABC-NET/abc-net/')
import graph_operations as go
import pyabc
import custom_pyabc as cp

def main():
    # handle input
    if len(sys.argv) != 2:
        print("You need to specify the directory location where to perform the simulation")
        return 0
    else:
        path = os.path.abspath(sys.argv[1])+'/'

    try:
        if os.path.isfile(os.path.join(path,'parameters.json')):
            with open(os.path.join(path,'parameters.json'),'r') as f:
                parameters = json.load(f)
    except OSError as error:
        print(error) 

    def ID_max(x, x0):
        return np.max(abs(x["ID"][:parameters["R_max"]] - x0["ID"][:parameters["R_max"]]))

    # path to external executable, in this csae a Julia package for ABNG
    origin = os.path.abspath('./')
    exe_dir = os.path.abspath('./julia_packages/')
    exe = 'exe_empty.jl'
    n_procs = 12 

    # model passed to PyABC object
    def model_ext(par):
        # convert params
        par_ext = list(par.values())
        par_ext.append(1.)
        rng = np.random.default_rng(seed=int(max(par_ext[:-1])*1e12))
        seed = rng.integers(2000000000)
               
        filename = os.path.join(path,'sim_' + str(seed)+'.edges')
        
        # perform simulation
        os.chdir(exe_dir)
        os.system('julia ' + exe +' \"' + str(par_ext) + '\" '+filename)
        G = gt.load_graph_from_csv(filename,hashed=False,directed=False, csv_options={"delimiter": ",", "quotechar": '"'})
        # compute id
        I3D, _ = go.gt_to_ide(G, elems=None, d_max=21)
        # add ID to dictionary
        observation = {"ID": I3D.return_id_scaling(np.arange(1, 21), plot=False)[0], "seed": seed}
        # return to main code directory
        os.chdir(origin)
        return observation

    # initialize the parameters of the model, depending on which model you use
    keys = ["p1","p2","p3","p4","p5","p6","p7","p8"]
    # build a suitable prior
    prior = cp.Dirichlet_marg(parameters["dirichlet_params"][:-1],parameters["dirichlet_params"][-1])
    
    # sampler = pyabc.sampler.MulticoreParticleParallelSampler()
    sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=n_procs)
    # sampler = pyabc.sampler.DaskDistributedSampler()
    # sampler = pyabc.sampler.SingleCoreSampler()
    sampler.show_progress=True
    
    abc = pyabc.ABCSMC(model_ext, prior, ID_max, 
                       population_size=parameters["smc_target_pop"],
                       sampler=sampler, 
                       # use a custom transition of any contraint needs to be imposed among the parameters,
                       # in this case they have to sum to 1
                       transitions=cp.my_MultivariateNormalTransition(),
                       # eps=pyabc.epsilon.QuantileEpsilon(initial_epsilon=parameters["smc_start_eps"])
                       )

    db_path = os.path.join(path, "data.db")
    if os.path.exists(db_path):
    	os.remove(db_path)  
    db = "sqlite:///" + db_path

    gt_observation = {"ID": np.loadtxt(path+'id_ref.dat'), "seed":0}
    
    # in case you want to resume a previous run
    # abc.load(db)

    #in case you want to start a new run
    abc.new(db, observed_sum_stat=gt_observation)

    history = abc.run(minimum_epsilon=parameters["smc_min_eps"],
                      max_nr_populations=parameters["smc_max_nr_pop"],
                      max_total_nr_simulations=50000,
                      )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
