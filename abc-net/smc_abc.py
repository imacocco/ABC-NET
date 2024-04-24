# Routines to simplify the interaction with PyABC.
# The class allows for multiple realization with the same parameters and return averaged observables


from collections.abc import Callable
from graph_operations import extract_observables

class Model:
    def __init__(self, generator: Callable, d_max: int = None, n_replicas: int = 1, **fixed_args):
        self.generator = generator
        self.fixed_args = fixed_args
        self.d_max = d_max
        self.n_replicas = n_replicas

    def sample(self, varying_args):
        return self.generator(*list(self.fixed_args.values()), *list(varying_args.values()))

    def __call__(self, par):

        obs_dict = extract_observables(self.sample(par), True, self.d_max)

        if self.n_replicas > 1:
            # add values from successive replicas
            [obs_dict.update({k: obs_dict[k]+v}) for k, v in
                zip(obs_dict, list(extract_observables(self.sample(par), self.d_max).values()))
                for _ in range(self.n_replicas-1)
             ]
            # take the mean
            [obs_dict.update({k: obs_dict[k]/self.n_replicas}) for k in obs_dict]

        return obs_dict