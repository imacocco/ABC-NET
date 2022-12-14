from collections.abc import Callable
from .graph_operations import extract_observables


class Model:
    def __init__(self, generator: Callable, **fixed_args):
        self.generator = generator
        self.fixed_args = fixed_args

    def sample(self, varying_args):
        return self.generator( *list(self.fixed_args.values()), *list(varying_args.values()) )

    def __call__(self, par):
        G = self.sample(par)
        return extract_observables(G)

