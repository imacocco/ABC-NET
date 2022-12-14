from setuptools import setup, Extension

cmdclass = {}


class get_numpy_include(object):
    """Defer numpy.get_include() until after numpy is installed.
    From: https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    """

    def __str__(self):
        import numpy

        return numpy.get_include()


ext_modules = []

setup(
    name="abc-net",
    version="0.0.0",
    # url="https://dadapy.readthedocs.io/",
    description="A Python package for parameter inference and model validation on generative network models",
    # long_description="A Python package for Distance-based Analysis of DAta-manifolds.",
    packages=["abc-net"],
    install_requires=["numpy", "scipy", "scikit-learn", "matplotlib", "graph-tool", "pyabc", "dadapy", 'networkx'],
    # extras_require={"dev": ["tox", "black", "isort", "pytest"]},
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    # include_package_data=True,
    # package_data={'dadapy': ['_utils/discrete_volumes/*.dat']},
)
