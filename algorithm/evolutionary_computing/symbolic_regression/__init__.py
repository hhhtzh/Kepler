"""
Import the core names of bingo symbolic_regression library

Programs that want to build bingo symbolic regression apps
without having to import specific modules can import this.
"""
import warnings


# Try to load in C++ cpython extensions
# TODO: consider this init file but remove imports for python
# that have C++ bindings
try:
    from bingocpp import AGraph, \
                         ExplicitRegression, \
                         ExplicitTrainingData, \
                         ImplicitRegression, \
                         ImplicitTrainingData
    ISCPP = True
except ImportError as import_err:
    from encoder.graph.agraph import AGraph
    from .explicit_regression import ExplicitRegression, ExplicitTrainingData
    from .implicit_regression import ImplicitRegression, ImplicitTrainingData
    ISCPP = False
    warnings.warn(f"Could not load C++ modules {import_err}")
