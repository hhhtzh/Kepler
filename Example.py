
from cmath import sin

import numpy
import numpy as np
from algorithm.evolutionary_computing.age_fitness import AgeFitnessEA
from algorithm.evolutionary_computing.evaluation.evaluation import Evaluation
from algorithm.evolutionary_computing.evolutionary_optimizers.island import Island
from local_optimizers.scipy_optimizer import ScipyOptimizer
from local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from encoder.graph.component_generator import ComponentGenerator
from operators.generator import AGraphGenerator
from operators.crossover import AGraphCrossover
from operators.mutation import AGraphMutation
from algorithm.evolutionary_computing.symbolic_regression.explicit_regression import ExplicitRegression, \
    ExplicitTrainingData
from algorithm.evolutionary_computing.symbolic_regression.string_parse import eq_string_to_command_array_and_constants

from util.log import configure_logging

from algorithm.evolutionary_computing.symbolic_regression.initial import GraphAutoInitial, OperonTreeInitial

from window.window import Window
import pyoperon

# def main():
#     a = GraphAutoInitial(-10, 10, 30, '{x}*2 + 3.5*{x}*3')
#     a.test()
import random, time, sys, os, json
import pandas as pd
from scipy import stats

import pyoperon as Operon
from pmlb import fetch_data
import pandas as pd1




def main():


    a=Window()
    a.init()

if __name__ == "__main__":
    main()

