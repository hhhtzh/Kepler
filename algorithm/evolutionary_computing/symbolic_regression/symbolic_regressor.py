import itertools
from abc import abstractmethod, ABCMeta
from time import time

import numpy as np
import os
import signal
from warnings import warn

from joblib import Parallel
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, RegressorMixin, clone, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.utils import compute_sample_weight
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_array

from algorithm.evolutionary_computing.evaluation.evaluation import Evaluation
from algorithm.evolutionary_computing.age_fitness import AgeFitnessEA
from algorithm.evolutionary_computing.evolutionary_optimizers.fitness import _Fitness, _fitness_map
from algorithm.evolutionary_computing.evolutionary_optimizers.fitness_predictor_island import FitnessPredictorIsland
from algorithm.evolutionary_computing.evolutionary_optimizers.island import Island
from encoder.tree.parralle_evolve import _parallel_evolve
from algorithm.evolutionary_computing.symbolic_regression import ExplicitTrainingData, ExplicitRegression
from algorithm.evolutionary_computing.symbolic_regression.stats.hall_of_fame import HallOfFame
from encoder.graph.component_generator import ComponentGenerator
from local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from local_optimizers.scipy_optimizer import ScipyOptimizer
from operators.crossover import AGraphCrossover
from operators.generator import AGraphGenerator
from operators.mutation import AGraphMutation
from util.randoms import check_random_state
from util.utils import _partition_estimators
from .functions import sig1 as sigmoid
from algorithm.evolutionary_computing.symbolic_regression.functions import _function_map, _Function

MAX_INT = np.iinfo(np.int32).max


class BingoSymbolicRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, *, population_size=500, stack_size=32,
                 operators=None, use_simplification=False,
                 crossover_prob=0.4, mutation_prob=0.4,
                 metric="mse", parallel=False, clo_alg="lm",
                 generations=int(1e19), fitness_threshold=1.0e-16,
                 max_time=1800, max_evals=int(1e19),
                 evolutionary_algorithm=AgeFitnessEA,
                 clo_threshold=1.0e-8, scale_max_evals=False,
                 random_state=None):
        self.population_size = population_size
        self.stack_size = stack_size

        if operators is None:
            operators = {"+", "-", "*", "/"}
        self.operators = operators

        self.use_simplification = use_simplification

        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.metric = metric

        self.parallel = parallel

        self.clo_alg = clo_alg

        self.generations = generations
        self.fitness_threshold = fitness_threshold
        self.max_time = max_time
        self.max_evals = max_evals
        self.scale_max_evals = scale_max_evals

        self.evolutionary_algorithm = evolutionary_algorithm

        self.clo_threshold = clo_threshold

        self.best_ind = None

        self.random_state = random_state

    def set_params(self, **params):
        # TODO not clean
        new_params = self.get_params()
        new_params.update(params)
        super().set_params(**new_params)
        self.__init__(**new_params)
        return self

    def _get_archipelago(self, X, y, n_processes):
        self.component_generator = ComponentGenerator(X.shape[1])
        for operator in self.operators:
            self.component_generator.add_operator(operator)

        self.crossover = AGraphCrossover()
        self.mutation = AGraphMutation(self.component_generator)

        self.generator = AGraphGenerator(self.stack_size, self.component_generator,
                                         use_simplification=self.use_simplification,
                                         use_python=True  # only using c++ backend
                                         )

        local_opt_fitness = self._get_clo(X, y, self.clo_threshold)
        evaluator = Evaluation(local_opt_fitness, n_processes)

        if self.evolutionary_algorithm == AgeFitnessEA:
            ea = self.evolutionary_algorithm(evaluator, self.generator, self.crossover,
                                             self.mutation, self.crossover_prob, self.mutation_prob,
                                             self.population_size)

        else:  # DeterministicCrowdingEA
            ea = self.evolutionary_algorithm(evaluator, self.crossover, self.mutation,
                                             self.crossover_prob, self.mutation_prob)

        # TODO pareto front based on complexity?
        hof = HallOfFame(5)

        island = self._make_island(len(X), ea, hof)
        self._force_diversity_in_island(island)

        # if self.parallel:
        #     return ParallelArchipelago(island, hall_of_fame=hof)
        # else:
        return island

    def _make_island(self, dset_size, ea, hof):
        if dset_size < 1200:
            return Island(ea, self.generator, self.population_size, hall_of_fame=hof)
        return FitnessPredictorIsland(ea, self.generator, self.population_size, hall_of_fame=hof,
                                      predictor_size_ratio=800 / dset_size)

    def _get_clo(self, X, y, tol):
        training_data = ExplicitTrainingData(X, y)
        fitness = ExplicitRegression(training_data=training_data, metric=self.metric)

        optimizer = ScipyOptimizer(fitness, method=self.clo_alg, tol=tol)
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        return local_opt_fitness

    def _force_diversity_in_island(self, island):
        diverse_pop = []
        pop_strings = set()

        i = 0
        while len(diverse_pop) < self.population_size:
            i += 1
            ind = self.generator()
            ind_str = str(ind)
            if ind_str not in pop_strings or i > 15 * self.population_size:
                pop_strings.add(ind_str)
                diverse_pop.append(ind)
        print(f" Generating a diverse population took {i} iterations.")
        island.population = diverse_pop

    def get_best_individual(self):
        if self.best_ind is None:
            raise ValueError("Best individual not set")
        return self.best_ind

    # TODO should return self
    def fit(self, X, y, sample_weight=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        if sample_weight is not None:
            print("sample weight not None, TODO")
            raise NotImplementedError

        n_cpus = os.environ.get('OMP_NUM_THREADS', 1)

        print(f"using {n_cpus} processes")
        self.archipelago = self._get_archipelago(X, y, n_cpus)
        print(f"archipelago: {type(self.archipelago)}")

        max_eval_scaling = 1
        if isinstance(self.archipelago, FitnessPredictorIsland):
            print("n_points per predictor:", self.archipelago._predictor_size)
            if self.scale_max_evals:
                max_eval_scaling = len(X) / self.archipelago._predictor_size / 1.1
        # print("max evals:", self.max_evals * max_eval_scaling)

        opt_result = self.archipelago.evolve_until_convergence(
            max_generations=self.generations,
            fitness_threshold=self.fitness_threshold,
            max_time=self.max_time,
            max_fitness_evaluations=self.max_evals * max_eval_scaling,
            convergence_check_frequency=10
        )

        if len(self.archipelago.hall_of_fame) == 0:  # most likely found sol in 0 gens
            self.best_ind = self.archipelago.get_best_individual()
        else:
            self.best_ind = self.archipelago.hall_of_fame[0]
        print(f"done with opt, best_ind: {self.best_ind}, fitness: {self.best_ind.fitness}")

        # rerun CLO on best_ind with tighter tol
        fit_func = self._get_clo(X, y, tol=1e-6)
        best_fitness = fit_func(self.best_ind)
        best_constants = tuple(self.best_ind.constants)
        for _ in range(5):
            self.best_ind._needs_opt = True
            fitness = fit_func(self.best_ind)
            if fitness < best_fitness:
                best_fitness = fitness
                best_constants = tuple(self.best_ind.constants)
        self.best_ind.fitness = best_fitness
        self.best_ind.set_local_optimization_params(best_constants)
        print(f"reran CLO, best_ind: {self.best_ind}, fitness: {self.best_ind.fitness}")

        # print("------------------hall of fame------------------", self.archipelago.hall_of_fame, sep="\n")
        # print("\nbest individual:", self.best_ind)

    def predict(self, X):
        best_ind = self.get_best_individual()
        output = best_ind.evaluate_equation_at(X)

        # convert nan to 0, inf to large number, and -inf to small number
        return np.nan_to_num(output, posinf=1e100, neginf=-1e100)


class TimeOutException(Exception):
    pass


def alarm_handler(signum, frame):
    print("raising TimeOutException")
    raise TimeOutException


class CrossValRegressor(GridSearchCV):
    # TODO there's probably a better way to set Symbolic
    #   Regressor's params than this and set_max_time().
    #   For the sake of time, doing this instead
    def set_params(self, **params):
        if "test" in params.keys():
            if params["test"] is True:
                self.set_max_time(1)
            params.pop("test")
        super().set_params(**params)

    def get_best_individual(self):
        return self.best_estimator_.get_best_individual()

    def set_max_time(self, new_max_time):
        self.estimator.set_params(**{"max_time": new_max_time})

    def fit(self, X, y=None, *, groups=None, **fit_params):
        signal.signal(signal.SIGALRM, alarm_handler)
        if len(X) <= 1000:  # probably not good to hardcode these values
            signal.alarm(60 * 60 - 5)  # 1 hour with 5 seconds of slack
            # signal.alarm(5)  # 1 hour with 5 seconds of slack
        else:
            signal.alarm(10 * 60 * 60 - 5)  # 10 hours with 5 seconds of slack
        try:
            super().fit(X, y, groups=groups, **fit_params)
        except TimeOutException as e:
            print("Got a TimeOutException")

            try:
                self.best_estimator_
            except AttributeError:  # no self.best_estimator
                # if we didn't get to the end of cv,
                # set best_estimator_ to the estimator passed in with the first
                # set of hyperparams
                print("Setting best estimator to first set of hyperparams")
                self.best_estimator_ = clone(self.estimator.set_params(**self.param_grid[0]))

            print(self.best_estimator_)
            return self


class GpBaseSymbolic(BaseEstimator, metaclass=ABCMeta):
    """Base class for symbolic regression / classification estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=None,
                 n_components=None,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 transformer=None,
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):

        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.transformer = transformer
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.feature_names = feature_names
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            print('    |{:^25}|{:^42}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 42 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>10}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', 'OOB Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            oob_fitness = 'N/A'
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>10}'
            if self.max_samples < 1.0:
                oob_fitness = run_details['best_oob_fitness'][-1]
                line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     oob_fitness,
                                     remaining_time))

    def fit(self, X, y, sample_weight=None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        # Check arrays
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if isinstance(self, ClassifierMixin):
            X, y = self._validate_data(X, y, y_numeric=False)
            check_classification_targets(y)

            if self.class_weight:
                if sample_weight is None:
                    sample_weight = 1.
                # modify the sample weights with the corresponding class weight
                sample_weight = (sample_weight *
                                 compute_sample_weight(self.class_weight, y))

            self.classes_, y = np.unique(y, return_inverse=True)
            n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
            if n_trim_classes != 2:
                raise ValueError("y contains %d class after sample_weight "
                                 "trimmed classes with zero weights, while 2 "
                                 "classes are required."
                                 % n_trim_classes)
            self.n_classes_ = len(self.classes_)

        else:
            X, y = self._validate_data(X, y, y_numeric=True)

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame,
                                                        self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components,
                                                     self.hall_of_fame))

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, ClassifierMixin):
            if self.metric != 'log loss':
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, TransformerMixin):
            if self.metric not in ('pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if not ((isinstance(self.const_range, tuple) and
                 len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, '
                             'or None.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        if self.feature_names is not None:
            if self.n_features_in_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_in_,
                                    len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either '
                                 '"sigmoid" or _Function object, got %s' %
                                 type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                 'got %d.' % (self._transformer.arity))

        params = self.get_params()
        params['_metric'] = self._metric
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not '
                 'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        for gen in range(prior_generations, self.generations):

            start_time = time()

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population = Parallel(n_jobs=n_jobs,
                                  verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          sample_weight,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)

            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            if self._metric.greater_is_better:
                best_program = population[np.argmax(fitness)]
            else:
                best_program = population[np.argmin(fitness)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # Check for early stopping
            if self._metric.greater_is_better:
                best_fitness = fitness[np.argmax(fitness)]
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                best_fitness = fitness[np.argmin(fitness)]
                if best_fitness <= self.stopping_criteria:
                    break

        if isinstance(self, TransformerMixin):
            # Find the best individuals in the final generation
            fitness = np.array(fitness)
            if self._metric.greater_is_better:
                hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
            else:
                hall_of_fame = fitness.argsort()[:self.hall_of_fame]
            evaluation = np.array([gp.execute(X) for gp in
                                   [self._programs[-1][i] for
                                    i in hall_of_fame]])
            if self.metric == 'spearman':
                evaluation = np.apply_along_axis(rankdata, 1, evaluation)

            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = np.abs(np.corrcoef(evaluation))
            np.fill_diagonal(correlations, 0.)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))
            # Iteratively remove least fit individual of most correlated pair
            while len(components) > self.n_components:
                most_correlated = np.unravel_index(np.argmax(correlations),
                                                   correlations.shape)
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                indices = list(range(len(components)))
            self._best_programs = [self._programs[-1][i] for i in
                                   hall_of_fame[components]]

        else:
            # Find the best individual in the final generation
            if self._metric.greater_is_better:
                self._program = self._programs[-1][np.argmax(fitness)]
            else:
                self._program = self._programs[-1][np.argmin(fitness)]

        return self


class GpSymbolicRegressor(GpBaseSymbolic, RegressorMixin):

    """A Genetic Programming symbolic regressor.

    A symbolic regressor is an estimator that begins by building a population
    of naive random formulas to represent a relationship. The formulas are
    represented as tree-like structures with mathematical functions being
    recursively applied to variables and constants. Each successive generation
    of programs is then evolved from the one that came before it by selecting
    the fittest individuals from the population to undergo genetic operations
    such as crossover, mutation or reproduction.

    Parameters
    ----------
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

        Available individual functions are:

        - 'add' : addition, arity=2.
        - 'sub' : subtraction, arity=2.
        - 'mul' : multiplication, arity=2.
        - 'div' : protected division where a denominator near-zero returns 1.,
          arity=2.
        - 'sqrt' : protected square root where the absolute value of the
          argument is used, arity=1.
        - 'log' : protected log where the absolute value of the argument is
          used and a near-zero argument returns 0., arity=1.
        - 'abs' : absolute value, arity=1.
        - 'neg' : negative, arity=1.
        - 'inv' : protected inverse where a near-zero argument returns 0.,
          arity=1.
        - 'max' : maximum, arity=2.
        - 'min' : minimum, arity=2.
        - 'sin' : sine (radians), arity=1.
        - 'cos' : cosine (radians), arity=1.
        - 'tan' : tangent (radians), arity=1.

    metric : str, optional (default='mean absolute error')
        The name of the raw fitness metric. Available options include:

        - 'mean absolute error'.
        - 'mse' for mean squared error.
        - 'rmse' for root mean squared error.
        - 'pearson', for Pearson's product-moment correlation coefficient.
        - 'spearman' for Spearman's rank-order correlation coefficient.

        Note that 'pearson' and 'spearman' will not directly predict the target
        but could be useful as value-added features in a second-step estimator.
        This would allow the user to generate one engineered feature at a time,
        using the SymbolicTransformer would allow creation of multiple features
        at once.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicTransformer

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(GpSymbolicRegressor, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def predict(self, X):
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples]
            Predicted values for X.

        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicRegressor not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        y = self._program.execute(X)

        return y



if __name__ == '__main__':
    # SRSerialArchipelagoExample with SymbolicRegressor
    import random

    # random.seed(7)
    # np.random.seed(7)
    x = np.linspace(-10, 10, 1000).reshape([-1, 1])
    y = x ** 2 + 3.5 * x ** 3

    regr = BingoSymbolicRegressor(population_size=100, stack_size=10,
                             operators=["+", "-", "*"],
                             use_simplification=True,
                             crossover_prob=0.4, mutation_prob=0.4, metric="mae",
                             parallel=False, clo_alg="lm", generations=500, fitness_threshold=-1,
                             evolutionary_algorithm=AgeFitnessEA, clo_threshold=1.0e-4, random_state=7)

    hyper_params = [
        {"population_size": [100], "stack_size": [24]},
        {"population_size": [500], "stack_size": [24]},
        {"population_size": [2500], "stack_size": [32]}
    ]

    print(regr.get_params())
    regr.set_params()
    print(regr)

    regr.fit(x, y)
    print(regr.get_best_individual())

    # TODO MPI.COMM_WORLD.bcast in parallel?
    # TODO rank in MPI
