from pmlb import fetch_data

import random, time, sys, os, json
import numpy as np
import pandas as pd
from scipy import stats

import pyoperon as Operon
from pmlb import fetch_data
from window.calculate import formula_format, final_calc
from data.data import Data
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
from algorithm.evolutionary_computing.symbolic_regression.symbolic_regressor import GpSymbolicRegressor
import pyoperon as Operon
from util.log import configure_logging
from util.randoms import check_random_state
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from operators.generator import AGraphGenerator
from operators.crossover import AGraphCrossover
from operators.mutation import AGraphMutation
from algorithm.evolutionary_computing.symbolic_regression.stats.pareto_front import ParetoFront


def agraph_similarity(ag_1, ag_2):
    """a similarity metric between agraphs"""
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()


class Initial:
    def __init__(self, population_size=50,
                 mutation_probability=0.4, crossover_probability=0.4,
                 error_tolerance=1e-6):
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.error_tolerance = error_tolerance


class BgGraphInitial(Initial):
    def __init__(self, x, y, train_p=0.3, num=1, population_size=5000,
                 mutation_probability=0.4, crossover_probability=0.4, stopping_criteria=0.01, p_subtree_mutation=0.1,
                 error_tolerance=1e-6, random_seed=0, if_print_train=1, parsimony_coefficient=0.01,
                 p_point_mutation=0.1, generations=20, p_hoist_mutation=0.05):
        super().__init__(population_size, mutation_probability, crossover_probability, error_tolerance)
        print(x)
        print(y)
        self.x_train = np.array(x[1:], dtype=np.float64)
        self.y_train = np.array(y[1:], dtype=np.float64)

    def do(self):
        print(type(self.x_train))
        print(self.x_train.shape)
        training_data = ExplicitTrainingData(self.x_train, self.y_train)
        component_generator = ComponentGenerator(input_x_dimension=self.x_train.shape[1])
        component_generator.add_operator("+")
        component_generator.add_operator("-")
        component_generator.add_operator("*")
        component_generator.add_operator("/")
        component_generator.add_operator("sin")
        crossover = AGraphCrossover()
        mutation = AGraphMutation(component_generator)
        STACK_SIZE = 10
        agraph_generator = AGraphGenerator(STACK_SIZE, component_generator)
        fitness = ExplicitRegression(training_data=training_data)
        optimizer = ScipyOptimizer(fitness, method="lm")
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        evaluator = Evaluation(local_opt_fitness)

        ea = AgeFitnessEA(
            evaluator,
            agraph_generator,
            crossover,
            mutation,
            mutation_probability=self.mutation_probability,
            crossover_probability=self.crossover_probability,
            population_size=self.population_size,
        )
        island = Island(ea, agraph_generator, self.population_size)
        print("-----  Generation %d  -----" % island.generational_age)
        print("Best individual:     ", island.get_best_individual())
        print("Best fitness:        ", island.get_best_fitness())
        print("Fitness evaluations: ", island.get_fitness_evaluation_count())
        island.evolve_until_convergence(
            max_generations=100, fitness_threshold=self.error_tolerance
        )
        island = Island(ea, agraph_generator, self.population_size)
        print("-----  Generation %d  -----" % island.generational_age)
        print("Best individual:     ", island.get_best_individual())
        print("Best fitness:        ", island.get_best_fitness())
        print("Fitness evaluations: ", island.get_fitness_evaluation_count())


class OperonTreeInitial(Initial):
    def __init__(self, csv_filename, train_p=0.3, num=1, population_size=5000,
                 mutation_probability=0.4, crossover_probability=1, stopping_criteria=0.01, p_subtree_mutation=0.1,
                 error_tolerance=1e-5, random_seed=0, if_print_train=1, parsimony_coefficient=0.01,
                 p_point_mutation=0.1, generations=20, p_hoist_mutation=0.05, time_limit=86400, minL=1, maxL=50,
                 maxD=10,threads=16,evaluator_budget=1000000):
        super().__init__(population_size, mutation_probability, crossover_probability, error_tolerance)
        self.csv_filename = csv_filename
        self.train_p = train_p
        self.generations = generations
        self.random_seed = random_seed
        self.time_limit = time_limit
        self.minL = minL
        self.maxL = maxL
        self.maxD = maxD
        self.threads=threads

    def do(self):
        D = pd.read_csv(self.csv_filename).to_numpy()
        # initialize a dataset from a numpy array
        ds = Operon.Dataset(D)

        # define the training and test ranges
        training_range = Operon.Range(0, int(ds.Rows * self.train_p))
        test_range = Operon.Range(int(ds.Rows * self.train_p), ds.Rows)

        # define the regression target
        target = ds.Variables[-1]  # take the last column in the dataset as the target

        # take all other variables as inputs
        inputs = Operon.VariableCollection(v for v in ds.Variables if v.Name != target.Name)
        # initialize a rng
        rng = Operon.RomuTrio(random.randint(1, 1000000))

        # initialize a problem object which encapsulates the data, input, target and training/test ranges
        problem = Operon.Problem(ds, inputs, target.Name, training_range, test_range)
        # initialize an algorithm configuration
        config = Operon.GeneticAlgorithmConfig(generations=self.generations, max_evaluations=1000000,
                                               local_iterations=0,
                                               population_size=self.generations, pool_size=1000,
                                               p_crossover=self.crossover_probability,
                                               p_mutation=self.mutation_probability,
                                               epsilon=self.error_tolerance, seed=self.random_seed,
                                               time_limit=self.time_limit)

        # use tournament selection with a group size of 5
        # we are doing single-objective optimization so the objective index is 0
        selector = Operon.TournamentSelector(objective_index=0)
        selector.TournamentSize = 5

        # initialize the primitive set (add, sub, mul, div, exp, log, sin, cos), constants and variables are implicitly added
        pset = Operon.PrimitiveSet()
        pset.SetConfig(
            Operon.PrimitiveSet.Arithmetic | Operon.NodeType.Exp | Operon.NodeType.Log | Operon.NodeType.Sin | Operon.NodeType.Cos)

        # define tree length and depth limits
        minL, maxL = self.minL, self.maxL
        maxD = self.maxD

        # define a tree creator (responsible for producing trees of given lengths)
        btc = Operon.BalancedTreeCreator(pset, inputs, bias=0.0)
        tree_initializer = Operon.UniformLengthTreeInitializer(btc)
        tree_initializer.ParameterizeDistribution(minL, maxL)
        tree_initializer.MaxDepth = maxD

        # define a coefficient initializer (this will initialize the coefficients in the tree)
        coeff_initializer = Operon.NormalCoefficientInitializer()
        coeff_initializer.ParameterizeDistribution(0, 1)

        # define several kinds of mutation
        mut_onepoint = Operon.NormalOnePointMutation()
        mut_changeVar = Operon.ChangeVariableMutation(inputs)
        mut_changeFunc = Operon.ChangeFunctionMutation(pset)
        mut_replace = Operon.ReplaceSubtreeMutation(btc, coeff_initializer, maxD, maxL)

        # use a multi-mutation operator to apply them at random
        mutation = Operon.MultiMutation()
        mutation.Add(mut_onepoint, 1)
        mutation.Add(mut_changeVar, 1)
        mutation.Add(mut_changeFunc, 1)
        mutation.Add(mut_replace, 1)

        # define crossover
        crossover_internal_probability = 0.9  # probability to pick an internal node as a cut point
        crossover = Operon.SubtreeCrossover(crossover_internal_probability, maxD, maxL)

        # define fitness evaluation
        interpreter = Operon.Interpreter()  # tree interpreter
        error_metric = Operon.R2()  # use the coefficient of determination as fitness
        evaluator = Operon.Evaluator(problem, interpreter, error_metric,
                                     True)  # initialize evaluator, use linear scaling = True
        evaluator.Budget = 1000 * 1000  # computational budget
        evaluator.LocalOptimizationIterations = 0  # number of local optimization iterations (coefficient tuning using gradient descent)

        # define how new offspring are created
        generator = Operon.BasicOffspringGenerator(evaluator, crossover, mutation, selector, selector)

        # define how the offspring are merged back into the population - here we replace the worst parents with the best offspring
        reinserter = Operon.ReplaceWorstReinserter(objective_index=0)
        gp = Operon.GeneticProgrammingAlgorithm(problem, config, tree_initializer, coeff_initializer, generator,
                                                reinserter)

        # report some progress
        gen = 0
        max_ticks = 50
        interval = 1 if config.Generations < max_ticks else int(np.round(config.Generations / max_ticks, 0))
        t0 = time.time()


        # run the algorithm
        gp.Run(rng, threads=self.threads)

        # get the best solution and print it
        best = gp.BestModel
        model_string = Operon.InfixFormatter.Format(best.Genotype, ds, 6)
        print('回归结果为:\n')
        print(f'\n{model_string}')
        print('回归结束\n')


class GraphAutoInitial(Initial):
    def __init__(self, start, stop, num_points, eq_str, population_size=1000, mutation_probability=0.4,
                 crossover_probability=0.4, error_tolerance=1e-6):
        super().__init__(population_size, mutation_probability, crossover_probability, error_tolerance)
        self.mutation = None
        self.crossover = None
        self.component_generator = None

        self.x = np.linspace(start, stop, num_points).reshape([-1, 1])
        self.eq_str = eq_str
        self.y = self.x

    def init(self):
        np.random.seed(4)
        n1 = np.array(object=list)
        n1.reshape([-1, 1])
        for ty in self.y:
            a = float(ty)
            eq_str1 = self.eq_str.format(x=str(a))
            formula_list = formula_format(eq_str1)
            result, _ = final_calc(formula_list)
            b = np.float64(result[0])
            n1 = np.append(n1, np.array(b))

        n1 = np.delete(n1, 0, 0)
        n1 = n1.reshape([-1, 1])
        n1 = n1.astype(np.float64)
        x = np.linspace(-10, 10, 30).reshape([-1, 1])
        y = x * 2 + 3.5 * x * 3

        print(self.x)
        print(n1)
        training_data = ExplicitTrainingData(self.x, n1)

        plt.plot(training_data.x, training_data.y, 'ro')
        plt.show()

        component_generator = ComponentGenerator(input_x_dimension=self.x.shape[1])
        component_generator.add_operator("+")
        component_generator.add_operator("-")
        component_generator.add_operator("*")

        agraph_generator = AGraphGenerator(agraph_size=10,
                                           component_generator=component_generator)
        agraph = agraph_generator()
        print("f(X_0) = ", agraph)

        crossover = AGraphCrossover()
        mutation = AGraphMutation(component_generator)

        fitness = ExplicitRegression(training_data=training_data)
        optimizer = ScipyOptimizer(fitness, method='lm')
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        evaluator = Evaluation(local_opt_fitness)
        np.random.seed(16)
        agraph = agraph_generator()
        print("Before local optimization: f(X_0) = ", agraph)
        print("                          fitness = ", fitness(agraph))
        _ = local_opt_fitness(agraph)
        print("After local optimization:  f(X_0) = ", agraph)
        print("                          fitness = ", fitness(agraph))

        agraph_y = agraph.evaluate_equation_at(training_data.x)

        plt.plot(training_data.x, training_data.y, 'ro')
        plt.plot(training_data.x, agraph_y, 'b-')
        plt.show()

        POPULATION_SIZE = self.population_size
        MUTATION_PROBABILITY = 0.4
        CROSSOVER_PROBABILITY = 0.4

        ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation,
                          CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, POPULATION_SIZE)

        def agraph_similarity(ag_1, ag_2):
            """a similarity metric between agraphs"""
            return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

        pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                                   similarity_function=agraph_similarity)

        np.random.seed(5)
        island = Island(ea, agraph_generator, POPULATION_SIZE, hall_of_fame=pareto_front)
        print("Best individual\n f(X_0) =", island.get_best_individual())

        ERROR_TOLERANCE = 1e-6

        best_indv_values = []
        best_indv_values.append(island.get_best_individual())
        best_indv_gen = []
        best_indv_gen.append(island.generational_age)

        while island.get_best_fitness() > ERROR_TOLERANCE:
            island.evolve(1)
            best_indv = island.get_best_individual()
            if best_indv.fitness < best_indv_values[-1].fitness:
                best_indv_values.append(best_indv)
                best_indv_gen.append(island.generational_age)

        print("Generation: ", island.generational_age)
        print("Success!")
        print("Best individual\n f(X_0) =", island.get_best_individual())

        print(" FITNESS   COMPLEXITY    EQUATION")
        for member in pareto_front:
            print("%.3e     " % member.fitness, member.get_complexity(),
                  "     f(X_0) =", member)

        def animate_data(list_of_best_indv, list_of_best_gens, training_data):

            fig, ax = plt.subplots()

            num_frames = len(list_of_best_indv)

            x = training_data.x
            y_actually = training_data.y
            y = list_of_best_indv
            g = list_of_best_gens
            plt.plot(training_data.x, training_data.y, 'ro')
            points, = ax.plot(x, y[0].evaluate_equation_at(x), 'b')
            points.set_label('Generation :' + str(g[0]))
            legend = ax.legend(loc='upper right', shadow=True)

            def animate(i):
                ax.collections.clear()
                points.set_ydata(y[i].evaluate_equation_at(x))  # update the data
                points.set_label('Generation :' + str(g[i]))
                legend = ax.legend(loc='upper right')
                return points, legend

            # Init only required for blitting to give a clean slate.
            def init():
                points.set_ydata(np.ma.array(x, mask=True))
                return points, points

            plt.xlabel('x', fontsize=15)
            plt.ylabel('y', fontsize=15)
            plt.title("Best Individual in Island", fontsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.tick_params(axis='x', labelsize=12)
            plt.close()

            return animation.FuncAnimation(fig, animate, num_frames, init_func=init,
                                           interval=250, blit=True)
            from IPython.display import HTML
            anim2 = animate_data(best_indv_values, best_indv_gen, training_data)
            HTML(anim2.to_jshtml())
            from IPython.display import HTML
            anim2 = animate_data(best_indv_values, best_indv_gen, training_data)
            HTML(anim2.to_jshtml())

    def test(self):

        x = np.linspace(-10, 10, 30).reshape([-1, 1])
        y = x ** 2 + 3.5 * x ** 3
        training_data = ExplicitTrainingData(x, y)
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        plt.plot(training_data.x, training_data.y, 'ro')
        plt.show()

        component_generator = ComponentGenerator(input_x_dimension=x.shape[1])
        component_generator.add_operator("+")
        component_generator.add_operator("-")
        component_generator.add_operator("*")

        agraph_generator = AGraphGenerator(agraph_size=10,
                                           component_generator=component_generator)
        agraph = agraph_generator()
        print("f(X_0) = ", agraph)

        crossover = AGraphCrossover()
        mutation = AGraphMutation(component_generator)

        fitness = ExplicitRegression(training_data=training_data)
        optimizer = ScipyOptimizer(fitness, method='lm')
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        evaluator = Evaluation(local_opt_fitness)
        np.random.seed(16)
        agraph = agraph_generator()
        print("Before local optimization: f(X_0) = ", agraph)
        print("                          fitness = ", fitness(agraph))
        _ = local_opt_fitness(agraph)
        print("After local optimization:  f(X_0) = ", agraph)
        print("                          fitness = ", fitness(agraph))

        agraph_y = agraph.evaluate_equation_at(training_data.x)

        plt.plot(training_data.x, training_data.y, 'ro')
        plt.plot(training_data.x, agraph_y, 'b-')
        plt.show()

        POPULATION_SIZE = 32
        MUTATION_PROBABILITY = 0.4
        CROSSOVER_PROBABILITY = 0.4

        ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation,
                          CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, POPULATION_SIZE)

        def agraph_similarity(ag_1, ag_2):
            """a similarity metric between agraphs"""
            return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

        pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                                   similarity_function=agraph_similarity)

        np.random.seed(5)

        island = Island(ea, agraph_generator, POPULATION_SIZE, hall_of_fame=pareto_front)
        print("Best individual\n f(X_0) =", island.get_best_individual())

        ERROR_TOLERANCE = 1e-6

        best_indv_values = []
        best_indv_values.append(island.get_best_individual())
        best_indv_gen = []
        best_indv_gen.append(island.generational_age)

        while island.get_best_fitness() > ERROR_TOLERANCE:
            island.evolve(1)
            best_indv = island.get_best_individual()
            if best_indv.fitness < best_indv_values[-1].fitness:
                best_indv_values.append(best_indv)
                best_indv_gen.append(island.generational_age)

        print("Generation: ", island.generational_age)
        print("Success!")
        print("Best individual\n f(X_0) =", island.get_best_individual())

        print(" FITNESS   COMPLEXITY    EQUATION")
        for member in pareto_front:
            print("%.3e     " % member.fitness, member.get_complexity(),
                  "     f(X_0) =", member)

    def animate_data(list_of_best_indv, list_of_best_gens, training_data):
        fig, ax = plt.subplots()

        num_frames = len(list_of_best_indv)

        x = training_data.x
        y_actually = training_data.y
        y = list_of_best_indv
        g = list_of_best_gens
        plt.plot(training_data.x, training_data.y, 'ro')
        points, = ax.plot(x, y[0].evaluate_equation_at(x), 'b')
        points.set_label('Generation :' + str(g[0]))
        legend = ax.legend(loc='upper right', shadow=True)

        def animate(i):
            ax.collections.clear()
            points.set_ydata(y[i].evaluate_equation_at(x))  # update the data
            points.set_label('Generation :' + str(g[i]))
            legend = ax.legend(loc='upper right')
            return points, legend

        # Init only required for blitting to give a clean slate.
        def init():
            points.set_ydata(np.ma.array(x, mask=True))
            return points, points

        plt.xlabel('x', fontsize=15)
        plt.ylabel('y', fontsize=15)
        plt.title("Best Individual in Island", fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        plt.close()

        return animation.FuncAnimation(fig, animate, num_frames, init_func=init,
                                       interval=250, blit=True)
        from IPython.display import HTML
        anim2 = animate_data(best_indv_values, best_indv_gen, training_data)
        HTML(anim2.to_jshtml())
        from IPython.display import HTML
        anim2 = animate_data(best_indv_values, best_indv_gen, training_data)
        HTML(anim2.to_jshtml())


class GpInitial(Initial):
    def __init__(self, x, y, train_p=0.3, num=1, population_size=5000,
                 mutation_probability=0.4, crossover_probability=0.4, stopping_criteria=0.01, p_subtree_mutation=0.1,
                 error_tolerance=1e-6, random_seed=0, if_print_train=1, parsimony_coefficient=0.01,
                 p_point_mutation=0.1, generations=20, p_hoist_mutation=0.05):
        super().__init__(population_size, mutation_probability, crossover_probability, error_tolerance)
        self.num = num
        temp_x = np.array(x)
        temp_y = np.array(y)
        np.random.shuffle(temp_x)
        np.random.shuffle(temp_y)
        print(temp_x)
        print(temp_y)

        self.x_train = temp_x[0:int(train_p * temp_x.shape[0])].reshape(-1, 1)
        self.y_train = temp_y[0:int(train_p * temp_y.shape[0])]
        print(self.x_train.shape)
        print(self.y_train.shape)
        self.x_test = temp_x[int(train_p * temp_x.shape[0]):-1]
        self.y_test = temp_y[int(train_p * temp_y.shape[0]):-1]
        self.generation = generations
        self.random_seed = random_seed
        self.if_print_train = if_print_train
        self.parsimony_coeffcient = parsimony_coefficient
        self.p_point_mutation = p_point_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.stopping_criteria = stopping_criteria
        self.p_subtree_mutation = p_subtree_mutation
        rng = check_random_state(0)

    def do(self):
        self.est_gp = GpSymbolicRegressor(population_size=self.population_size
                                          , generations=self.generation,
                                          stopping_criteria=self.stopping_criteria,
                                          p_crossover=self.crossover_probability
                                          , p_subtree_mutation=self.p_subtree_mutation,
                                          p_hoist_mutation=self.p_hoist_mutation,
                                          p_point_mutation=self.p_point_mutation,
                                          verbose=self.if_print_train,
                                          parsimony_coefficient=self.parsimony_coeffcient,
                                          random_state=self.random_seed)
        self.est_gp.fit(self.x_train, self.y_train)
        print(self.est_gp)
        print(self.est_gp._program)
