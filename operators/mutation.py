"""The genetic operation of Mutation.

This module defines the basis of mutation in bingo evolutionary analyses.
"""

from abc import ABCMeta, abstractmethod
from copy import copy

from algorithm.evolutionary_computing.symbolic_regression.functions import _Function
from encoder.tree.tree import get_subtree
from operators.crossover import GpCrossover

"""Mutation of acyclic graph individuals.

This module contains the implementation of mutation for acyclic graph
individuals, which is composed of 4 possible mutation strategies: command
mutation, node mutation, parameter mutation and pruning.
"""
from random import randint, randrange

import numpy as np

from algorithm.evolutionary_computing.symbolic_regression.operator_definitions import IS_ARITY_2_MAP, IS_TERMINAL_MAP, \
    CONSTANT, VARIABLE, INTEGER
from util.argument_validation import argument_validation
from util.probability_mass_function import ProbabilityMassFunction


class Mutation(metaclass=ABCMeta):
    """A mutator of individuals.

    An abstract base class for the mutation of chromosomes
    """

    @abstractmethod
    def __call__(self, parent):
        """Mutates individuals

        Parameters
        ----------
        parent : Chromosome
                 The individual to be mutated.

        Returns
        -------
        Chromosome :
            A mutated generated individual
        """
        raise NotImplementedError


COMMAND_MUTATION = "command"
NODE_MUTATION = "node"
PARAMETER_MUTATION = "parameter"
PRUNE_MUTATION = "prune"
FORK_MUTATION = "fork"


class AGraphMutation(Mutation):
    """Mutation of acyclic graph individual

    Mutation of an agraph individual my modification of its command array.
    Mutation randomly takes one of the following 5 forms
      * command mutation: An entire command (row) of the command array is
        replaced by a new random one
      * node mutation: The node of a command is replaced by a random new one.
        Change from a terminal to operators (or reversed) will automatically
        have updated parameters to maintain consistency.
      * parameter mutation: The parameters of a command are randomly changed.
      * pruning: The command array is adjusted to remove an operators from
        the evaluation of the `AGraph`. Pruning a terminal does nothing.
      * forking: A random command is selected to become the child of a
        new command (e.g., X_0 -> sin(X_0)). Forking on an equation with no
        non-utilized commands does nothing.

    Parameters
    ----------
    command_probability : float
        probability of command mutation. Default 0.2
    node_probability : float
        probability of node mutation. Default 0.2
    parameter_probability : float
        probability of parameter mutation. Default 0.2
    prune_probability : float
        probability of pruning. Default 0.2
    fork_probability : float
        probability of forking. Default 0.2

    Attributes
    ----------
    types : iterable of str
        an iterable of the possible mutation types
    last_mutation_type : str
        the last mutation type that happened (or None)

    Notes
    -----
    The input probabilities are normalized if their sum is not equal to 1.

    Mutation can result in no change if, for instance,
      * a prune mutation is executed on a `AGraph` utilizing only a single
        terminal.
      * a parameter mutation occurs on a `AGraph` utilizing only a single
        constant.
    """

    @argument_validation(command_probability={">=": 0, "<=": 1},
                         node_probability={">=": 0, "<=": 1},
                         parameter_probability={">=": 0, "<=": 1},
                         prune_probability={">=": 0, "<=": 1},
                         fork_probability={">=": 0, "<=": 1})
    def __init__(self, component_generator, command_probability=0.2,
                 node_probability=0.2, parameter_probability=0.2,
                 prune_probability=0.2, fork_probability=0.2):
        self._component_generator = component_generator
        self._mutation_function_pmf = \
            ProbabilityMassFunction([self._mutate_command,
                                     self._mutate_node,
                                     self._mutate_parameters,
                                     self._prune_branch,
                                     self._fork_mutation],
                                    [command_probability,
                                     node_probability,
                                     parameter_probability,
                                     prune_probability,
                                     fork_probability])
        self._last_mutation_location = None
        self.last_mutation_type = None
        self.types = [COMMAND_MUTATION, NODE_MUTATION, PARAMETER_MUTATION,
                      PRUNE_MUTATION, FORK_MUTATION]

    def __call__(self, parent):
        """Single point mutation.

        Parameters
        ----------
        parent : `AGraph`
            The parent individual

        Returns
        -------
        `AGraph` :
            The child of the mutation
        """
        child = parent.copy()
        mutation_algorithm = self._mutation_function_pmf.draw_sample()
        mutation_algorithm(child)
        return child

    def _mutate_command(self, individual):
        mutation_location = \
            self._get_random_command_mutation_location(individual)
        self._last_mutation_location = mutation_location
        self.last_mutation_type = COMMAND_MUTATION

        old_command = individual.command_array[mutation_location]
        new_command = \
            self._component_generator.random_command(mutation_location)
        while np.array_equal(new_command, old_command) \
                or old_command[0] == new_command[0] == CONSTANT:
            new_command = \
                self._component_generator.random_command(mutation_location)

        individual.mutable_command_array[mutation_location] = new_command

    @staticmethod
    def _get_random_command_mutation_location(child):
        utilized_commands = child.get_utilized_commands()
        indices = [n for n, x in enumerate(utilized_commands) if x]
        index = np.random.randint(len(indices))
        return indices[index]

    def _mutate_node(self, individual):
        mutation_location = self._get_random_node_mutation_location(individual)
        self._last_mutation_location = mutation_location
        self.last_mutation_type = NODE_MUTATION

        old_command = individual.command_array[mutation_location]
        new_command = old_command.copy()
        while old_command[0] == new_command[0]:
            self._randomize_node(new_command)

        individual.mutable_command_array[mutation_location] = new_command

    def _get_random_node_mutation_location(self, child):
        utilized_commands = child.get_utilized_commands()
        terminals_ok = self._component_generator.get_number_of_terminals() > 1
        operators_ok = self._component_generator.get_number_of_operators() > 1
        indices = []
        for i, (x, node) in enumerate(zip(utilized_commands,
                                          child.command_array[:, 0])):
            if x:
                if (IS_TERMINAL_MAP[node] and terminals_ok) or \
                        (not IS_TERMINAL_MAP[node] and operators_ok):
                    indices.append(i)
        index = np.random.randint(len(indices))
        return indices[index]

    def _randomize_node(self, command):
        if IS_TERMINAL_MAP[command[0]]:
            command[0] = self._component_generator.random_terminal()
            command[1] = \
                self._component_generator.random_terminal_parameter(command[0])
            command[2] = command[1]
        else:
            command[0] = self._component_generator.random_operator()

    def _mutate_parameters(self, individual):
        mutation_location = self._get_random_param_mut_location(individual)
        self._last_mutation_location = mutation_location
        self.last_mutation_type = PARAMETER_MUTATION
        if mutation_location is None:
            return

        old_command = individual.command_array[mutation_location]
        new_command = old_command.copy()
        while np.array_equal(old_command, new_command):
            self._randomize_parameters(new_command, mutation_location)

        individual.mutable_command_array[mutation_location] = new_command

    def _get_random_param_mut_location(self, individual):
        utilized_commands = individual.get_utilized_commands()
        no_param_mut = [CONSTANT, INTEGER]
        if self._component_generator.input_x_dimension <= 1:
            no_param_mut += [VARIABLE]

        indices = [i for i, x in enumerate(utilized_commands)
                   if x and
                   individual.command_array[i, 0] not in no_param_mut]

        if 1 in indices and \
                not IS_TERMINAL_MAP[individual.command_array[1, 0]]:
            indices.remove(1)

        if not indices:
            return None
        index = np.random.randint(len(indices))
        return indices[index]

    def _randomize_parameters(self, command, mutation_location):
        if IS_TERMINAL_MAP[command[0]]:
            command[1] = \
                self._component_generator.random_terminal_parameter(command[0])
            command[2] = command[1]
        else:
            command[1] = \
                self._component_generator.random_operator_parameter(
                    mutation_location)
            if IS_ARITY_2_MAP[command[0]]:
                command[2] = \
                    self._component_generator.random_operator_parameter(
                        mutation_location)

    def _prune_branch(self, individual):
        mutation_location = self._get_random_prune_location(individual)
        self._last_mutation_location = mutation_location
        self.last_mutation_type = PRUNE_MUTATION
        if mutation_location is None:
            return

        mutated_node = individual.command_array[mutation_location, 0]
        if IS_ARITY_2_MAP[mutated_node]:
            pruned_param_num = np.random.randint(2)
        else:
            pruned_param_num = 0
        pruned_param = individual.command_array[mutation_location,
        1 + pruned_param_num]

        for i, (node, p_1, p_2) in \
                enumerate(individual.command_array[mutation_location:]):
            if not IS_TERMINAL_MAP[node]:
                if p_1 == mutation_location:
                    individual.mutable_command_array[
                        mutation_location + i, 1] = pruned_param
                if p_2 == mutation_location:
                    individual.mutable_command_array[
                        mutation_location + i, 2] = pruned_param

    @staticmethod
    def _get_random_prune_location(individual):
        utilized_commands = individual.get_utilized_commands()
        indices = [i for i, x in enumerate(utilized_commands[:-1])
                   if x and
                   not IS_TERMINAL_MAP[individual.command_array[i, 0]]]
        if not indices:
            return None
        index = np.random.randint(len(indices))
        return indices[index]

    def _fork_mutation(self, individual):
        """
        Create a fork before a command in the individual's command stack
        if there are enough unused commands in the command stack

        Parameters
        ----------
        individual : `AGraph`
            individual to mutate on
        """
        # NOTE (David Randall): Increasing past 4 should work but has not been
        # tested
        MAX_FORK_SIZE = 4

        utilized_commands = individual.get_utilized_commands()
        n_unutilized_commands = utilized_commands.count(False)

        if n_unutilized_commands < 2:
            self._last_mutation_location = None
            self.last_mutation_type = FORK_MUTATION
            return

        max_fork = min(n_unutilized_commands, MAX_FORK_SIZE)
        fork_size = np.random.randint(2, max_fork + 1)

        inds = [n for n, x in enumerate(utilized_commands) if x]
        mutation_location = np.random.choice(inds)
        stack = individual.mutable_command_array

        new_stack, new_utilized_commands, index_shifts, \
            mutated_command_location, unutilized_range = \
            self._move_utilized_commands(stack, utilized_commands,
                                         mutation_location)
        self._fix_indices(new_stack, new_utilized_commands, index_shifts)

        new_stack = self._insert_fork(new_stack, fork_size,
                                      mutated_command_location,
                                      *unutilized_range)
        individual.mutable_command_array[:] = new_stack

        self._last_mutation_location = mutation_location
        self.last_mutation_type = FORK_MUTATION

    def _insert_fork(self, stack, fork_size, mutated_command_location, start_i,
                     end_i):
        """
        Inserts commands/a fork of size fork_size in the stack's
        unutilized command section

        start_i and end_i represent the starting and ending indices
        of the stack's unutilized command section
        """
        try:  # normal case
            arity_2_op = self._get_arity_operator(2)
            n_terminals = randint(1, fork_size // 2)

            for i in range(start_i, start_i + fork_size):
                if i < start_i + n_terminals:
                    stack[i] = \
                        self._component_generator.random_terminal_command()
                    # insert terminals
                elif i == start_i + fork_size - 1:
                    # make sure end of the stack is always a utilized command
                    stack[end_i] = \
                        np.array([arity_2_op, mutated_command_location,
                                  randrange(start_i, i)], dtype=int)
                    # insert an arity 2 operators that
                    # links to the mutated_command
                else:
                    stack[i] = \
                        np.array([self._component_generator.random_operator(),
                                  randrange(start_i, i), randrange(start_i, i)],
                                 dtype=int)
                    # insert a random operators that connects
                    # to previously generated commands in the fork
        except RuntimeError:  # case where we only have ar1 ops
            for i in range(start_i, start_i + fork_size):
                if i == start_i:
                    stack[i] = \
                        np.array([self._component_generator.random_operator(),
                                  mutated_command_location,
                                  mutated_command_location],
                                 dtype=int)
                    # insert a random operators that links to the mutated command
                elif i == start_i + fork_size - 1:
                    # make sure end of the stack is always a utilized command
                    stack[end_i] = \
                        np.array([self._component_generator.random_operator(),
                                  i - 1,
                                  i - 1], dtype=int)
                else:
                    stack[i] = \
                        np.array([self._component_generator.random_operator(),
                                  i - 1,
                                  i - 1], dtype=int)
                    # insert a random operators that connects
                    # to the previous command in the fork
        return stack

    @staticmethod
    def _move_utilized_commands(stack, utilized_commands, mutation_location):
        """
        Prepares a stack for fork mutation by moving
        utilized commands that occur before/at the mutation_location
        to the front of the stack, unutilized commands to the middle
        of the stack, and utilized commands that occur after the
        mutation_location to the end of the stack

        Returns the prepared stack, an updated list of utilized commands,
        a dictionary that shows how command indices changed/shifted during
        the method, the new location of the mutated command, and
        the beginning and ending indices of where the unutilized commands
        occur in the stack
        """
        indices = range(len(stack))
        before_mutation_location = []
        unutilized = []
        after_mutation_location = []

        for i, stack_util_index_tuple in \
                enumerate(zip(stack, utilized_commands, indices)):
            if stack_util_index_tuple[1]:  # if utilized
                if i <= mutation_location:
                    before_mutation_location.append(stack_util_index_tuple)
                else:
                    after_mutation_location.append(stack_util_index_tuple)
            else:
                unutilized.append(stack_util_index_tuple)

        final_tuples = before_mutation_location + unutilized + \
                       after_mutation_location
        new_stack, new_utilized_commands, new_indices = zip(*final_tuples)

        index_shifts = dict(zip(new_indices, indices))  # dictionary that shows
        # how to translate old indices to new indices
        mutated_command_location = index_shifts[mutation_location]
        index_shifts[mutation_location] = \
            len(before_mutation_location) + len(unutilized) - 1
        # change the mutation location
        # to be at the end of the unutilized commands

        return np.array(new_stack), list(new_utilized_commands), \
            index_shifts, mutated_command_location, \
            [len(before_mutation_location),
             len(before_mutation_location) + len(unutilized) - 1]

    def _fix_indices(self, stack, utilized_commands, index_shifts):
        """
        Convert any non-terminal command parameters
        using index_shifts and fix any non-terminal command
        parameters y generating new ones
        """
        non_terminals = ~np.vectorize(IS_TERMINAL_MAP.get)(stack[:, 0])
        utilized_operators = np.logical_and(non_terminals, utilized_commands)
        index_shifts = \
            dict(sorted(index_shifts.items(), key=lambda pair: pair[0]))
        index_values = np.array(list(index_shifts.values()))
        # convert index_shifts to an array corresponding to how each index
        # has changed e.g. [0, 2, 1] means that index 0 stayed the same
        # and indices 1 and 2 swapped places ({0: 0, 1: 2, 2: 1} -> [0, 2, 1])

        # convert parameters in utilized operators according to index shifts
        for i in range(1, 3):
            stack[utilized_operators, i] = \
                index_values[stack[utilized_operators, i]]

        # fix any non-terminal command parameters that are invalid
        # by generating new ones
        indices = np.array(range(len(stack)))[non_terminals]
        for i in range(1, 3):
            indices_to_fix = \
                indices[np.where(stack[non_terminals][:, i] >= indices)[0]]
            if len(indices_to_fix) > 0:
                stack[:, i][indices_to_fix] = np.vectorize(
                    self._component_generator.random_operator_parameter)(
                    indices_to_fix)

    def _get_arity_operator(self, arity):
        """
        Tries to get an operators of a particular arity,
        else raises a RuntimeError
        """
        attempts = 0
        operator = None
        if arity == 1:
            while operator is None or IS_ARITY_2_MAP[operator]:
                if attempts >= 100:
                    raise RuntimeError("Could not generate arity "
                                       f"{arity} operators")
                operator = self._component_generator.random_operator()
                attempts += 1
        else:
            while operator is None or not IS_ARITY_2_MAP[operator]:
                if attempts >= 100:
                    raise RuntimeError("Could not generate arity "
                                       f"{arity} operators")
                operator = self._component_generator.random_operator()
                attempts += 1
        return operator


class GpSubtreeMutation(Mutation):
    """Perform the subtree mutation operation on the program.

            Subtree mutation selects a random subtree from the embedded program to
            be replaced. A donor subtree is generated at random and this is
            inserted into the original parent to form an offspring. This
            implementation uses the "headless chicken" method where the donor
            subtree is grown using the initialization methods and a subtree of it
            is selected to be donated to the parent.

            Parameters
            ----------
            random_state : RandomState instance
                The random number generator.

            Returns
            -------
            program : list
                The flattened tree representation of the program.

            """

    def __init__(self, random_state):
        self.type = ['GpMutation']
        self.random_state = random_state

    def __call__(self, parent):
        # Build a new naive program
        chicken = parent.build_program(self.random_state)
        # Do subtree mutation via the headless chicken method!
        cr = GpCrossover(self.random_state)
        return cr(parent, chicken)


class HoistMutation(Mutation):
    """Perform the hoist mutation operation on the program.

          Hoist mutation selects a random subtree from the embedded program to
          be replaced. A random subtree of that subtree is then selected and this
          is 'hoisted' into the original subtrees location to form an offspring.
          This method helps to control bloat.

          Parameters
          ----------
          random_state : RandomState instance
              The random number generator.

          Returns
          -------
          program : list
              The flattened tree representation of the program.

          """

    def __init__(self, random_state):
        self.type = ['GpMutation']
        self.random_state = random_state

    def __call__(self, tree):
        # Get a subtree to replace
        start, end = tree.get_subtree(self.random_state)
        subtree = tree.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = tree.get_subtree(self.random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return tree.program[:start] + hoist + tree.program[end:], removed


class PointMutation(Mutation):
    """Perform the point mutation operation on the program.

           Point mutation selects random nodes from the embedded program to be
           replaced. Terminals are replaced by other terminals and functions are
           replaced by other functions that require the same number of arguments
           as the original node. The resulting tree forms an offspring.

           Parameters
           ----------
           random_state : RandomState instance
               The random number generator.

           Returns
           -------
           program : list
               The flattened tree representation of the program.

           """

    def __init__(self, random_state):
        self.type = ['GpMutation']
        self.random_state = random_state

    def __call__(self, tree):
        program = copy(tree.program)

        # Get the nodes to modify
        mutate = np.where(self.random_state.uniform(size=len(program)) <
                          tree.p_point_replace)[0]

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(tree.arities[arity])
                replacement = self.random_state.randint(replacement)
                replacement = tree.arities[arity][replacement]
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if tree.const_range is not None:
                    terminal = self.random_state.randint(tree.n_features + 1)
                else:
                    terminal = self.random_state.randint(tree.n_features)
                if terminal == tree.n_features:
                    terminal = self.random_state.uniform(*tree.const_range)
                    if tree.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        return program, list(mutate)
