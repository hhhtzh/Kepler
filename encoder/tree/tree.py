from copy import copy

from sklearn.utils._random import sample_without_replacement

from encoder.encoder import Encoder
from algorithm.evolutionary_computing.symbolic_regression.functions import _Function
import numpy as np

from encoder.graph.evaluation_backend import evaluation_backend
from util.probability_mass_function import LOGGER
from util.randoms import check_random_state


def get_subtree(random_state, program=None):
    """Get a random subtree from the program.

    Parameters
    ----------
    random_state : RandomState instance
        The random number generator.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, the
        embedded tree in the object will be used.

    Returns
    -------
    start, end : tuple of two ints
        The indices of the start and end of the random subtree.

    """
    if program is None:
        raise ValueError('获取子树的树参数为空')
    # Choice of crossover points follows Koza's (1992) widely used approach
    # of choosing functions 90% of the time and leaves 10% of the time.
    probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                      for node in program])
    probs = np.cumsum(probs / probs.sum())
    start = np.searchsorted(probs, random_state.uniform())

    stack = 1
    end = start
    while stack > end - start:
        node = program[end]
        if isinstance(node, _Function):
            stack += node.arity
        end += 1

    return start, end


class GpTree(Encoder):
    """A program-like representation of the evolved program.

       This is the underlying data-structure used by the public classes in the
       :mod:`gplearn.genetic` module. It should not be used directly by the user.

       Parameters
       ----------
       function_set : list
           A list of valid functions to use in the program.

       arities : dict
           A dictionary of the form `{arity: [functions]}`. The arity is the
           number of arguments that the function takes, the functions must match
           those in the `function_set` parameter.

       init_depth : tuple of two ints
           The range of tree depths for the initial population of naive formulas.
           Individual trees will randomly choose a maximum depth from this range.
           When combined with `init_method='half and half'` this yields the well-
           known 'ramped half and half' initialization method.

       init_method : str
           - 'grow' : Nodes are chosen at random from both functions and
             terminals, allowing for smaller trees than `init_depth` allows. Tends
             to grow asymmetrical trees.
           - 'full' : Functions are chosen until the `init_depth` is reached, and
             then terminals are selected. Tends to grow 'bushy' trees.
           - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
             'grow', making for a mix of tree shapes in the initial population.

       n_features : int
           The number of features in `X`.

       const_range : tuple of two floats
           The range of constants to include in the formulas.

       metric : _Fitness object
           The raw fitness metric.

       p_point_replace : float
           The probability that any given node will be mutated during point
           mutation.

       parsimony_coefficient : float
           This constant penalizes large programs by adjusting their fitness to
           be less favorable for selection. Larger values penalize the program
           more which can control the phenomenon known as 'bloat'. Bloat is when
           evolution is increasing the size of programs without a significant
           increase in fitness, which is costly for computation time and makes for
           a less understandable final result. This parameter may need to be tuned
           over successive runs.

       random_state : RandomState instance
           The random number generator. Note that ints, or None are not allowed.
           The reason for this being passed is that during parallel evolution the
           same program object may be accessed by multiple parallel processes.

       transformer : _Function object, optional (default=None)
           The function to transform the output of the program to probabilities,
           only used for the SymbolicClassifier.

       feature_names : list, optional (default=None)
           Optional list of feature names, used purely for representations in
           the `print` operation or `export_graphviz`. If None, then X0, X1, etc
           will be used for representations.

       program : list, optional (default=None)
           The flattened tree representation of the program. If None, a new naive
           random tree will be grown. If provided, it will be validated.

       Attributes
       ----------
       program : list
           The flattened tree representation of the program.

       raw_fitness_ : float
           The raw fitness of the individual program.

       fitness_ : float
           The penalized fitness of the individual program.

       oob_fitness_ : float
           The out-of-bag raw fitness of the individual program for the held-out
           samples. Only present when sub-sampling was used in the estimator by
           specifying `max_samples` < 1.0.

       parents : dict, or None
           If None, this is a naive random program from the initial population.
           Otherwise it includes meta-data about the program's parent(s) as well
           as the genetic operations performed to yield the current program. This
           is set outside this class by the controlling evolution loops.

       depth_ : int
           The maximum depth of the program tree.

       length_ : int
           The number of functions and terminals in the program.

       """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state=None,
                 transformer=None,
                 feature_names=None,
                 program=None,
                 ):
        super().__init__()
        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.random_state = random_state

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state):
        """Build a naive random program.
        创建一个新的随机树
        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]
        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def distance(self, other):
        """Computes the distance to another `Tree`

        Distance is a measure of similarity of the two command_arrays

        param
        other:'GpTree'

            The individual to which distance will be calculated
        :return:
        int :
            distance from self to individual
        """
        dist = np.sum(np.array(self.program) != np.array(other.program))
        return dist

    def evaluate_equation_at(self, x):
        pass

    def evaluate_equation_with_local_opt_gradient_at(self, x):
        _simplified_command_array=np.array()
        _simplified_constants=np.array()
        for st in self.program:
            if (st in self.init_method):
                np.append(_simplified_command_array,int(st))
            else:
                np.append(_simplified_constants,int(st))
        try:
            f_of_x, df_dc = evaluation_backend.evaluate_with_derivative(
                _simplified_command_array, x,
                _simplified_constants, False)
            return f_of_x, df_dc
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/const-deriv", err)
            nan_array = np.full((x.shape[0], len(_simplified_constants)),
                                np.nan)
            return nan_array, np.array(nan_array)

    def get_complexity(self):
        _simplified_command_array = np.array()
        for st in self.program:
            if (st in self.init_method):
                np.append(_simplified_command_array, int(st))
        return _simplified_command_array.shape[0]

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
