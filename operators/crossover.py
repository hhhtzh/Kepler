"""The genetic operation of crossover.

This module defines the basis of crossover in bingo evolutionary analyses.
"""

from abc import ABCMeta, abstractmethod
from encoder.tree.tree import get_subtree

"""Definition of crossover between two acyclic graph individuals

This module contains the implementation of single point crossover between
acyclic graph individuals.
"""
import numpy as np



class Crossover(metaclass=ABCMeta):
    """Crossover for chromosomes.

    An abstract base class for the crossover between two genetic individuals
    in bingo.
    """
    @abstractmethod
    def __call__(self, parent_1, parent_2):
        """Crossover between two individuals

        Parameters
        ----------
        parent_1 : Chromosome
                   The first parent individual
        parent_2 : Chromosome
                   The second parent individual

        Returns
        -------
        tuple(Chromosome, Chromosome) :
            The two children from the crossover.
        """
        raise NotImplementedError


class AGraphCrossover(Crossover):
    """Crossover between acyclic graph individuals

    Attributes
    ----------
    types : iterable of str
        an iterable of the possible crossover types
    last_crossover_types : tuple(str, str)
        the crossover type (or None) that happened to create the first child
        and second child, respectively
    """

    def __init__(self):
        self.types = ["bingoGraph"]

    def __call__(self, parent_1, parent_2):
        """Single point crossover.

        Parameters
        ----------
        parent_1 : `AGraph`
            The first parent individual
        parent_2 : `AGraph`
            The second parent individual

        Returns
        -------
        tuple(`AGraph`, `AGraph`) :
            The two children from the crossover.
        """
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()

        ag_size = parent_1.command_array.shape[0]
        cross_point = np.random.randint(1, ag_size - 1)
        child_1.mutable_command_array[cross_point:] = \
            parent_2.command_array[cross_point:]
        child_2.mutable_command_array[cross_point:] = \
            parent_1.command_array[cross_point:]

        child_age = max(parent_1.genetic_age, parent_2.genetic_age)
        child_1.genetic_age = child_age
        child_2.genetic_age = child_age

        self.last_crossover_types = ("default", "default")

        return child_1, child_2

class GpCrossover(Crossover):
    """Perform the crossover genetic operation on the program.

            Crossover selects a random subtree from the embedded program to be
            replaced. A donor also has a subtree selected at random and this is
            inserted into the original parent to form an offspring.

            Parameters
            ----------
            donor : list
                The flattened tree representation of the donor program.

            random_state : RandomState instance
                The random number generator.

            Returns
            -------
            program : list
                The flattened tree representation of the program.

            """
    def __init__(self,random_state):
        self.type=['GpTree']
        self.random_state=random_state
    def __call__(self,parent,donor):
        # Get a subtree to replace
        start, end = parent.get_subtree(self.random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = parent.get_subtree(self.random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (parent.program[:start] +
                donor[donor_start:donor_end] +
                parent.program[end:]), removed, donor_removed






