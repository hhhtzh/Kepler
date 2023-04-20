import copy
from abc import ABCMeta, abstractmethod

"""
Encoder类代表了一个个体，也就是一个方程

Attributes
    ----------
    fitness : numeric
    genetic_age : int
        age of the oldest component of the genetic material in the individual
        在这个个体中最年张的基因成分
    fit_set : bool
        whether the fitness has been calculated for the individual
        这个个体的适应度是否已经被计算
"""

class Encoder(metaclass=ABCMeta):
    def __init__(self,genetic_age=0,fitness=None,fit_set=False):
        self._genetic_age = genetic_age
        self._fitness = fitness
        self._fit_set = fit_set

    @property
    def fitness(self):
        """numeric or tuple of numeric: The fitness of the individual"""
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness
        self._fit_set = True
    @property
    def genetic_age(self):
        """The age of the oldest components of the individual"""
        return self._genetic_age

    @genetic_age.setter
    def genetic_age(self, genetic_age):
        self._genetic_age = genetic_age

    @property
    def fit_set(self):
        """Indication of whether the fitness has been set"""
        return self._fit_set

    @fit_set.setter
    def fit_set(self, fit_set):
        self._fit_set = fit_set

    def copy(self):
        """copy

        Returns
        -------
            A deep copy of self
        """
        return copy.deepcopy(self)

    @abstractmethod
    def __str__(self):
        """String conversion of individual
        #以字符串的形式输出一个个体
        Returns
        -------
        str
            Individual string form
        """
        raise NotImplementedError

    @abstractmethod
    def distance(self, other):
        """Distance from self to other

        Parameters
        ----------
        other : Encoder
            The other to compare to.

        Returns
        -------
        float
            Distance from self to other
        """
        raise NotImplementedError

    def needs_local_optimization(self):
        """Does the `Encoder` need local optimization

        Returns
        -------
        bool
            Whether `Encoder` needs optimization
        """
        raise NotImplementedError("This Encoder cannot be used in local "
                                  "optimization until its local optimization "
                                  "interface has been implemented")

    def get_number_local_optimization_params(self):
        """Get number of parameters in local optimization

        Returns
        -------
        int
            Number of parameters to be optimized
        """
        raise NotImplementedError("This Encoder cannot be used in local "
                                  "optimization until its local optimization "
                                  "interface has been implemented")

    def set_local_optimization_params(self, params):
        """Set local optimization parameters

        Parameters
        ----------
        params : list-like of numeric
            Values to set the parameters to
        """
        raise NotImplementedError("This Encoder cannot be used in local "
                                  "optimization until its local optimization "
                                  "interface has been implemented")

    @abstractmethod
    def evaluate_equation_at(self, x):
        """Evaluate the equation.

        Get value of the equation at points x.

        Parameters
        ----------
        x : MxD array of numeric.
        x是M行D列的矩阵
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        Mx1 array of numeric
        返回一个M行1列的矩阵
            :math:`f(x)`
        """
        raise NotImplementedError

    def evaluate_equation_with_x_gradient_at(self, x):
        """Evaluate equation and get its derivatives.
        //对方程的每个维度上求梯度
        Get value the equation at x and its gradient with respect to x.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxD array of numeric)
            :math:`f(x)` and :math:`df(x)/dx_i`
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_equation_with_local_opt_gradient_at(self, x):
        """Evaluate equation and get its derivatives.

        Get value the equation at x and its gradient with respect to
        optimization parameters.
        获得x的局部最优化参数

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxL array of numeric)
            :math:`f(x)` and :math:`df(x)/dc_i`. L is the number of
            optimization parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def get_complexity(self):
        """Calculate complexity of equation.
        计算一个方程的复杂度
        Returns
        -------
        numeric
            complexity measure of equation
        """
        raise NotImplementedError