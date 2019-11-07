import abc
from abc import abstractmethod

class Operator(abc.ABC):
	"""abstract base class for operators in inverse problem.
	The operators should be able to deal with a batch of data.
	Parameters:
		config:  configuration fot the operator
	"""

	def __init__(self, config=None):
		self._config = config
		self._initialize()

	def _initialize(self):
		"""Additional initializer that can be overwritten by
		subclasses without redefining the full __init__ method
		including all arguments and documentation."""
		pass

	@abstractmethod
	def __call__(self, x, **kwargs):
		raise NotImplementedError

	def name(self):
		"""Returns a human readable name that uniquely identifies
		the operator with its hyperparameters.

		Returns
		-------
		str
		    Human readable name that uniquely identifies the attack
		    with its hyperparameters.

		Notes
		-----
		Defaults to the class name but subclasses can provide more
		descriptive names and must take hyperparameters into account.

		"""
		return self.__class__.__name__

class forwardOperator(Operator):
	"""Forward operator in inverse problems
	"""
	def __call__(self, x):
		return x

class backwardOperator(Operator):
	"""Adjoint/backward operator in inverse problems
	""" 
	def __call__(self, x):
		return x