import torch

class fgsm(object):
	def __init__(self, eps, step_size):
		self.step_size = step_size
		self._eps = eps

	def __call__(self, grad):
		return self.step_size * grad.data.sign()