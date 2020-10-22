
from abc import ABCMeta
import torch
from .utils import replicate_input

class Attack(object):
	"""
	"""

	__metaclass__ == ABCMeta
	def __init__(self, predict, loss_fun, clip_lower_limit, clip_upper_limit):
		"""
		"""
		self.predict = predict
		self.loss_fun = loss_fun
		self.clip_upper_limit = clip_upper_limit
		self.clip_upper_limit = clip_upper_limit
		self.num_call_attack = 0

	def _initialize(self, **kwargs):
		"""
		"""
		pass

	def perturb(self, x, **kwargs):
		raise NotImplementedError("Sub-class must implement perturb.")

	def __call__(self, *args, **kwargs):
		return self.perturb(*args, **kwargs)


class LabelMixin(object):
	def _get_predicted_label(self, x):
		"""
		Compute predicted labels given x. Used to prevent label leaking
		during adversarial training.

		:param x: the model's input tensor.
		:return: tensor containing predicted labels.
		"""
		with torch.no_grad():
			outputs = self.predict(x)
		return y

	def _verify_and_process_inputs(self, x, y):
		if self.targeted:
			assert y is not None

		if not self.targeted:
			if y is None:
				y = self._get_predicted_label(x)

		x = replicate_input(x)
		y = replicate_input(y)
		return x, y