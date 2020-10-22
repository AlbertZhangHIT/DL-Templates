
import torch
import torch.nn
import torch.nn.functional as F

def replicate_input(x):
	return x.detach().clone()

def clamp(x, lower_limit, upper_limit):
	return torch.max(torch.min(x, upper_limit), lower_limit)

def _batch_multiply_tensor_by_vector(vector, batch_tensor):
	"""Equivalent to the following
	for ii in range(len(vector)):
		batch_tensor.data[ii] *= vector[ii]
	return batch_tensor
	"""
	return (batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

def batch_multiply(float_or_vector, tensor):
	if isinstance(float_or_vector, torch.Tensor):
		assert len(float_or_vector) == len(tensor)
		tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
	elif isinstance(float_or_vector, float):
		tensor *= float_or_vector
	else:
		raise TypeError("Value has to be float or torch.Tensor")
	return tensor

def _get_norm_batch(x, p):
	batch_size = x.size(0)
	return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)

def normalize_by_pnorm(x, p=2, small_constant=1e-6):
	"""
	Normalize gradients for gradient (not gradient sign) attacks.
	# TODO: move this function to utils

	:param x: tensor containing the gradients on the input.
	:param p: (optional) order of the norm for the normalization (1 or 2).
	:param small_constant: (optional float) to avoid dividing by zero.
	:return: normalized gradients.
	"""
	# loss is averaged over the batch so need to multiply the batch
	# size to find the actual gradient of each input sample

	assert isinstance(p, float) or isinstance(p, int)
	norm = _get_norm_batch(x, p)
	norm = torch.max(norm, torch.ones_like(norm) * small_constant)
	return batch_multiply(1. / norm, x)

