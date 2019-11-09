import os, shutil
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import functools
import abc
from abc import abstractmethod
from .operator import forwardOperator, backwardOperator

class AverageMeter(object):
	"""Computes and stores the average and current values
	"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val
		self.count += n
		self.avg = self.sum/self.count

class Training(abc.ABC):
	"""Abstract base class for deep learning training

	Parameters
	-------------
	net : a :class: `torch.nn.Module` instance
		The net should be neural network in torch.
	optimizer : a :class: `torch.optim.optimizer` instance
		The optimizer for training the network.
	dataloader : a :class: `dict` instance
		It must have `train` key and the correponding value is
		`torch.utils.data.DataLoader` instance. An optional 
		key is `val`.
	num_epochs : a :class: `int` 
		Number of running epochs to train the network
	batch_size : a :class: `dict` instance
		It should have `train` and `val` keys, the corresponding value should
		be integer, which represents the batch size.
	loss_fun :  a :class: `nn.Loss` istance or a function
		The loss function for training network.

	measure_fun: a :class: function
		This function measures the performance.

	device : device

	scheduler : a :class: `torch.optim.lr_scheduler` instance
		The learning rate tuning scheduler.

	save_all_checkpoints: a :class: `bool` instance
		The flag for whether saving all checkpoints
	train: a :class: `bool` instance
		The flag for training or testing

	other_config: a :class: `dict`
		Other configurations for customization.
	Notes
	----------
	If a subclass overwrites the constructor, it should call the super
	constructor with *args and **kwargs.
	"""

	def __init__(self, net, dataloader, batch_size, loss_fun, measure_fun, 
		optimizer=None, num_epochs=1, device='gpu', scheduler=None,
		save_all_checkpoints=False, train=True,
		forward_op=forwardOperator(), backward_op=backwardOperator(),
		other_config=None
	):
		self._net = net
		self._optimizer = optimizer
		self._num_epochs = num_epochs
		try:
			self._train_loader = dataloader['train']
		except KeyError:
			self._train_loader = None
		try:
			self._val_loader = dataloader['val']
		except KeyError:
			self._val_loader = None
		try:
			self._train_batch_size = batch_size['train']
		except KeyError:
			self._train_batch_size = 1
		try:
			self._val_batch_size = batch_size['val']
		except KeyError:
			self._val_batch_size = 1
		self._loss_fun = loss_fun
		self._measure = measure_fun
		self._device = device
		self._scheduler = scheduler
		self._save_all_ckpt = save_all_checkpoints
		self._other_config = other_config
		self._train_flag = train
		if self._train_flag is True and self._optimizer is None:
			raise ValueError("Optimizer should be set for training.")
		self._logger = None
		self._log_freq_train = 100
		self._log_freq_val = 50
		self._forward_op = forward_op
		self._backward_op = backward_op
		# to customize the initialization in subclasses, please
		# try to overwrite _initialize instead of __init__ if
		# possible
		self._initialize()

	def _initialize(self):
		"""Additional initializer that can be overwritten by
		subclasses without redefining the full __init__ method
		including all arguments and documentation."""
		pass		

	def name(self):
		"""Returns a human readable name that uniquely identifies
		the training with its hyperparameters.

		Returns
		-------
		str
			Human readable name that uniquely identifies the training
			with its hyperparameters.

		Notes
		-----
		Defaults to the class name but subclasses can provide more
		descriptive names and must take hyperparameters into account.

		"""
		return self.__class__.__name__		

class BaseTraining(Training):
	"""Common base class for training networks.
	"""
	@abc.abstractmethod
	def _train(self, epoch):
		raise NotImplementedError

	@abc.abstractmethod
	def _val(self, epoch):
		raise NotImplementedError

	def _init_logger(self, exp_dir):
		# first create logging file
		f = open(os.path.join(exp_dir, 'log.txt'), 'w')
		f.close()

		f = open(os.path.join(exp_dir, 'log.txt'), 'a')
		self._logger = f


	def run(self, exp_dir):
		if self._train_flag:
			os.makedirs(exp_dir, exist_ok=True)
			self._init_logger(exp_dir)
			save_model_path = os.path.join(exp_dir, 'ckpt.pth.tar')
			best_model_path = os.path.join(exp_dir, 'best.pth.tar')
			best_measure = 0.
			best_epoch = 0.
			start = time.time()
			for epoch in range(1, self._num_epochs+1):
				if self._save_all_ckpt:
					save_model_path = os.path.join(exp_dir, "epoch_%d.pth.tar"%epoch)

				self._train(epoch)
				avg_loss, avg_measre = self._val(epoch)

				# Update the learning rate
				if self._scheduler is not None and not isinstance(self._scheduler, torch.optim.lr_scheduler.CyclicLR) \
					and not isinstance(self._scheduler, torch.optim.lr_scheduler.OneCycleLR):
					self._scheduler.step()

				torch.save({'epoch': epoch,
					'state_dict': self._net.state_dict()}, save_model_path)
				if avg_measre is not None and avg_measre.avg >= best_measure:
					torch.save({'epoch': epoch,
						'state_dict': self._net.state_dict(),
						'measure': avg_measre.avg,
						'loss': avg_loss.avg}, best_model_path)
					best_measure = avg_measre.avg
					best_epoch = epoch
				print('Validating Best Measure: %.4f, epoch %d' % (best_measure, best_epoch))
			end = time.time()
			print('Work done! Total elapsed time: %.2f s' % (end - start))
			self._logger.close()
		else:
			start = time.time()
			avg_loss, avg_measre = self._val(epoch=1)
			end = time.time()
			print('Validating loss: %.4f, measure: %.4f' % (avg_loss.avg, avg_measre.avg))
			print('Elapsed time: %.2f s' % (end - start))
