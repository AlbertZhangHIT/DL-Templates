import os, shutil
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import functools
import abc
from abc import abstractmethod
from .operator import forwardOperator, backwardOperator
from .logger import setup_logger

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

	save_checkpoint_freq: a :class: `int` instance
		The flag for saving checkpoints frequency, default 0
	train: a :class: `bool` instance
		The flag for training or testing
	validate: a :class: `bool` instance
		The flag for validating during training
	forward_op: a :class: `Operator` instance
		forward operator for inverse problems
	backward_op: a :class: `Operator` instance
		backward operator for inverse problems
	log_freq_train: a :class: `int` instance
		frequency for logging training information
	log_freq_val: a :class: `int` instance
		frequency for logging validating information

	other_config: a :class: `dict`
		Other configurations for customization.
	Notes
	----------
	If a subclass overwrites the constructor, it should call the super
	constructor with *args and **kwargs.
	"""

	def __init__(self, net, dataloader, batch_size, loss_fun, measure_fun, 
		optimizer=None, num_epochs=1, device='gpu', scheduler=None,
		save_checkpoint_freq=0, train=True, validate=True,
		forward_op=None, backward_op=None, log_freq_train=100, log_freq_val=50, 
		**other_config
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
		self._save_ckpt_freq = save_checkpoint_freq
		self._other_config = other_config
		self._train_flag = train
		self._validate_flag = validate
		if self._train_flag is True and self._optimizer is None:
			raise ValueError("Optimizer should be set for training.")
		if self._train_flag is True and self._train_loader is None:
			raise ValueError("Attending to train on training dataset but the data loader is none.")
		if self._validate_flag is True and self._val_loader is None:
			raise ValueError("Attending to validate on validating dataset but the data loader is none.")
		self._train_logger = None
		self._val_logger = None
		self._log_freq_train = log_freq_train
		self._log_freq_val = log_freq_val
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
		"""Base training logger initialization, some base arguments are logged out.
		For logging more advanced arguments it could be finished in _sub_init_logger
		in subclasses.
		"""
		train_log_file = os.path.join(exp_dir, 'train.log')
		val_log_file = os.path.join(exp_dir, 'val.log')
		if os.path.exists(train_log_file):
			os.remove(train_log_file)
		if os.path.exists(val_log_file):
			os.remove(val_log_file)
		self._train_logger = setup_logger(name='train_logger', log_file=train_log_file)
		self._val_logger = setup_logger(name='val_logger', log_file=val_log_file)

		self._train_logger.info("%s basic setting: train_batch_size=%d, val_batch_size=%d, \
			epochs=%d, optimizer=%s, lr_scheduler=%s, weight_decay=%.6f", 
			self.__class__.__name__,
			self._train_batch_size, self._val_batch_size, 
			self._num_epochs, self._optimizer.__class__.__name__, self._scheduler.__class__.__name__, 
			self._optimizer.defaults['weight_decay'])
		self._val_logger.info("%s basic setting: train_batch_size=%d, val_batch_size=%d, \
			epochs=%d, optimizer=%s, lr_scheduler=%s, weight_decay=%.6f", self.__class__.__name__, 
			self._train_batch_size, self._val_batch_size, 
			self._num_epochs, self._optimizer.__class__.__name__, self._scheduler.__class__.__name__, 
			self._optimizer.defaults['weight_decay'])

		self._sub_init_logger()

		self._train_logger.info("Epoch \t Iterations \t LR \t Train Loss \t Train Loss(avg) \t \
				Train measure \t Train measure(avg)")
		self._val_logger.info("Epoch \t Iterations \t LR \t Val Loss \t Val Loss(avg) \t \
				Val measure \t Val measure(avg)")

	def _sub_init_logger(self):
		"""More logging information in subclasses.
		"""
		pass

	def run(self, exp_dir=None):
		if self._train_flag:
			assert exp_dir != None
			os.makedirs(exp_dir, exist_ok=True)
			self._init_logger(exp_dir)
			save_model_path = os.path.join(exp_dir, 'ckpt.pth.tar')
			best_model_path = os.path.join(exp_dir, 'best.pth.tar')
			best_measure = 0.
			best_epoch = 0.
			start = time.time()
			for epoch in range(1, self._num_epochs+1):
				if self._save_ckpt_freq:
					if epoch%self._save_ckpt_freq==0:
						save_model_path = os.path.join(exp_dir, "epoch_%d.pth.tar"%epoch)

				self._train(epoch)
				torch.save({'epoch': epoch,
					'state_dict': self._net.state_dict()}, save_model_path)

				if self._validate_flag:
					avg_loss, avg_measre = self._val(epoch)
					if avg_measre is not None and avg_measre.avg >= best_measure:
						torch.save({'epoch': epoch,
							'state_dict': self._net.state_dict(),
							'measure': avg_measre.avg,
							'loss': avg_loss.avg}, best_model_path)
						best_measure = avg_measre.avg
						best_epoch = epoch
					print('Validating Best Measure: %.4f, epoch %d' % (best_measure, best_epoch))
			end = time.time()
			self._train_logger.info('Total train time: %.4f minutes', (end - start)/60)
		else:
			start = time.time()
			avg_loss, avg_measre = self._val(epoch=1)
			end = time.time()
			print('Testing loss: %.4f, measure: %.4f' % (avg_loss.avg, avg_measre.avg))
			print('Elapsed time: %.2f s' % (end - start))
