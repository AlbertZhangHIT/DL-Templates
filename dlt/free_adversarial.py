import abc
import os, shutil
import torch
import torch.nn as nn
from .base import Training, AverageMeter
from tqdm import tqdm
import warnings

class fgsm(object):
	def __init__(self, stepsize, clip_eps):
		self.stepsize = stepsize
		self.clip_eps = clip_eps

	def __call__(self, grad):
		return self.stepsize * grad.data.sign()

""" Free Adversarial Training
Reference: Shafahi, A., Najibi, M., Ghiasi, A., Xu, Z., 
	Dickerson, J., Studer, C., Davis, L.S., Taylor, G. and Goldstein, T., 2019. 
	Adversarial Training for Free!. arXiv preprint arXiv:1904.12843.

https://github.com/mahyarnajibi/FreeAdversarialTraining
"""

class FreeAversarialBaseTraining(Training):
	"""Free Adversarial Training frames if both data and label are ready.
	The measure function only receives two arguments.
	If you want to redirect the logging information to files using `tee`,
	please use `python -u yourscript.py | tee yourfile` to get the expected
	results. (Refer to https://github.com/tqdm/tqdm/issues/706 answered by f0k)

	The method _initialize should be overloaded to register the perturbation method
	 through the item 'perturb' in `other_config`.
	"""
	def _initialize(self):
		try:
			self._num_replay = self._other_config['replay']
		except KeyError:
			self._num_replay = 5
			warnings.warn("Number of replays should be set, default to 5")
		assert isinstance(self._num_replay, int)

		try:
			self._perturb = self._other_config['perturb']
		except KeyError:
			raise KeyError("Perturbation method is required in other_config")			
		try:
			self._min = self._other_config['min']
		except KeyError:
			self._min = 0.
		try:
			self._max = self._other_config['max']
		except KeyError:
			self._max = 1.		
		assert type(self._min) in (int, float) and type(self._max) in (int, float)

		try:
			self._mean = self._other_config['mean']
		except KeyError:
			self._mean = [0]
		try:
			self._std = self._other_config['std']
		except KeyError:
			self._std = [1]	
		assert type(self._mean) in (list, tuple) and type(self._std) in (list, tuple)
		self._mean = torch.Tensor(self._mean)
		self._std = torch.Tensor(self._std)

	def _global_adversarial_noise(self, data):
		self.adversarial_noise = torch.zeros((self._train_batch_size, data.size())).to(self._device)

	@abc.abstractmethod
	def _train(self, epoch):
		raise NotImplementedError

	@abc.abstractmethod
	def _val(self, epoch):
		raise NotImplementedError

	def run(self, exp_dir):
		example = self._train_loader.dataset.__getitem__(1)[0]
		self._global_adversarial_noise(example)
		self._mean = self._mean.expand_as(example).to(self._device)
		self._std = self._std.expand_as(example).to(self._device)

		best_model_path = os.path.join(exp_dir, 'best.pth.tar')
		best_measure = 0.
		best_epoch = 0.
		start = time.time()
		for epoch in range(1, self._num_epochs+1):
			save_model_path = os.path.join(exp_dir, "epoch_%d.pth.tar"%epoch)
			# Update the learning rate
			if self._scheduler is not None:
				self._scheduler.step()

			self._train(epoch)
			avg_loss, avg_measre = self._val(epoch)

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


class FreeAdversarialTraining(FreeAversarialBaseTraining):
	def _train(self, epoch):
		avg_loss = AverageMeter()
		avg_measre = AverageMeter()
		data_stream = tqdm(enumerate(self._train_loader, 1))

		for i, (data, label) in data_stream:
			self._net.train()
			# where we are
			num_batches = len(self._train_loader)
			current_iter = (epoch - 1) * num_batches + i
			data, label = data.to(self._device), label.to(self._device)			
			noise = self.adversarial_noise.clone().requires_grad_(True)		

			noisy = data + noise
			noisy.clamp_(self._min, self._max)
			noisy.sub_(self._mean).div_(self._std) # is the re-normalization crutial?

			logits = self._net(noisy)
			current_loss = self._loss_fun(logits, label)

			avg_loss.update(current_loss.item(), data.size(0))
			current_measure = self._measure(logits, label)
			avg_measre.update(current_measure, data.size(0))

			self._optimizer.zero_grad()
			current_loss.backward()		

			# Update the noise for the next iteration
			pert = self._perturb(noisy.grad)
			self.adversarial_noise += pert.data
			self.adversarial_noise.clamp_(-self._perturb.clip_eps, self._perturb.clip_eps)

			self._optimizer.step()			

			#Update the progress
			data_stream.set_description((
				'Training Epoch: [{epoch}/{epochs}] | '
				'Iteration: {iters} | '
				'progress: [{trained}/{total}] ({percent:.0f}%) | '
				'loss: {loss.val: .4f} (Avg {loss.avg:.4f}) | '
				'measure: {mvalue.val: .2f} (Avg {mvalue.avg: .4f})'
				).format(
					epoch=epoch,
					epochs=self._num_epochs,
					iters=current_iter,
					trained=i,
					total=num_batches,
					percent=(100.*i/num_batches),
					loss=avg_loss,
					mvalue=avg_measre,
					)
			)
			# Print out final information
			print('Training [{epoch}/{epochs}], Iters {iters}, \
				loss: {loss.val: .4f} (Avg {loss.avg:.4f}) | \
				measure: {mvalue.val: .2f} (Avg {mvalue.avg: .4f})'.format(
					epoch=epoch, epochs=self._num_epochs, 
					iters=current_iter, loss=avg_loss, mvalue=avg_measre)
			)

	def _val(self, epoch):
		# check validate data loader
		if self._val_loader is None:
			return None, None
		avg_loss = AverageMeter()
		avg_measre = AverageMeter()
		data_stream = tqdm(enumerate(self._val_loader, 1))		
		with torch.no_grad():
			for i, (data, label) in data_stream:
				self._net.eval()
				# where we are
				num_batches = len(self._train_loader)
				current_iter = (epoch - 1) * num_batches + i

				data, label = data.to(self._device), label.to(self._device)				
				logits = self._net(data)
				current_loss = self._loss_fun(logits, label)
				avg_loss.update(current_loss.item(), data.size(0))
				current_measure = self._measure(logits, label)
				avg_measre.update(current_measure, data.size(0))
				#Update the progress
				data_stream.set_description((
					'Validating Epoch: [{epoch}/{epochs}] | '
					'Iteration: {iters} | '
					'progress: [{trained}/{total}] ({percent:.0f}%) | '
					'loss: {loss.val: .4f} (Avg {loss.avg:.4f}) | '
					'measure: {mvalue.val: .2f} (Avg {mvalue.avg: .4f})'
					).format(
						epoch=epoch,
						epochs=self._num_epochs,
						iters=current_iter,
						trained=i,
						total=num_batches,
						percent=(100.*i/num_batches),
						loss=avg_loss,
						mvalue=avg_measre,
						)
				)
				# Print out final information
				print('Validating [{epoch}/{epochs}], Iters {iters}, \
					loss: {loss.val: .4f} (Avg {loss.avg:.4f}) | \
					measure: {mvalue.val: .2f} (Avg {mvalue.avg: .4f})'.format(
						epoch=epoch, epochs=self._num_epochs, 
						iters=current_iter, loss=avg_loss, mvalue=avg_measre)
				)			
		return avg_loss, avg_measre
