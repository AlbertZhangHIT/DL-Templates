import abc
import time
import os, shutil
import torch
import torch.nn as nn
from .base import BaseTraining, AverageMeter
from tqdm import tqdm
import warnings


""" Free Adversarial Training
Reference: Shafahi, A., Najibi, M., Ghiasi, A., Xu, Z., 
	Dickerson, J., Studer, C., Davis, L.S., Taylor, G. and Goldstein, T., 2019. 
	Adversarial Training for Free!. arXiv preprint arXiv:1904.12843.

https://github.com/mahyarnajibi/FreeAdversarialTraining
"""

class FreeAdversarialBaseTraining(BaseTraining):
	"""Free Adversarial Training frames if both data and label are ready.
	The measure function only receives two arguments.
	If you want to redirect the logging information to files using `tee`,
	please use `python -u yourscript.py | tee yourfile` to get the expected
	results. (Refer to https://github.com/tqdm/tqdm/issues/706 answered by f0k)

	The method _initialize should be overloaded to register the perturbation method
	 through the argument 'perturb' in `other_config`.
	"""
	def _initialize(self):
		if self._perturb is None:
			raise KeyError("Perturbation method is required.")

	def _global_adversarial_noise(self, data):
		self.adversarial_noise = torch.zeros((self._train_batch_size,)+data.size()).to(self._device)

	def run(self, exp_dir):
		example = self._train_loader.dataset.__getitem__(1)[0]
		self._global_adversarial_noise(example)
		self._mean = self._mean.expand_as(example).to(self._device)
		self._std = self._std.expand_as(example).to(self._device)

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
		self._logger.close()	


class FreeAdversarialTraining(FreeAdversarialBaseTraining):
	def _train(self, epoch):
		avg_loss = AverageMeter()
		avg_measre = AverageMeter()
		data_stream = tqdm(enumerate(self._train_loader, 1))

		for i, (data, label) in data_stream:
			# train on the same minibatch `_num_replay` times
			for _ in range(self._num_replay):
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
				current_measure = self._measure(logits.detach(), label.detach())
				avg_measre.update(current_measure, data.size(0))

				self._optimizer.zero_grad()
				self._net.zero_grad()
				current_loss.backward()		

				# Update the noise for the next iteration
				pert = self._perturb(noise.grad)
				self.adversarial_noise += pert.data
				self.adversarial_noise.clamp_(-self._perturb._eps, self._perturb._eps)

				self._optimizer.step()			

				if self._scheduler is not None:
					self._scheduler.step()
					lr = self._scheduler.get_lr()[0]
				else:
					lr = self._optimizer.defaults['lr']

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
			if (current_iter-1)%self._log_freq_train == 0:
				self._train_logger.info('%d \t %d \t %.5f \t %.4f \t %.4f \t %.4f \t %.4f', 
					epoch, current_iter, lr, avg_loss.val, avg_loss.avg, 
					avg_measre.val, avg_measre.avg)

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
				num_batches = len(self._val_loader)
				current_iter = (epoch - 1) * num_batches + i

				data, label = data.to(self._device), label.to(self._device)	
				data.sub_(self._mean).div_(self._std)	

				logits = self._net(data)
				current_loss = self._loss_fun(logits, label)
				avg_loss.update(current_loss.item(), data.size(0))
				current_measure = self._measure(logits.detach(), label.detach())
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
				if self._train_flag:
					if self._scheduler is not None:
						lr = self._scheduler.get_lr()[0]
					else:
						lr = self._optimizer.defaults['lr']					
					if (current_iter-1)%self._log_freq_val == 0:
						self._val_logger.info('%d \t %d \t %.5f \t %.4f \t %.4f \t %.4f \t %.4f', 
							epoch, current_iter, self._scheduler.get_lr()[0], avg_loss.val, avg_loss.avg, 
							avg_measre.val, avg_measre.avg)		
		return avg_loss, avg_measre
