import abc
import time
import os, shutil
import torch
import torch.nn as nn
from .base import BaseTraining, AverageMeter
from tqdm import tqdm
import warnings

def clamp(x, lower_limit, upper_limit):
    return torch.max(torch.min(x, upper_limit), lower_limit)

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

		self._num_replay = self._other_config.get('num_replay', 1)
		warnings.warn("Number of replays is defaultly set to 1")
		assert isinstance(self._num_replay, int)

		self._lower_limit = self._other_config.get('lower_limit', 0.)
		self._upper_limit = self._other_config.get('upper_limit', 1.)
		self._eps = self._other_config.get('eps', 0.1)
		assert type(self._lower_limit) in (float, tuple, list)
		assert type(self._upper_limit) in (float, tuple, list)
		assert type(self._eps) in (float, tuple, list)
		if type(self._lower_limit)==float:
			self._lower_limit = [self._lower_limit]
		if type(self._upper_limit)==float:
			self._upper_limit = [self._upper_limit]
		if type(self._eps)==float:
			self._eps = [self._eps]
		self._lower_limit = torch.Tensor(self._lower_limit).to(self._device)
		self._upper_limit = torch.Tensor(self._upper_limit).to(self._device)
		self._eps = torch.Tensor(self._eps).to(self._device)
		assert len(self._lower_limit)==len(self._upper_limit)==len(self._eps)

	def _sub_init_logger(self):
		self._train_logger.info("%s advaced setting: eps=%s, num_replay=%s, lower_limit=%s, upper_limit=%s", 
			self.__class__.__name__, str(self._eps), str(self._num_replay), str(self._lower_limit), str(self._upper_limit))
		self._val_logger.info("%s advaced setting: eps=%s, num_replay=%s, lower_limit=%s, upper_limit=%s", 
			self.__class__.__name__, str(self._eps), str(self._num_replay), str(self._lower_limit), str(self._upper_limit))


	def _global_adversarial_noise(self, data):
		self.adversarial_noise = torch.zeros((self._train_batch_size,)+data.size()).to(self._device)
		self.adversarial_noise.requires_grad_(True)
		if self._eps.ndim < data.ndim:
			for _ in range(data.ndim - self._eps.ndim):
				self._eps = self._eps.unsqueeze(-1)
				self._lower_limit = self._lower_limit.unsqueeze(-1)
				self._upper_limit = self._upper_limit.unsqueeze(-1)

	def run(self, exp_dir):
		if self._train_flag:
			example = self._train_loader.dataset.__getitem__(1)[0]
			self._global_adversarial_noise(example)

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
			if exp_dir is not None:
				os.makedirs(exp_dir, exist_ok=True)
			start = time.time()
			avg_loss, avg_measre = self._val(epoch=1)
			end = time.time()
			print('Testing loss: %.4f, measure: %.4f' % (avg_loss.avg, avg_measre.avg))
			print('Elapsed time: %.2f s' % (end - start))

class FreeAdversarialTraining(FreeAdversarialBaseTraining):
	def _train(self, epoch):
		avg_loss = AverageMeter()
		avg_measre = AverageMeter()
		data_stream = tqdm(enumerate(self._train_loader, 1))

		for i, (data, label) in data_stream:
			data, label = data.to(self._device), label.to(self._device)	
			# where we are
			num_batches = len(self._train_loader)
			current_iter = (epoch - 1) * num_batches + i
			# train on the same minibatch `_num_replay` times
			for _ in range(self._num_replay):
				self._net.train()			
				noisy = data + self.adversarial_noise[:data.size(0)]
				if self._forward_op is not None:
					y = self._forward_op(noisy)
				else:
					y = noisy
				logits = self._net(y)
				current_loss = self._loss_fun(logits, label)
				self._optimizer.zero_grad()
				current_loss.backward()	
				# Update the noise for the next iteration
				grad = self.adversarial_noise.grad.detach()
				self.adversarial_noise.data = clamp(self.adversarial_noise + self._eps * torch.sign(grad), 
													-self._eps, self._eps)
				self.adversarial_noise.data[:data.size(0)] = clamp(self.adversarial_noise[:data.size(0)], 
												self._lower_limit - data, self._upper_limit - data)
				self._optimizer.step()			
				self.adversarial_noise.grad.zero_()
				
				if self._scheduler is not None:
					self._scheduler.step()
					lr = self._scheduler.get_lr()[0]
				else:
					lr = self._optimizer.defaults['lr']

			avg_loss.update(current_loss.item(), data.size(0))
			current_measure = self._measure(logits.detach(), label.detach())
			avg_measre.update(current_measure, data.size(0))

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

				if self._forward_op is not None:
					y = self._forward_op(data)
				else:
					y = data
				logits = self._net(y)
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
							epoch, current_iter, lr, avg_loss.val, avg_loss.avg, 
							avg_measre.val, avg_measre.avg)		
		return avg_loss, avg_measre
