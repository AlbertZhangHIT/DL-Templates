import abc
import time
import os, shutil
import torch
import torch.nn as nn
from .base import BaseTraining, AverageMeter
from tqdm import tqdm


class PlainTraining(BaseTraining):
	"""Plain Training frames if both data and label are ready.
	The measure function only receives two arguments.
	If you want to redirect the logging information to files using `tee`,
	please use `python -u yourscript.py | tee yourfile` to get the expected
	results. (Refer to https://github.com/tqdm/tqdm/issues/706 answered by f0k)
	"""
	def _grad_clip_fun(self):
		pass

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
			self._optimizer.zero_grad()
			logits = self._net(data)
			current_loss = self._loss_fun(logits, label)
			current_loss.backward()

			self._grad_clip_fun()

			self._optimizer.step()

			avg_loss.update(current_loss.item(), data.size(0))
			current_measure = self._measure(logits, label)
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
				print('Training Epoch: [{epoch}/{epochs}] | '
					'Iteration: {iters} | '
					'loss: {loss.val: .4f} (Avg {loss.avg:.4f}) | '
					'measure: {mvalue.val: .2f} (Avg {mvalue.avg: .4f})'
					.format(
						epoch=epoch,
						epochs=self._num_epochs,
						iters=current_iter,
						loss=avg_loss,
						mvalue=avg_measre,), file=self._logger, flush=True
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
				num_batches = len(self._val_loader)
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
				if (current_iter-1)%self._log_freq_val == 0:
					print('Validating Epoch: [{epoch}/{epochs}] | '
						'Iteration: {iters} | '
						'loss: {loss.val: .4f} (Avg {loss.avg:.4f}) | '
						'measure: {mvalue.val: .2f} (Avg {mvalue.avg: .4f})'
						.format(
							epoch=epoch,
							epochs=self._num_epochs,
							iters=current_iter,
							loss=avg_loss,
							mvalue=avg_measre,
							), file=self._logger, flush=True
					)
		
		return avg_loss, avg_measre
