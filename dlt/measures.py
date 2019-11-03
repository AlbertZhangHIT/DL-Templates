import torch

def batchSNR(recon, clean, data_range=1.):
	batch_size = recon.size(0)
	norm = clean.view(batch_size, -1).pow(2).mean(dim=1)
	mse = (clean - recon).view(batch_size, -1).pow(2).mean(dim=1)
	snrs = 10. * torch.log10(norm/(mse+1e-20))

	return snrs.mean()

def batchPSNR(recon, clean, data_range=1.):
	batch_size = recon.size(0)
	mse = (clean - recon).view(batch_size, -1).pow(2).mean(dim=1)
	psnrs = 10. * torch.log10(data_range**2 / (mse+1e-20))

	return psnrs.mean()

def classPredict(prediction, label):
	_, predicted = torch.max(prediction.data, 1)

	return predicted.eq(label.data).sum().item()