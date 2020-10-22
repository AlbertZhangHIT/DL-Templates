from torch.autograd import grad
from math import sqrt, inf
import torch
from .gradient import GradientPerturb

def clamp(x, lower_limit, upper_limit):
    return torch.max(torch.min(x, upper_limit), lower_limit)

class IterativeBasePerturbation(GradientPerturb):
    """Base class for iterative (projected) gradient perturbations.
    """
    def _initialize(self):
        if self._step_size is None:
            self._step_size = self._eps


class LinfIterativePerturbation(IterativeBasePerturbation):
    """ 
    PGD Adversarial Training
    """
    def _clip_perturbation(self, perturbed, original):
        perturbed = clamp(perturbed, original - self._eps_tensor, original + self._eps_tensor)
        perturbed = clamp(perturbed, self._lower_limit, self._upper_limit)
        return perturbed 

    def _gradient(self, x, y):
        with torch.enable_grad():
            logits = self._net(x)
            loss = self._criterion(logits, y)        
        dx = grad(loss, x)[0] 
        gradient = dx.sign()

        return self._step_size * gradient

ProjectedGradientDescent = LinfIterativePerturbation
PGD = ProjectedGradientDescent

class L2IterativePerturbation(IterativeBasePerturbation):
    """Iterative perturbation minimizes L2 distance and takes 
    L2 clipping.
    """
    def _clip_perturbation(self, perturbed, original):
        norm = perturbed.pow(2).mean().sqrt()
        norm = max(1e-15, norm)
        factor = min(1, self._eps * (self._max_ - self._min_) / norm)

        return perturbed * factor

    def _gradient(self, x, y):
        with torch.enable_grad():
            logits = self._net(x)
            loss = self._criterion(logits, y)        
        dx = grad(loss, x)[0] 
        norm = dx.pow(2).mean().sqrt()
        norm = max(1e-15, norm)
        gradient = dx / norm

        return self._step_size * gradient

class L1IterativePerturbation(IterativeBasePerturbation):
    """Iterative perturbation minimizes L1 distance and takes 
    L1 clipping.
    """
    def _clip_perturbation(self, perturbed, original):
        norm = perturbed.abs().mean()
        norm = max(1e-15, norm)
        factor = min(1, self._eps * (self._max_ - self._min_) / norm)

        return perturbed * factor

    def _gradient(self, x, y):
        with torch.enable_grad():
            logits = self._net(x)
            loss = self._criterion(logits, y)        
        dx = grad(loss, x)[0] 
        norm = dx.abs().mean()
        norm = max(1e-15, norm)
        gradient = dx.abs() / norm

        return self._step_size * gradient   