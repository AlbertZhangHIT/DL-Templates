from torch.autograd import grad
from math import sqrt, inf
import torch
from .gradient import GradientPerturb

class IterativeBasePerturbation(GradientPerturb):
    """Base class for iterative (projected) gradient perturbations.
    """
    def _initialize(self):
        if self._step_size is None:
            self._step_size = self._eps / self._num_steps


class LinfIterativePerturbation(IterativeBasePerturbation):
    """ 
    PGD Adversarial Training
    """
    def _clip_perturbation(self, perturbed, original):
        perturbed = torch.min(torch.max(perturbed, original-self._eps),
                             original+self._eps)
        perturbed = torch.clamp(perturbed, self._min_, self._max_)
        return perturbed

    def _gradient(self, x, y):
        with torch.enable_grad():
            xx = self._forward_op(x)
            logits = self._net(xx)
            loss = self._criterion(logits, y)        
        dx = grad(loss, x)[0] 
        dx = self._backward_op(dx)
        gradient = dx.sign() * (self._max_ - self._min_)

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
            xx = self._forward_op(x)
            logits = self._net(xx)
            loss = self._criterion(logits, y)        
        dx = grad(loss, x)[0] 
        dx = self._backward_op(dx)
        norm = dx.pow(2).mean().sqrt()
        norm = max(1e-15, norm)
        gradient = dx / norm * (self._max_ - self._min_)

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
            xx = self._forward_op(x)
            logits = self._net(xx)
            loss = self._criterion(logits, y)        
        dx = grad(loss, x)[0] 
        dx = self._backward_op(dx)
        norm = dx.abs().mean()
        norm = max(1e-15, norm)
        gradient = dx / norm * (self._max_ - self._min_)

        return self._step_size * gradient   