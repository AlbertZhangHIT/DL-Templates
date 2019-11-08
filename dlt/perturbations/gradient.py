"""Functions to adversarially perturb an input image during net training"""
from torch.autograd import grad
from math import sqrt, inf
import torch
import abc
from ..operator import forwardOperator, backwardOperator

class GradientPerturb(abc.ABC):
    """Base class for all adversarial training perturbations.
    Parameters
    ------------
    net : a :class: `torch.nn.Module` instance
        The net should be neural network in torch.
    criterion: a :class: `torch.nn.Loss`
        The loss function used in training neural networks.
    eps: a :class: `float`
        The maximum quantity of adversarial noise.
    num_steps: a :class: `int`
        The number of times of adversarial attack.
    step_size: a :class: `float`
        The step size of each step adversarial attack.
    max_: a :class: `float`
        The maximum numerical bound.
    min_: a :class: `float`
        The minimum numerical bound.
    random_start: a :class: `bool`
        Whether to do random perturbation before adversarial attack.
    """

    def __init__(self, net, criterion, eps=0., num_steps=1, 
        step_size=None, min_=0., max_=1., random_start=False, 
        forward_op=forwardOperator(), backward_op=backwardOperator()):
        """Initialize with a net, a maximum Euclidean perturbation 
        distance epsilon, and a criterion (eg the loss function)"""
        self._net = net
        self._criterion = criterion
        self._eps = eps
        self._num_steps = num_steps
        self._step_size = step_size
        self._min_ = min_
        self._max_ = max_
        self._random_start = random_start
        self._forward_op = forward_op
        self._backward_op = backward_op

        self._initialize()

    def _initialize(self):
        pass

    def _prep_net(self):
        self._net.eval()
        for p in self._net.parameters():
            p.requires_grad_(False)

    def _release_net(self):
        self._net.train()
        for p in self._net.parameters():
            p.requires_grad_(True)

    @abc.abstractmethod
    def _clip_perturbation(self, perturbed, original):
        raise NotImplementedError

    @abc.abstractmethod
    def _gradient(self, x, y):
        raise NotImplementedError

    def __call__(self, x, y):
        self._prep_net()
 
        if self._random_start:
            noise = torch.zeros_like(x).uniform_(-self._eps, self._eps)
            x_hat = x.clone() + noise
        else:
            x_hat = x.clone()
        for _ in range(self._num_steps):
            x_hat.requires_grad_(True)
            gradient = self._gradient(x_hat, y)
            x_hat = x_hat.detach() + gradient
            x_hat = self._clip_perturbation(x_hat, x)
        self._release_net()

        return x_hat.detach()        

class SingleStepGradientPerturb(GradientPerturb):
    """Single step perturbation
    """
    def _initialize(self):
        self._num_steps = 1
        self._step_size = self._eps

    def _clip_perturbation(self, perturbed, original):
        return perturbed

class L1Perturbation(SingleStepGradientPerturb):
    """
    Penalize a loss function by the L1 norm of the loss's gradient.

    Equivalent to the Fast Gradient Signed Method.
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
        dx = grad(loss, xx)[0]
        dx = self._backward_op(dx)
        gradient = dx.sign() * (self._max_ - self._min_)

        return self._step_size * gradient

FGSM = L1Perturbation

class L2Perturbation(SingleStepGradientPerturb):
    """
    Penalize a loss function by the L2 norm of the loss's gradient.
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

        dxshape = dx.shape
        dx = dx.view(dxshape[0],-1)
        dxn = dx.norm(dim=1, keepdim=True)
        b = (dxn>0).squeeze()
        dx[b] = dx[b]/dxn[b]
        gradient = dx.view(*dxshape) * (self._max_ - self._min_)

        return self._step_size * gradient


class LInfPerturbation(SingleStepGradientPerturb):
    """
    Penalize a loss function by the L-infinity norm of the loss's gradient.
    """

    def _gradient(self, x, y):
        with torch.enable_grad():
            xx = self._forward_op(x)
            logits = self._net(xx)
            loss = self._criterion(logits, y)
        dx = grad(loss, x)[0]
        dx = self._backward_op(dx)  

        dxshape = dx.shape
        bsz = dxshape[0]
        dx = dx.view(bsz,-1)
        ix = dx.abs().argmax(dim=-1)
        dx_ = torch.zeros_like(dx.view(bsz,-1))
        jx = torch.arange(bsz, device=dx.device)
        dx_[jx,ix] = dx.sign()[jx,ix]
        gradient = dx_.view(*dxshape) * (self._max_ - self._min_)

        return self._step_size * gradient
