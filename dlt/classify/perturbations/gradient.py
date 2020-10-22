"""Functions to adversarially perturb an input image during net training"""
from torch.autograd import grad
from math import sqrt, inf
import torch
import abc

def clamp(x, lower_limit, upper_limit):
    return torch.max(torch.min(x, upper_limit), lower_limit)

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
        step_size=None, lower_limit=0., upper_limit=1., random_start=False):
        """Initialize with a net, a maximum Euclidean perturbation 
        distance epsilon, and a criterion (eg the loss function)"""
        self._net = net
        self._criterion = criterion
        self._eps = torch.Tensor([eps])
        self._num_steps = num_steps
        self._step_size = step_size
        self._lower_limit = torch.Tensor([lower_limit])
        self._upper_limit = torch.Tensor([upper_limit])
        self._random_start = random_start
        self._num_calls = 0
        self._eps_tensor = self._eps

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
        self._num_calls += 1
        noise = torch.zeros_like(x)
        if self._num_calls == 1:
            self._eps_tensor = self._eps_tensor.to(x.device)
            self._lower_limit = self._lower_limit.to(x.device)
            self._upper_limit = self._upper_limit.to(x.device)
            if self._eps_tensor.ndim < x.ndim-1:
                for _ in range(x.ndim-1-self._eps_tensor.ndim):
                    self._eps_tensor = self._eps_tensor.unsqueeze(-1)
                    self._lower_limit = self._lower_limit.unsqueeze(-1)
                    self._upper_limit = self._upper_limit.unsqueeze(-1)
        if self._random_start:
            for i in range(len(self._eps)):
                noise[:, i, ...].uniform_(-self._eps[i], self._eps[i])
            noise.data = clamp(noise, self._lower_limit - x, self._upper_limit - x)
        noise.requires_grad_(True)
        x_hat = x.clone() + noise
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

    def _clip_perturbation(self, perturbed, original):
        return perturbed

class L1Perturbation(SingleStepGradientPerturb):
    """
    Penalize a loss function by the L1 norm of the loss's gradient.

    Equivalent to the Fast Gradient Signed Method.
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

FGSM = L1Perturbation

class RFGSM(L1Perturbation):
    """
    R+FGSM: FGSM variant that prepends the gradient computation by a random step
    Reference: Ensemble adversarial training: attacks and defenses. ICLR 2018, 
    url: https://openreview.net/pdf?id=rkZvSe-RZ
    """
    def _gradient(self, x, y):
        x_new = x + self._step_size * torch.zeros_like(x).normal_().sign()
        x_new = self._clip_perturbation(x_new, x)
        with torch.enable_grad():
            logits = self._net(x_new)
            loss = self._criterion(logits, y)
        dx = grad(loss, x_new)[0]
        gradient = dx.sign()

        return (self._eps_tensor - self._step_size) * gradient


class L2Perturbation(SingleStepGradientPerturb):
    """
    Penalize a loss function by the L2 norm of the loss's gradient.
    """
    def _clip_perturbation(self, perturbed, original):
        norm = perturbed.pow(2).mean().sqrt()
        norm = max(1e-15, norm)
        one = torch.ones_like(self._eps_tensor)
        factor = torch.min(one, self._eps_tensor * (self._lower_limit - self._upper_limit) / norm)

        return perturbed * factor
        
    def _gradient(self, x, y):
        with torch.enable_grad():
            logits = self._net(x)
            loss = self._criterion(logits, y)
        dx = grad(loss, x)[0]

        dxshape = dx.shape
        dx = dx.view(dxshape[0],-1)
        dxn = dx.norm(dim=1, keepdim=True)
        b = (dxn>0).squeeze()
        dx[b] = dx[b]/dxn[b]
        gradient = dx.view(*dxshape)

        return self._step_size * gradient


class LInfPerturbation(SingleStepGradientPerturb):
    """
    Penalize a loss function by the L-infinity norm of the loss's gradient.
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

        dxshape = dx.shape
        bsz = dxshape[0]
        dx = dx.view(bsz,-1)
        ix = dx.abs().argmax(dim=-1)
        dx_ = torch.zeros_like(dx.view(bsz,-1))
        jx = torch.arange(bsz, device=dx.device)
        dx_[jx,ix] = dx.sign()[jx,ix]
        gradient = dx_.view(*dxshape)

        return self._step_size * gradient
