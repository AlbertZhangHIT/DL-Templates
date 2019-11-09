import torch
from torch.optim.optimizer import Optimizer, required

def complex_div(x, y):
    """pytorch complex division 
        z = x/y
    """
    z = torch.zeros_like(x)
    dnorm = y[...,0]*y[...,0] + y[...,0]*y[...,0]
    z[..., 0] = (x[...,0]*y[...,0] + x[...,1]*y[...,1])/dnorm
    z[..., 1] = (x[...,1]*y[...,0] - x[...,0]*y[...,1])/dnorm
    return z

class LSSGD(Optimizer):
    r"""Implements laplacian smoothed stochastic gradient descent (optionally with momentum).
      Osher, Stanley and Wang, Bao and Yin, Penghang and Luo, Xiyang and Barekat, 
        Farzin and Pham, Minh and Lin, Alex
            Laplacian smoothing gradient descent. arXiv preprint arXiv:1806.06317

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.LSSGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of LSSGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, sigma=0,
                momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if sigma < 0.0:
            raise ValueError("Invalid sigma value: {}".format(sigma))

        defaults = dict(lr=lr, sigma=sigma, momentum=momentum,
                    dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LSSGD, self).__init__(params, defaults)

        sizes = []
        for param in self.param_groups[0]['params']:
            sizes.append(torch.numel(param))
        cs = []
        for s in sizes:
            if s >= 2:
                v = torch.zeros(s)
                v[0], v[1], v[-1] = -2., 1., 1.
                vv = torch.rfft(v, 1, onesided=False)
                c = 1. - sigma*vv # the image part is zeros
                cs.append(c)
            else:
                cs.append(None)
        self.sizes = sizes
        self.cs = cs


    def __setstate__(self, state):
        super(LSSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('sigma', 0)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            sigma = group['sigma']

            index = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                # Laplacian smoothing gradient
                if sigma > 0:
                    if self.cs[index] is not None:
                        g = p.grad.data.view(-1)
                        fg= torch.rfft(g, 1, onesided=False)
                        g = torch.zeros_like(fg)
                        self.cs[index] = self.cs[index].to(fg.device)
                        g = complex_div(fg, self.cs[index])
                        g = torch.irfft(g, 1, onesided=False).view(p.grad.data.size())
                        p.grad.data = g
                    index += 1

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
