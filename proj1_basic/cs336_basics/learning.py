import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math

import torch.optim.optimizer

def cross_entropy_loss(predictions, targets):
    return torch.nn.functional.cross_entropy(predictions, targets)


def learning_rate_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if t < warmup_iters:
        return lr_max * t / warmup_iters
    elif warmup_iters <= t and t <= cosine_cycle_iters:
        progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    else:
        return lr_min
    

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    total_norm = 0.0
    params = [p for p in parameters if p.grad is not None]
    for p in params:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # Scale gradients if needed
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            p.grad.data.mul_(scale)
    """
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None if closure is None else closure()        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            decay = group['weight_decay']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                t = state.get('step', 1)
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                state['m'] = state['m'] * beta1 + grad * (1 - beta1)
                state['v'] = state['v'] * beta2 + grad * grad * (1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1
                denom =  state['v'].sqrt() + eps
                p.data -= alpha_t * state['m'] / denom               
                p.data -= p.data * decay * lr
                
                state['step'] = t + 1
        
        return loss