import torch
import numpy as np
import matplotlib.pyplot as plt
import config as cfg

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, decay):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self.decay = decay or 0.5
        self._rate = 0
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
        
    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 0 if not step else self.factor * self.get_rate(step)
    
    def get_rate(self, step_num):
        lr = self.model_size ** (-0.5) * min(step_num ** (-self.decay), step_num * self.warmup ** (-1.5))
        return lr

def main():

    optimizer = NoamOpt(
        model_size=cfg.arch_args.encoder_embed_dim, 
        factor=cfg.config.lr_factor, 
        warmup=cfg.config.lr_warmup, 
        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))
    
    plt.plot(np.arange(1, 100000), [optimizer.rate(i) for i in range(1, 100000)])
    plt.legend([f"{optimizer.model_size}:{optimizer.warmup}"])
