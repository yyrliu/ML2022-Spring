import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

"""
examples of lr_scheduler_conf:

lr_scheduler_conf = {
    "learning_rate": 1e-3,
    "type": "transformer",
    "num_warmup_steps": 2000,
}

lr_scheduler_conf = {
    "type": "milestone",
    "num_warmup_steps": 7500,
    "min_lr": 0.05,
    "milestones": 30000,
    "decay_const": 0.85,
}

"""

def get_scheduler(
    optimizer: Optimizer, *,
    type: str,
    num_warmup_steps: int = 0,
    warmup_mode: str = "linear",
    min_lr: float = 0.0,
    last_epoch: int = -1,
    **kwargs,
):
    if type == "cosine":
        def lr_lambda(current_step):
            # kwargs {"num_training_steps": int}
            progress = float(current_step) / float(kwargs["num_training_steps"])
            return max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * progress))
            )
        
    elif type == "transformer":
        def lr_lambda(current_step):
            step = current_step + 1
            return min(step**(-0.5), step * num_warmup_steps**(-1.5)) / num_warmup_steps**(-0.5)
        
    elif type == "exp":
        def lr_lambda(current_step):
            # kwargs {"decay_const": float}
            return max(math.exp(-kwargs["decay_const"] * current_step), min_lr)
    
    elif type == "milestone":
        def lr_lambda(current_step):
            # kwargs {"milestones": list, "decay_const": float}
            return max(kwargs["decay_const"] ** (current_step // kwargs["milestones"]), min_lr)

    lr_lambda = lambda_warmup_wrapper(lr_lambda, num_warmup_steps, warmup_mode)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def lambda_warmup_wrapper(
    lambda_fn,
    num_warmup_steps: int,
    warmup_mode: str = "linear",
):
    if num_warmup_steps == 0:
        def lr_lambda(current_step):
            return lambda_fn(current_step)
    
    if warmup_mode == "linear":
        def lr_lambda(current_step):
            lr = min(1.0, float(current_step) / float(num_warmup_steps))
            return lr * lambda_fn(current_step) 
    else:
        raise NotImplementedError

    return lr_lambda
