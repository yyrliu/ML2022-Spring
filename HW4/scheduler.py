from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_transformer_milestone_scheduler(
	optimizer: Optimizer,
	num_warmup_steps: int,
	switch_to_milestone_steps: int,
	milestones: int = 5000,
	milestone_decay_rate: float = 0.7,
	min_lr: float = 0.01,
	last_epoch: int = -1,
):
	def lr_lambda(current_step):
		if current_step < switch_to_milestone_steps:
			step = current_step + 1
			return min(step**(-0.5), step * num_warmup_steps**(-1.5)) / num_warmup_steps**(-0.5)
		else:
			return max(switch_to_milestone_steps**(-0.5) / num_warmup_steps**(-0.5) * (milestone_decay_rate ** ((current_step - switch_to_milestone_steps) // milestones + 1)), min_lr)

	return LambdaLR(optimizer, lr_lambda, last_epoch)
