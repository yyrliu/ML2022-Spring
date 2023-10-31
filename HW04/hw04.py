import numpy as np
import torch
import random
import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(87)



# Comformer2 Implementation



import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
	optimizer: Optimizer,
	num_warmup_steps: int,
	num_training_steps: int,
	num_cycles: float = 0.5,
	last_epoch: int = -1,
):
	"""
	Create a schedule with a learning rate that decreases following the values of the cosine function between the
	initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
	initial lr set in the optimizer.

	Args:
		optimizer (:class:`~torch.optim.Optimizer`):
		The optimizer for which to schedule the learning rate.
		num_warmup_steps (:obj:`int`):
		The number of steps for the warmup phase.
		num_training_steps (:obj:`int`):
		The total number of training steps.
		num_cycles (:obj:`float`, `optional`, defaults to 0.5):
		The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
		following a half-cosine).
		last_epoch (:obj:`int`, `optional`, defaults to -1):
		The index of the last epoch when resuming training.

	Return:
		:obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
	"""
	def lr_lambda(current_step):
		# Warmup
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		# decadence
		progress = float(current_step - num_warmup_steps) / float(
			max(1, num_training_steps - num_warmup_steps)
		)
		return max(
			0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule(
	optimizer: Optimizer,
	num_training_steps: int,
	num_cycles: float = 0.5,
	last_epoch: int = -1,
):
	"""
	Create a schedule with a learning rate that decreases following the values of the cosine function between the
	initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
	initial lr set in the optimizer.

	Args:
		optimizer (:class:`~torch.optim.Optimizer`):
		The optimizer for which to schedule the learning rate.
		num_warmup_steps (:obj:`int`):
		The number of steps for the warmup phase.
		num_training_steps (:obj:`int`):
		The total number of training steps.
		num_cycles (:obj:`float`, `optional`, defaults to 0.5):
		The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
		following a half-cosine).
		last_epoch (:obj:`int`, `optional`, defaults to -1):
		The index of the last epoch when resuming training.

	Return:
		:obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
	"""
	def lr_lambda(current_step):
		progress = float(current_step) / float(
			max(1, num_training_steps)
		)
		return max(
			0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_exp_schedule_with_warmup(
	optimizer: Optimizer,
	num_warmup_steps: int,
	num_training_steps: int,
	decay_const: float = 2.5,
	last_epoch: int = -1,
):
	"""
	Create a schedule with a learning rate that decreases following the values of the cosine function between the
	initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
	initial lr set in the optimizer.

	Args:
		optimizer (:class:`~torch.optim.Optimizer`):
		The optimizer for which to schedule the learning rate.
		num_warmup_steps (:obj:`int`):
		The number of steps for the warmup phase.
		num_training_steps (:obj:`int`):
		The total number of training steps.
		num_cycles (:obj:`float`, `optional`, defaults to 0.5):
		The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
		following a half-cosine).
		last_epoch (:obj:`int`, `optional`, defaults to -1):
		The index of the last epoch when resuming training.

	Return:
		:obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
	"""
	def lr_lambda(current_step):
		# Warmup
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		# decadence
		progress = float(current_step - num_warmup_steps) / float(
			max(1, num_training_steps - num_warmup_steps)
		)
		return max(
			0.0, math.exp(-decay_const * progress)
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_transformer_scheduler(
	optimizer: Optimizer,
	num_warmup_steps: int,
	last_epoch: int = -1,
):

	def lr_lambda(current_step):
		step = current_step + 1
		return min(step**(-0.5), step * num_warmup_steps**(-1.5)) / num_warmup_steps**(-0.5)
	
	return LambdaLR(optimizer, lr_lambda, last_epoch)
	
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmax(nn.Module):
    # https://github.com/CoinCheung/pytorch-loss
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 feat_norm=False,
                 weight_norm=True,
                 m=0.35,
                 s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.feat_norm = feat_norm
        self.weight_norm = weight_norm
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.empty(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[1] == self.in_feats
        assert x.size()[0] == lb.size()[0]

        if self.feat_norm:
            x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-9)
            x = torch.div(x, x_norm)

        if self.weight_norm:
            w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-9)
            w = torch.div(self.W, w_norm)
            costh = torch.mm(x, w)
        else:
            costh = torch.mm(x, self.W)

        delt_costh = torch.zeros_like(costh).scatter_(1, lb.unsqueeze(1), self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return costh, loss
    
class ComformerAMSLoss2(nn.Module):
    def __init__(self, *, comformer_v=1, m, ams_feat_norm, ams_weight_norm, ams_s, pred_layer, n_spks=600, d_model=80, **kargs):
        super().__init__()
        if comformer_v == 1:
            self.comformer = Comformer(n_spks=n_spks, d_model=d_model, pred_layer=pred_layer, **kargs)
        if comformer_v == 2:
            self.comformer = Comformer2(n_spks=n_spks, d_model=d_model, pred_layer=pred_layer, **kargs)
        self.amsLoss = AMSoftmax(d_model, n_spks, m=m, s=ams_s, feat_norm=ams_feat_norm, weight_norm=ams_weight_norm)
        
    def forward(self, mels, labels):
        x = self.comformer(mels)
        out, loss = self.amsLoss(x, labels)
        return out, loss
    
import torch


def model_fn(batch, model, criterion, device):
	"""Forward a batch through the model."""

	mels, labels = batch
	mels = mels.to(device)
	labels = labels.to(device)

	outs = model(mels)
	# print(outs.shape)
	# print(outs[0])
	# print(labels.shape)
	# raise Exception
	loss = criterion(outs, labels)

	# Get the speaker id with highest probability.
	preds = outs.argmax(1)
	# Compute accuracy.
	accuracy = torch.mean((preds == labels).float())

	return loss, accuracy

def model_ams_loss_fn(batch, model, device):
	"""Forward a batch through the model."""

	mels, labels = batch
	mels = mels.to(device)
	labels = labels.to(device)

	outs, loss = model(mels, labels)
	# print(outs.shape)
	# print(outs[0])
	# print(labels.shape)
	# raise Exception

	# Get the speaker id with highest probability.
	preds = outs.argmax(1)
	# Compute accuracy.
	accuracy = torch.mean((preds == labels).float())

	return loss, accuracy

from tqdm import tqdm
import torch


def valid(dataloader, model, criterion, device): 
	"""Validate on validation set."""

	model.eval()
	running_loss = 0.0
	running_accuracy = 0.0
	pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

	for i, batch in enumerate(dataloader):
		with torch.no_grad():
			loss, accuracy = model_fn(batch, model, criterion, device)
			running_loss += loss.item()
			running_accuracy += accuracy.item()

		pbar.update(dataloader.batch_size)
		pbar.set_postfix(
			loss=f"{running_loss / (i+1):.2f}",
			accuracy=f"{running_accuracy / (i+1):.2f}",
		)

	pbar.close()
	model.train()

	return (running_loss / len(dataloader), running_accuracy / len(dataloader))

def valid_ams_loss(dataloader, model, device): 
	"""Validate on validation set."""

	model.eval()
	running_loss = 0.0
	running_accuracy = 0.0
	pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

	for i, batch in enumerate(dataloader):
		with torch.no_grad():
			loss, accuracy = model_ams_loss_fn(batch, model, device)
			running_loss += loss.item()
			running_accuracy += accuracy.item()

		pbar.update(dataloader.batch_size)
		pbar.set_postfix(
			loss=f"{running_loss / (i+1):.2f}",
			accuracy=f"{running_accuracy / (i+1):.2f}",
		)

	pbar.close()
	model.train()

	return (running_loss / len(dataloader), running_accuracy / len(dataloader))

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW, RAdam, SGD
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary




def parse_args():
    """arguments"""
    config = {
        "data_dir": "./Dataset",
        "save_path": "model.ckpt",
        "batch_size": 64,
        "n_workers": 8,
        "valid_steps": 2000,
        "warmup_steps": 7500,
        "sdg_step": 50000,
        "save_steps": 10000,
        "total_steps": 100000,
        "learning_rate": 1e-3,
        "comment": "d_model=100, nhead=4, comformer2_conv_2, drops, layers=12, self_attention_pooling_2, pred_layer_n=2, AMS_loss_2_with_norm, m=0.2, s=30, lr=0.001, milestone_sch_decay=0.85, batch_n=64, warmup_steps=7.5k, start_milestones=30k, total_steps=100k"
    }
    return config


def main(
    data_dir,
    save_path,
    batch_size,
    n_workers,
    valid_steps,
    warmup_steps,
    sdg_step,
    total_steps,
    save_steps,
    learning_rate,
    comment,
):
    writer = SummaryWriter(log_dir=f"./runs/{comment}")

    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!",flush = True)

    # model = Classifier(d_model=240, nhead=6, encoder_layers=1, n_spks=speaker_num).to(device)
    # model = ComformerAMSLoss(d_model=240, nhead=6, comformer_layers=4, n_spks=speaker_num).to(device)
    model = ComformerAMSLoss2(comformer_v=2, m=0.2, ams_s=30, d_model=100, ams_weight_norm=True, ams_feat_norm=True, pred_layer=2, nhead=4, comformer_layers=12, n_spks=speaker_num, norm_after_cf_block=False).to(device)

    # print(sum(p.numel() for p in model.parameters()))
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model = RComformer(num_classes=speaker_num, input_dim=40, encoder_dim=240, num_attention_heads=6, num_encoder_layers=4).to(device)
    # model = RComformer(num_classes=speaker_num, input_dim=40, encoder_dim=64, num_attention_heads=1, num_encoder_layers=8).to(device)

    print(sum(p.numel() for p in model.parameters()))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # return

    # criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # criterion_validation = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # optimizer = RAdam(model.parameters(), lr=1e-3)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # scheduler = get_transformer_scheduler(optimizer, warmup_steps)
    scheduler = get_transformer_milestone_scheduler(optimizer, warmup_steps, 30000, min_lr=0.05, milestone_decay_rate=0.85, milestones=7500)
    # scheduler = get_cosine_schedule(optimizer, total_steps)
    # scheduler = get_exp_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!", flush = True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    train_loss = []
    train_acc = []
    grad_norm = []

    step_size = batch_size // 32

    print(f"step_size = {step_size}")

    for step in range(total_steps // step_size):

        step = (step + 1) * step_size

        # if step == sdg_step:
        # 	optimizer = SGD(model.parameters(), lr=scheduler.get_last_lr()[0], momentum=0.8, weight_decay=1e-5)
        # 	scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        # Get data
        try:
            batch = next(train_iterator)
            # print(batch[0].shape)
            # raise Exception
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        # print(batch[0].shape)
        # return
        # loss, accuracy = model_fn(batch, model, criterion, device)
        loss, accuracy = model_ams_loss_fn(batch, model, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Updata model
        loss.backward()
        optimizer.step()
        for _ in range(step_size):
            scheduler.step()

        # if (step + 1)% 2000 == 0:
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             writer.add_histogram(f"params/{name}", param.detach(), step)


        grad_norm.append(torch.max(torch.stack([p.grad.detach().abs().max() for p in model.parameters() if p.requires_grad])))

        optimizer.zero_grad()

        # Log
        for _ in range(step_size):
            pbar.update()

        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step,
        )
        train_loss.append(batch_loss)
        train_acc.append(batch_accuracy)

        # Do validation		
        if step % valid_steps == 0:
            pbar.close()

            # valid_loss, valid_accuracy = valid(valid_loader, model, criterion_validation, device)
            valid_loss, valid_accuracy = valid_ams_loss(valid_loader, model, device)
            writer.add_scalar("Accuracy/valid", valid_accuracy, step)
            writer.add_scalar("Loss/valid", valid_loss, step)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if step % save_steps == 0 and best_state_dict is not None:
            torch.save({
                'step': step,
                'model_state_dict': best_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),    # HERE IS THE CHANGE
                }, f"{comment}.ckpt")
            # torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

        if step % 1000 == 0:
            writer.add_scalar("Loss/train", sum(train_loss) / len(train_loss), step)
            writer.add_scalar("Accuracy/train", sum(train_acc) / len(train_acc), step)
            writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], step)
            train_loss = []
            train_acc = []

            for idx, gn in enumerate(grad_norm):
                writer.add_scalar("GradNorm/train", gn, step - (1000 - 1 - idx * step_size))

            grad_norm = []

    pbar.close()
    writer.close()

# if __name__ == "__main__":
# 	main(**parse_args())

config = parse_args()

# config["comment"] = "d_model=240, nhead=6, comformer_with_layer_norm, layers=4, self_attention_pooling_2, no_pred_layer, AMS_loss_2_with_norm, m=0.325, lr=0.001, transformer_milestone_scheduler, warmup_steps=5k, start_milestones=35k, total_steps=70k, padding=1e-20, v1"

main(**config)

# config["comment"] = "d_model=240, nhead=6, comformer_with_layer_norm, layers=4, self_attention_pooling_2, no_pred_layer, AMS_loss_2_with_norm, m=0.325, lr=0.001, transformer_milestone_scheduler, warmup_steps=5k, start_milestones=35k, total_steps=70k, padding=1e-20, v2"

# main(**config)

# config["comment"] = "d_model=240, nhead=6, comformer_with_layer_norm, layers=4, self_attention_pooling_2, no_pred_layer, AMS_loss_2_with_norm, m=0.325, lr=0.001, transformer_milestone_scheduler, warmup_steps=5k, start_milestones=35k, total_steps=70k, padding=1e-20, v3"

# main(**config)

# config["comment"] = "d_model=240, nhead=6, comformer_with_layer_norm, layers=4, self_attention_pooling_2, no_pred_layer, AMS_loss_2_with_norm, m=0.325, lr=0.001, transformer_milestone_scheduler, warmup_steps=5k, start_milestones=35k, total_steps=70k, padding=1e-20, v4"

# main(**config)

# config["comment"] = "d_model=240, nhead=6, comformer_with_layer_norm, layers=4, self_attention_pooling_2, no_pred_layer, AMS_loss_2_with_norm, m=0.325, lr=0.001, transformer_milestone_scheduler, warmup_steps=5k, start_milestones=35k, total_steps=70k, padding=1e-20, v5"

# main(**config)


