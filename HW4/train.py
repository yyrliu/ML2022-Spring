import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from scheduler import get_transformer_milestone_scheduler
from get_dataloader import get_dataloader


def model_fn(batch, model, device):
	"""Forward a batch through the model."""

	mels, labels = batch
	mels = mels.to(device)
	labels = labels.to(device)

	outs, loss = model(mels, labels)
	
	preds = outs.argmax(1)
	accuracy = torch.mean((preds == labels).float())

	return loss, accuracy

def valid_loss(dataloader, model, device): 
	"""Validate on validation set."""

	model.eval()
	running_loss = 0.0
	running_accuracy = 0.0
	pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

	for i, batch in enumerate(dataloader):
		with torch.no_grad():
			loss, accuracy = model_fn(batch, model, device)
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
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!",flush = True)


    model = ComformerAMSLoss2(comformer_v=2, m=0.2, ams_s=30, d_model=100, ams_weight_norm=True, ams_feat_norm=True, pred_layer=2, nhead=4, comformer_layers=12, n_spks=speaker_num, norm_after_cf_block=False).to(device)


    print(sum(p.numel() for p in model.parameters()))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_transformer_milestone_scheduler(optimizer, warmup_steps, 30000, min_lr=0.05, milestone_decay_rate=0.85, milestones=7500)
    print(f"[Info]: Finish creating model!", flush = True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    train_loss = []
    train_acc = []
    grad_norm = []

    step_size = batch_size // 32

    print(f"step_size = {step_size}")
    writer = SummaryWriter(log_dir=f"./runs/{comment}")

    for step in range(total_steps // step_size):

        step = (step + 1) * step_size

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_ams_loss_fn(batch, model, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

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
            valid_loss, valid_accuracy = valid_ams_loss(valid_loader, model, device)
            writer.add_scalar("Accuracy/valid", valid_accuracy, step)
            writer.add_scalar("Loss/valid", valid_loss, step)

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