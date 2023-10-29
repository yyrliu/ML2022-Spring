import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import logging
from itertools import chain

from model.model import make_model
from loss_fn import AMSoftmax
import model_conf
from get_dataloader import get_dataloader
from scheduler import get_scheduler

logger = logging.getLogger(__name__)

def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""
    
    mels, labels = batch
    mels, labels = mels.to(device), labels.to(device)
   
    outs = model(mels)
    costh, loss = criterion(outs, labels)

    # Get the speaker id with highest probability.
    # preds = outs.argmax(1)
    preds = costh.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy

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

def get_loss_fn(type, conf):
    if type == "amsoftmax":
        return AMSoftmax(
            in_feats=conf["in_feats"],
            n_class=conf["n_class"],
            m=conf["m"],
            s=conf["s"],
            norm_affine=conf["norm_affine"],
            feat_norm=conf["feat_norm"],
        )
    else:
        raise NotImplementedError(f"Loss function type `{type}` not implemented.")
    
def main():

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

    conf = {
        "comment": "test",
        "data_dir": "data/Dataset",
        "batch_size": 64,
        "valid_steps": 2000,
        "save_steps": 2000,
        "total_steps": 30000,
        "model": {
            "input_mels": 40,
            "d_model": 100,
            "conf": model_conf.comformer_default_conf
        },
        "optimizer": {
            "lr": 1e-3,
        },
        "lr_scheduler": lr_scheduler_conf
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Use {device} now!")


    logging.info(f"Data loaded!")

    model = make_model(
        n_class=speaker_num,
        **conf["model"]
    ).to(device)
        
    criterion = AMSoftmax(
        in_feats=speaker_num,
        n_class=speaker_num,
    ).to(device)

    parameters = model.parameters(chain([model.parameters(), criterion.parameters()]))

    optimizer = AdamW(parameters, lr=conf["optimizer"]["lr"])

    print(sum(p.numel() for p in model.parameters()))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    scheduler = get_scheduler(optimizer, **conf["lr_scheduler"])

    print(f"[Info]: Finish creating model!", flush = True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=conf["valid_steps"], ncols=0, desc="Train", unit=" step")

    train_loss = []
    train_acc = []
    grad_norm = []

    train_iterator = iter(train_loader)

    writer = SummaryWriter(log_dir=f'./runs/{conf["comment"]}')

    for step in range(conf["total_steps"]):

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1)% 2000 == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f"params/{name}", param.detach(), step)
        grad_norm.append(torch.max(torch.stack([p.grad.detach().abs().max() for p in model.parameters() if p.requires_grad])))
        optimizer.zero_grad()

       
        pbar.update()

        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step,
        )
        train_loss.append(batch_loss)
        train_acc.append(batch_accuracy)

        # Do validation		
        if step % conf["valid_steps"] == 0:
            pbar.close()
            valid_loss, valid_accuracy = valid(valid_loader, model, criterion, device)
            writer.add_scalar("Accuracy/valid", valid_accuracy, step)
            writer.add_scalar("Loss/valid", valid_loss, step)

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=conf["valid_steps"], ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if step % conf["save_steps"] == 0 and best_state_dict is not None:
            torch.save({
                'config': conf,
                'step': step,
                'model_state_dict': best_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                }, f'{conf["comment"]}.ckpt')
            # torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

        if step % 1000 == 0:
            writer.add_scalar("Loss/train", sum(train_loss) / len(train_loss), step)
            writer.add_scalar("Accuracy/train", sum(train_acc) / len(train_acc), step)
            writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], step)
            train_loss = []
            train_acc = []

            for idx, gn in enumerate(grad_norm):
                writer.add_scalar("GradNorm/train", gn, step - (1000 - 1 - idx * 100))

            grad_norm = []

    pbar.close()
    writer.close()

if __name__ == "__main__":
    main()
