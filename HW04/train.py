import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import logging
from itertools import chain

import config
from model.model import make_model
from loss_fn import get_loss_fn
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

def train(dataloader, model, criterion, optimizer, scheduler, device, writers):
    
    model.train()

    pbar = tqdm(total=len(dataloader), ncols=0, desc="Train", unit=" batch")

    running_loss = 0.0
    running_accuracy = 0.0

    for i, batch in enumerate(dataloader):

        optimizer.zero_grad()
        loss, accuracy = model_fn(batch, model, criterion, device)
        running_loss += loss.item()
        running_accuracy += accuracy.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        for writer in writers:
            writer(config.step)

        config.step += 1

        pbar.update()
        pbar.set_postfix(
            loss=f"{running_loss / (i+1):.2f}",
            accuracy=f"{running_accuracy / (i+1):.2f}",
            step=config.step,
            epoch=config.epoch + 1,
        )

    pbar.close()
    config.epoch += 1

    return (running_loss / len(dataloader)), (running_accuracy / len(dataloader))

def valid(dataloader, model, criterion, device):
	"""Validate on validation set."""

	model.eval()
	running_loss = 0.0
	running_accuracy = 0.0
	pbar = tqdm(total=len(dataloader), ncols=0, desc="Valid", unit=" batch")

	for i, batch in enumerate(dataloader):
		with torch.no_grad():
			loss, accuracy = model_fn(batch, model, criterion, device)
			running_loss += loss.item()
			running_accuracy += accuracy.item()

		pbar.update()
		pbar.set_postfix(
			loss=f"{running_loss / (i+1):.2f}",
			accuracy=f"{running_accuracy / (i+1):.2f}",
		)

	pbar.close()
	return (running_loss / len(dataloader), running_accuracy / len(dataloader))

def get_param_writer(writer, model, interval):
    def _write(step):
        if (step + 1)% interval == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f"params/{name}", param.detach(), step + 1)

    return _write

def get_grad_writer(writer, model, interval):
    grad_norm = []
    def _write(step):
        nonlocal grad_norm
        grad_norm.append(torch.max(torch.stack([p.grad.detach().abs().max() for p in model.parameters() if p.requires_grad])))
        if (step + 1)% interval == 0:
            for idx, gn in enumerate(grad_norm):
                writer.add_scalar("GradNorm/train", gn, step - (interval - idx) + 1)
            grad_norm = []

    return _write

def get_train_writer(writer):
    def _write(step, loss, acc, lr):
        writer.add_scalar("Loss/train", loss, step)
        writer.add_scalar("Accuracy/train", acc, step)
        writer.add_scalar("Learning_rate", lr, step)

    return _write

def get_valid_writer(writer):
    def _write(step, loss, acc):
        writer.add_scalar("Loss/valid", loss, step)
        writer.add_scalar("Accuracy/valid", acc, step)

    return _write

def main(conf):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Use {device} now!")

    train_loader, valid_loader, speaker_num = get_dataloader(
        data_dir=conf["data_dir"],
        batch_size=conf["batch_size"]
    )
    logging.info(f"Data loaded!")

    model = make_model(
        n_class=speaker_num,
        **conf["model"]
    ).to(device)
        
    criterion = get_loss_fn(
         conf["loss_fn"]["type"],
         in_feats=speaker_num,
         n_class=speaker_num,
         **conf["loss_fn"]["conf"]
    ).to(device)

    parameters = model.parameters(chain([model.parameters(), criterion.parameters()]))

    optimizer = AdamW(parameters, lr=conf["optimizer"]["lr"])

    logging.info(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters to be trained.")

    scheduler = get_scheduler(optimizer, **conf["lr_scheduler"])

    print(f"[Info]: Finish creating model!", flush = True)

    best_accuracy = -1.0
    best_loss = 1e10
    best_state_dict = None

    writer = SummaryWriter(comment=conf["comment"], max_queue=1000)
    train_writer = get_train_writer(writer)
    valid_writer = get_valid_writer(writer)
    param_writer = get_param_writer(writer, model, 2000)
    grad_writer = get_grad_writer(writer, model, 1000)
    model_writers = [param_writer, grad_writer]

    logger.info(f"Start training in {conf['total_steps']} steps / {conf['total_steps']/len(train_loader)} epochs.")
    logger.info(f"({len(train_loader)} batches per epoch)")

    config.step = 0
    config.epoch = 0

    while config.step < conf["total_steps"]:
        
        # Do training
        train_loss, train_accuracy = train(train_loader, model, criterion, optimizer, scheduler, device, model_writers)
        train_writer(config.step, train_loss, train_accuracy, scheduler.get_last_lr()[0])
        
        # Do validation		
        valid_loss, valid_accuracy = valid(valid_loader, model, criterion, device)
        valid_writer(config.step, valid_loss, valid_accuracy)

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_loss = valid_loss
            best_state_dict = {
                'config': conf,
                'step': config.step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            
        # Save the best model so far.
        if (config.step) % conf["save_steps"] < len(train_loader) and best_state_dict is not None:
            torch.save(best_state_dict, f'{writer.log_dir}.ckpt')
            print(f"Step {config.step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

        # Early stop
        if config.step - best_state_dict['step'] > conf["early_stop"]:
            print(f"Eearly stop at step {config.step + 1} (best model saved at step {best_state_dict['step']}, accuracy={best_accuracy:.4f})")
            break

    writer.add_hparams(
        { "encoder.layers": 3 },
        { "accuracy": best_accuracy, "loss": best_loss }
    )
    writer.close()

if __name__ == "__main__":
    from conf_loader import conf, get_default
    
    main(get_default(conf))
