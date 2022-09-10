import sys
import time
import datetime
import shutil
import torch

from pathlib import Path
from utils import load_yaml, eval_config
from torch.utils.tensorboard import SummaryWriter
from core.metric.evaluate import calculate_test_loss, calculate_test_accuracy, calculate_embeddings


def train_epoch(model, optim, loss_func, dataloader, device, epoch, writer):
    start_timer = time.time()
    model.train()

    running_loss = 0.0
    running_fraction_positive_triplets = 0.0

    train_loss = 0.0
    train_fraction_positive_triplets = 0.0

    n_iters = 1
    for _, (images, targets, image_paths) in enumerate(train_loader):
        optim.zero_grad()

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        embeddings = model(images)
        loss, fraction_positive_triplets = loss_func(embeddings, targets)

        loss.backward()
        optim.step()

        running_loss += loss.item()
        train_loss += loss.item()
        running_fraction_positive_triplets += float(fraction_positive_triplets)
        train_fraction_positive_triplets += float(fraction_positive_triplets)

        n_iters += 1

    # Logging to tensorboard
    average_loss: float = train_loss / n_iters
    average_positive_triplets: float = 100 * train_fraction_positive_triplets / n_iters

    print(
        "Epoch: {:d} | Time: {:f} | Loss: {:f} | Avg hard triplets: {:.2f}%".format(
            epoch, time.time() - start_timer, average_loss, average_positive_triplets
        )
    )

    writer.add_scalar("train/loss", average_loss, epoch)
    writer.add_scalar(
        "train/average_positive_triplets", average_positive_triplets, epoch
    )


if __name__ == "__main__":
    config_path = sys.argv[1]

    # load config
    config = load_yaml(config_path)
    config = eval_config(config)

    # init checkpoint dir and copy config file to this folder
    checkpoint_dir = Path(
        config["trainer"]["checkpoint_dir"],
        datetime.datetime.now().strftime("%Y%d%m_%H%M%S"),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, checkpoint_dir)

    # device
    device = config["trainer"]["device"]

    # data loader
    train_loader = config["data"]["train"]
    valid_loader = config["data"]["valid"]

    # optimizer
    optim = config["optim"]

    # loss
    train_loss_func = config["loss"]["train_loss"]
    valid_loss_func = config["loss"]["valid_loss"]

    # model and resume
    model = config["model"]
    model.to(device)
    resume_path = config["trainer"]["resume_checkpoint_path"]
    if resume_path != "":
        if Path(resume_path).exists():
            checkpoint = torch.load(resume_path, map_location=device)
            weight = checkpoint["model_state_dict"]
            model.load_state_dict(weight)
            print("Resumed checkpoint!")
        else:
            raise ValueError(
                "Not found resume check point in {} folder!".format(resume_path)
            )

    # init tensorboard and add graph
    writer = SummaryWriter(log_dir=checkpoint_dir)
    # add graph for visualization
    with torch.no_grad():
        dummy_input = torch.zeros(
            1,
            train_loader.dataset.num_channel,
            train_loader.dataset.image_size[0],
            train_loader.dataset.image_size[1],
        ).to(device)
        writer.add_graph(model, dummy_input, verbose=False)

    # training
    print("Training...")
    best_acc, best_loss, patient = -1.0, 100, 0
    for epoch in range(config["trainer"]["max_epoch"]):
        train_epoch(
            model=model,
            optim=optim,
            loss_func=train_loss_func,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            writer=writer,
        )
        if epoch % config["trainer"]["valid_frequency"] == 0:
            print("=" * 100)
            print("Evaluating...")
            start_timer = time.time()
            valid_loss = calculate_test_loss(
                model=model,
                test_loader=valid_loader,
                loss_function=valid_loss_func,
                epoch=epoch,
                device=device,
                writer=writer,
            )
            valid_acc = calculate_test_accuracy(
                model=model,
                test_loader=valid_loader,
                train_loader=train_loader,
                epoch=epoch,
                device=device,
                writer=writer,
            )
            print(
                "Valid Loss: {:f} | Time: {:f} | Valid Accuracy: {:.2f}%".format(
                valid_loss, time.time() - start_timer, valid_acc)
            )
            # save best model
            if valid_acc > best_acc or (
                valid_acc == best_acc and best_loss > valid_loss
            ):
                best_acc = valid_acc
                patient = 0
                best_loss = valid_loss
                # Save checkpoint
                # checkpoint_name = "e{}_loss{:1.5f}_acc{:1.5f}.pt".format(
                #     epoch, valid_loss, valid_acc
                # )
                checkpoint_name = "best.pt"
                checkpoint_pth = checkpoint_dir.joinpath(checkpoint_name)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "valid_loss": valid_loss,
                        "valid_acc": valid_acc,
                    },
                    checkpoint_pth,
                )
            else:
                patient += 1
        if patient > config["trainer"]["early_stoping"]:
            print(
                f'Model converge after {epoch - config["trainer"]["early_stoping"]*config["trainer"]["valid_frequency"]} epoch. Best accuracy: {best_acc}. Best loss: {best_loss}'
            )
            break

    print("Calculate all embeddings for visualization")
    # Calculating embedding of training set for visualization
    train_loader.sampler.sequential_sampling = True
    embeddings, labels, _ = calculate_embeddings(model, train_loader, device)
    name_labels = [train_loader.dataset.idx_to_class[idx_class.item()] for idx_class in labels]
    writer.add_embedding(embeddings, metadata=name_labels, tag="train")

    # Calculating embedding of testing set for visualization
    valid_loader.dataset.return_triplets = False
    embeddings, labels, _ = calculate_embeddings(model, valid_loader, device)
    name_labels = [valid_loader.dataset.idx_to_class[idx_class.item()] for idx_class in labels]
    writer.add_embedding(embeddings, metadata=name_labels, tag="test")

    print("Everything is done!")
