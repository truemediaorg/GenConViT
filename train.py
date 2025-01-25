import sys, os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from time import perf_counter
import pickle
from model.config import load_config
from model.genconvit_ed import GenConViTED
from model.genconvit_vae import GenConViTVAE
from dataset.loader import load_data, load_checkpoint
import optparse
import copy
import matplotlib.pyplot as plt


config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained(model,optimizer, pretrained_model_filename):
    
    assert os.path.isfile(
        pretrained_model_filename
    ), "Saved model file does not exist. Exiting."

    new_model, new_optimizer, start_epoch, min_loss = load_checkpoint(
        model, optimizer, filename=pretrained_model_filename
    )
    # now individually transfer the optimizer parts...
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return new_model, new_optimizer, start_epoch, min_loss


def load_pretrained(model, pretrained_model_filename):
    
    assert os.path.isfile(
        pretrained_model_filename
    ), "Saved model file does not exist. Exiting."

    new_model, _, _, _ = load_checkpoint(
        model, filename=pretrained_model_filename
    )
    return new_model, _, _, _


def train_model(
    dir_path, mod, num_epochs, pretrained_model_filename, test_model, batch_size
):
    print("Loading data...")
    dataloaders, dataset_sizes = load_data(dir_path, batch_size)
    print("Done.")

    if mod == "ed":
        from train.train_ed import train, valid
        model = GenConViTED(config)
    else:
        from train.train_vae import train, valid
        model = GenConViTVAE(config)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    mse = nn.MSELoss()
    min_val_loss = int(config["min_val_loss"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    if pretrained_model_filename:
        model, _, _, _ = load_pretrained(
            model, pretrained_model_filename
        )

    model.to(device)
    torch.manual_seed(1)
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    since = time.time()

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_accs, valid_accs = [], []
    for epoch in range(0, num_epochs):
        train_loss, train_acc, epoch_loss = train(
            model,
            device,
            dataloaders["train"],
            criterion,
            optimizer,
            epoch,
            train_loss,
            train_acc,
            mse,
        )
        valid_loss, valid_acc = valid(
            model,
            device,
            dataloaders["validation"],
            criterion,
            epoch,
            valid_loss,
            valid_acc,
            mse,
        )
        scheduler.step()


        # Check if the current epoch's accuracy is the best we've seen so far
        if valid_acc[-1] >= best_acc:
            best_acc = valid_acc[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        train_accs.append(train_acc[-1])
        valid_accs.append(valid_acc[-1])
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

  
    print("\nSaving model...\n")
    file_path = os.path.join(
        "weight",
        f'genconvit_{mod}_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}',
    )
    
    # saving figures 
    save_figures(train_loss, valid_loss, mod)
    save_acc_figures(train_accs, valid_accs, mod)
    print("\nSaving model...\n")
    file_path = os.path.join(
        "weight",
        f'genconvit_{mod}_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}',
    )

    with open(f"{file_path}.pkl", "wb") as f:
        pickle.dump([train_loss, train_acc, valid_loss, valid_acc], f)

    state = {
        "epoch": num_epochs + 1,
        "state_dict": best_model_wts,
        "optimizer": optimizer.state_dict(),
        "min_loss": epoch_loss,
    }

    weight = f"{file_path}.pth"
    state_dict_only = f"{file_path}_acc_{best_acc}_inference.pth"
    torch.save(state, weight)
    torch.save(model.state_dict(), state_dict_only)

    print("Done.")

    if test_model:
        model.load_state_dict(best_model_wts)  # Load the best model weights
        test(model, dataloaders, dataset_sizes, mod, weight)

def save_acc_figures(train_accs, valid_accs, mod):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(valid_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Ensure the directory exists
    results_dir = "result/figs"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the plot
    plot_filename = os.path.join(results_dir, f'acc_trend_{mod}_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    print(f"Plot saved as {plot_filename}")

def save_figures(train_epoch_losses, valid_epoch_losses, mod):
    plt.figure(figsize=(10, 5))
    plt.plot(train_epoch_losses, label='Training Loss')
    plt.plot(valid_epoch_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend()

    # Ensure the directory exists
    results_dir = "result/figs"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the plot
    plot_filename = os.path.join(results_dir, f'loss_trend_{mod}_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    print(f"Plot saved as {plot_filename}")


def test(model, dataloaders, dataset_sizes, mod, weight):
    print("\nRunning test...\n")
    model.eval()
    checkpoint = torch.load(weight, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    _ = model.eval()

    Sum = 0
    counter = 0
    for inputs, labels in dataloaders["test"]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if mod == "ed":
            output = model(inputs).to(device).float()
        else:
            output = model(inputs)[0].to(device).float()

        _, prediction = torch.max(output, 1)

        pred_label = labels[prediction]
        pred_label = pred_label.detach().cpu().numpy()
        main_label = labels.detach().cpu().numpy()
        bool_list = list(map(lambda x, y: x == y, pred_label, main_label))
        Sum += sum(np.array(bool_list) * 1)
        counter += 1
        print(f"Pediction: {Sum}/{len(inputs)*counter}")

    print(
        f'Prediction: {Sum}/{dataset_sizes["test"]} {(Sum / dataset_sizes["test"]) * 100:.2f}%'
    )


def gen_parser():
    parser = optparse.OptionParser("Train GenConViT model.")
    parser.add_option(
        "-e",
        "--epoch",
        type=int,
        dest="epoch",
        help="Number of epochs used for training the GenConvNextViT model.",
    )
    parser.add_option("-v", "--version", dest="version", help="Version 0.1.")
    parser.add_option("-d", "--dir", dest="dir", help="Training data path.")
    parser.add_option(
        "-m",
        "--model",
        dest="model",
        help="model ed or model vae, model variant: genconvit (A) ed or genconvit (B) vae.",
    )
    parser.add_option(
        "-p",
        "--pretrained",
        dest="pretrained",
        help="Saved model file name. If you want to continue from the previous trained model.",
    )
    parser.add_option("-t", "--test", dest="test", help="run test on test dataset.")
    parser.add_option("-b", "--batch_size", dest="batch_size", help="batch size.")

    (options, _) = parser.parse_args()

    dir_path = options.dir
    epoch = options.epoch
    mod = "ed" if options.model == "ed" else "vae"
    test_model = "y" if options.test else None
    pretrained_model_filename = options.pretrained if options.pretrained else None
    batch_size = options.batch_size if options.batch_size else config["batch_size"]

    return dir_path, mod, epoch, pretrained_model_filename, test_model, int(batch_size)


def main():
    start_time = perf_counter()
    path, mod, epoch, pretrained_model_filename, test_model, batch_size = gen_parser()
    train_model(path, mod, epoch, pretrained_model_filename, test_model, batch_size)
    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()
