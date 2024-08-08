import matplotlib.pyplot as plt
import os
import numpy as np

config_path = "conf"

def load_training_log(config_folder_dir):
    log_dir = os.path.join(config_folder_dir, f"running/")

    train_losses = []
    train_losses_dir = os.path.join(log_dir, f"train_losses.txt")
    test_losses = []
    test_losses_dir = os.path.join(log_dir, f"test_losses.txt")
    test_accuracies = []
    test_accuracies_dir = os.path.join(log_dir, f"test_accuracies.txt")
    test_wers = []
    test_wers_dir = os.path.join(log_dir, f"test_wers.txt")
    with open(train_losses_dir, 'r') as tl, open(test_losses_dir, 'r') as tl2, open(test_accuracies_dir, 'r') as ta, open(test_wers_dir, 'r') as tw:
        train_losses = tl.read().split(";")
        train_losses = train_losses[:-1]
        train_losses = np.array(train_losses,dtype=float)

        test_losses = tl2.read().split(";")
        test_losses = test_losses[:-1]
        test_losses = np.array(test_losses,dtype=float)

        test_accuracies = ta.read().split(";")
        test_accuracies = test_accuracies[:-1]
        test_accuracies = np.array(test_accuracies,dtype=float)

        test_wers = tw.read().split(";")
        test_wers = test_wers[:-1]
        test_wers = np.array(test_wers,dtype=float)

    return train_losses, test_losses, test_accuracies, test_wers

def plot_metrics(train_losses, test_losses, test_accuracies, test_wers, config_folder_dir):
    epochs = range(1, len(train_losses) + 1)
    last_epoch = epochs[-1]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss') 
    plt.title('Training Loss')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, test_wers, label='Test WER')
    plt.xlabel('Epochs')
    plt.ylabel('WER')
    plt.title('Test WER')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    
    save_dir = os.path.join(config_folder_dir, f"running/")
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f'training_metrics_{last_epoch}.svg'))
    plt.savefig(os.path.join(save_dir, f'training_metrics_{last_epoch}.pdf'))
    plt.show()

train_losses, test_losses, test_accuracies, test_wers = load_training_log(config_path)
plot_metrics(train_losses, test_losses, test_accuracies, test_wers, config_path)