import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from acm.optimiser import ACM
from experiments.utils import train_and_evaluate, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def main():
    set_seed(42)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Corrupt 40% of labels
    print("Preparing noisy dataset...")
    noise_level = 0.4
    num_noisy = int(noise_level * len(train_set))
    noisy_indices = random.sample(range(len(train_set)), num_noisy)
    for idx in noisy_indices:
        train_set.targets[idx] = random.randint(0, 9)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    print("\nTraining with Adam...")
    model_adam = SimpleCNN().to(device)
    opt_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)
    _, _, _, hist_adam = train_and_evaluate(model_adam, train_loader, test_loader, criterion, opt_adam, epochs=30, patience=30)

    print("\nTraining with ACM...")
    model_acm = SimpleCNN().to(device)
    opt_acm = ACM(model_acm.parameters(), lr=0.01, kappa=1.0)
    _, _, _, hist_acm = train_and_evaluate(model_acm, train_loader, test_loader, criterion, opt_acm, epochs=30, patience=30)

    # Plotting the results for the paper
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 31), hist_adam['val_acc'], label='Adam', color='cyan', linewidth=2, marker='o')
    plt.plot(range(1, 31), hist_acm['val_acc'], label='ACM (Ours)', color='black', linewidth=2, linestyle='--', marker='s')

    plt.title('Validation Accuracy on FashionMNIST (40% Noisy Labels)', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig('fashionmnist.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print("Saved fashionmnist.pdf")

if __name__ == '__main__':
    main()
