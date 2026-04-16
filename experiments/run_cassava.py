import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

from acm.optimiser import ACM
from experiments.utils import train_and_evaluate, train_model_v2, evaluate_and_plot, set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CassavaDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.dataframe.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label


def get_resnet_model(num_classes=5):
    # Load pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze early layers to speed up training (optional, but recommended for low compute)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer for our 5 Cassava classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model.to(device)


def plot_optimiser_comparison(history_1, label_1, history_2, label_2):
    epochs = range(1, len(history_1['train_loss']) + 1)

    plt.figure(figsize=(14, 6))

    # Subplot 1: Validation Loss Comparison
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_1['val_loss'], label=f'{label_1}', linestyle='-', marker='o', linewidth=2)
    plt.plot(epochs, history_2['val_loss'], label=f'{label_2}', linestyle='--', marker='s', linewidth=2)
    plt.title('Validation Loss Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)

    # Subplot 2: Validation Accuracy Comparison
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_1['val_acc'], label=f'{label_1}', linestyle='-', marker='o', linewidth=2)
    plt.plot(epochs, history_2['val_acc'], label=f'{label_2}', linestyle='--', marker='s', linewidth=2)
    plt.title('Validation Accuracy Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.suptitle('Optimiser Benchmark: Custom ACM (LR=0.005) vs. Standard Adam (LR=0.001)', fontsize=16,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig('optimiser_benchmark.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print("Saved optimiser_benchmark.pdf")
    plt.show()


def main():
    set_seed(42)

    # Data download
    if not os.path.exists('raw_kaggle_data'):
        print("Downloading Cassava dataset... This relies on the kaggle CLI tool being configured on your system.")
        os.system("kaggle competitions download -c cassava-leaf-disease-classification")
        os.system("unzip -q cassava-leaf-disease-classification.zip -d raw_kaggle_data")

    # If it still isn't retrieved, exit gracefully
    if not os.path.exists('raw_kaggle_data/train.csv'):
        print("Kaggle data download failed or unzipping failed. Skipping execution.")
        return

    df = pd.read_csv('raw_kaggle_data/train.csv')
    train_df, val_df = torch.utils.data.random_split(df, [int(len(df) * 0.8), len(df) - int(len(df) * 0.8)])
    train_df, val_df = df.iloc[train_df.indices], df.iloc[val_df.indices]

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CassavaDataset(train_df, 'raw_kaggle_data/train_images', transform_train)
    val_dataset = CassavaDataset(val_df, 'raw_kaggle_data/train_images', transform_val)

    # Make sure to handle Windows threading by setting num_workers=0 or wrapping in __main__
    # We are in __main__ so num_workers=2 is safe if it works, default to 0 on Windows if problems arise, keeping 2.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Cassava Data Ready.")

    # Cassava Classes
    cassava_classes = ['Cassava Bacterial Blight', 'Cassava Brown Streak Disease',
                       'Cassava Green Mottle', 'Cassava Mosaic Disease', 'Healthy']

    # 1. ACM Training
    print("Training ResNet18 with Tuned ACM Optimiser...")
    model_acm = get_resnet_model(num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer_acm = ACM(model_acm.parameters(), lr=0.005, kappa=1.0)
    scheduler_acm = torch.optim.lr_scheduler.StepLR(optimizer_acm, step_size=5, gamma=0.5)

    model_acm, history_acm = train_model_v2(model_acm, criterion, optimizer_acm, train_loader, val_loader,
                                            scheduler=scheduler_acm, epochs=10)
    evaluate_and_plot(model_acm, val_loader, history_acm, cassava_classes, 'acm')

    # 2. Adam Baseline
    print("Initialising baseline ResNet18 with Adam optimiser...")
    model_adam = get_resnet_model(num_classes=5)
    optimizer_adam = torch.optim.Adam(model_adam.fc.parameters(), lr=0.001)

    print("Training ResNet18 with standard Adam...")
    model_adam, history_adam = train_model_v2(model_adam, criterion, optimizer_adam, train_loader, val_loader,
                                              epochs=10)

    # 3. Compare Both models
    print("Generating benchmark comparison plot...")
    plot_optimiser_comparison(history_acm, 'Custom ACM (LR=0.005)', history_adam, 'Adam Baseline (LR=0.001)')

    print("\n--- STARTING PHASE 3: ABLATION STUDY (Kappa) ---")
    kappas = [0.1, 1.0, 5.0, 10.0]
    ablation_results = {}

    set_seed(42)
    acm_optimal_lr = 0.005  # Extracted from the default found in demo

    for k in kappas:
        print(f"Testing ACM with kappa = {k}...", end="")
        model = get_resnet_model(num_classes=5)
        optimizer = ACM(model.parameters(), lr=acm_optimal_lr, kappa=k)
        criterion = nn.CrossEntropyLoss()

        val_acc, _, _, _ = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs=8)
        ablation_results[f'k={k}'] = val_acc * 100
        print(f" Acc: {val_acc * 100:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.bar(list(ablation_results.keys()), list(ablation_results.values()), color='darkslategray')
    plt.ylim(min(ablation_results.values()) - 2, max(ablation_results.values()) + 1)
    plt.title("Ablation Study: Impact of Manifold Sensitivity ($\kappa$)", fontsize=14)
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('ablation_kappa.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print("Saved ablation_kappa.pdf")


if __name__ == '__main__':
    main()
