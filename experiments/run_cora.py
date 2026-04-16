import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from acm.optimiser import ACM
from experiments.utils import set_seed


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        # Standard 2-layer GCN
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Training Loop for Node Classification
def train_gcn(optimizer_class, data, num_features, num_classes, epochs=200, **opt_kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features, num_classes).to(device)
    data = data.to(device)
    optimizer = optimizer_class(model.parameters(), **opt_kwargs)

    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        # GCNs calculate loss only on the specific training nodes
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation Phase
        model.eval()
        with torch.no_grad():
            pred = model(data).argmax(dim=1)
            correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
            acc = int(correct) / int(data.val_mask.sum())
            val_accuracies.append(acc * 100)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Val Acc: {val_accuracies[-1]:.2f}%')

    return val_accuracies


def main():
    set_seed(42)

    print("Downloading Cora Dataset...")
    # Fix the generic root path from /tmp to standard local folder './data/Cora'
    cora_dataset = Planetoid(root='./data/Cora', name='Cora')
    cora_data = cora_dataset[0]

    print("=== Training GCN ===\n")
    print("Training with Adam...")
    cora_adam = train_gcn(torch.optim.Adam, cora_data, cora_dataset.num_node_features, cora_dataset.num_classes,
                          lr=0.01)

    print("\nTraining with ACM...")
    cora_acm = train_gcn(ACM, cora_data, cora_dataset.num_node_features, cora_dataset.num_classes, lr=0.1, kappa=2.0)

    # Plotting the Results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 201), cora_adam, label='Adam', color='cyan', alpha=0.7)
    plt.plot(range(1, 201), cora_acm, label='ACM (Ours)', color='black', linewidth=2)

    plt.title('Validation Accuracy on Cora Citation Graph (Node Classification)', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig('cora.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print("Saved cora.pdf")


if __name__ == '__main__':
    main()
