import copy
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# STRICT REPRODUCIBILITY FUNCTION
# ==========================================
def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    This ensures identical random states across PyTorch, NumPy, and Python's
    built-in hashing.

    Args:
        seed (int, optional): The random seed to use (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10, patience=3):
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'epoch_times': []}

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        if use_cuda:
            starter.record()
        else:
            start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if use_cuda:
            ender.record()
            torch.cuda.synchronize()
            epoch_time = starter.elapsed_time(ender) / 1000.0
        else:
            epoch_time = time.time() - start_time

        history['epoch_times'].append(epoch_time)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        if scheduler: scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1:02d}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc * 100:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if use_cuda else 0
    avg_epoch_time = np.mean(history['epoch_times'])
    if best_model_state: model.load_state_dict(best_model_state)

    return best_val_acc, avg_epoch_time, peak_memory, history


def train_model_v2(model, criterion, optimizer, train_loader, val_loader, scheduler=None, epochs=10):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    epoch_times = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        if torch.cuda.is_available():
            starter.record()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

        if torch.cuda.is_available():
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            epoch_times.append(curr_time)

    avg_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    print(f'\n--- Compute Metrics ---')
    print(f'Average Epoch Time: {avg_time:.2f} seconds')
    print(f'Peak GPU Memory: {peak_mem:.2f} MB')

    return model, history


def evaluate_and_plot(model, dataloader, history, class_names, model_name):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n--- {model_name.upper()} Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (ResNet18 + {model_name.upper()})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Plot Training/Validation Curves
    plt.subplot(1, 2, 2)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_cm.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
