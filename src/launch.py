from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from models import Simple3DCNN, Simple3DCNNBN, Simple3DCNNBN_Modular, CNN3DWithAttention, MODELS
from dataloader import CTScansDataset, SliceDataset, CLASS_TRAIN_COUNTS, DATASETS, TRANSFORMS, ROOT_DIRS
import wandb
import os
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from utils import load_config
import yaml
from tqdm import tqdm
import csv
import pandas as pd
import numpy as np

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run the 3D CNN model with specified configuration")
parser.add_argument('--config', type=str, help='Path to the YAML configuration file', required=True)
parser.add_argument('--force', action='store_true', help='Force overwrite of output directory')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate for training')
parser.add_argument('--classes_to_include', type=int, nargs='+', default=None, help='Classes to include in the training')
parser.add_argument('--class_mapping', type=str, default=None, help='Class Mapping')
parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs for training')
parser.add_argument('--exp_name', type=str, default=None, help='Name of the experiment')
parser.add_argument('--no_wandb', action='store_true', help='Disable logging to Weights & Biases')
parser.add_argument('--weighted', action='store_true', help='Use weighted sampling')
parser.add_argument('--no_transform', action='store_true', help='Skip Transforms, Just Overfit')
parser.add_argument('--model', type=str, help='Model to use')
parser.add_argument('--overfit', action='store_true', help='Overfit to a single batch')
parser.add_argument('--dataset_id', type=str, help='Dataset to use')
parser.add_argument('--dropout', type=float, help='Dropout')
parser.add_argument('--relaunch', action='store_true', help='Relaunch the experiment')
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint to load')

args = parser.parse_args()
    
# Load from the YAML
config_path = args.config
config = load_config(config_path)
config['batch_size'] = args.batch_size if args.batch_size is not None else config.get('batch_size', 16)  # Default value if not in YAML
config['learning_rate'] = args.learning_rate if args.learning_rate is not None else config.get('learning_rate', 0.001)  # Default value
config['num_epochs'] = args.num_epochs if args.num_epochs is not None else config.get('num_epochs', 400)  # Default value
config['exp_name'] = args.exp_name if args.exp_name is not None else config.get('exp_name', '3d-cnn')  # Default value
config['classes_to_include'] = args.classes_to_include if args.classes_to_include is not None else config.get('classes_to_include', [0, 1, 2, 3, 4])  # Default value
config['class_mapping'] = args.class_mapping if args.class_mapping is not None else config.get('class_mapping', None)  # Default value
config['weighted'] = args.weighted if args.weighted is not None else config.get('weighted', False)  # Default value
config['no_transform'] = args.no_transform if args.no_transform is not None else config.get('no_transform', False)  # Default value
config['model'] = args.model if args.model is not None else config.get('model', 'simple_modular')  # Default value
config['overfit'] = args.overfit if args.overfit is not None else config.get('overfit', False)  # Default value
config['dataset_id'] = args.dataset_id if args.dataset_id is not None else config.get('dataset_id', '3d')  # Default value
config['dropout'] = args.dropout if args.dropout is not None else config.get('dropout', 0)  # Default value
config['relaunch'] = args.relaunch if args.relaunch is not None else config.get('relaunch', False)  # Default value
config['checkpoint'] = args.checkpoint if args.checkpoint is not None else config.get('checkpoint', None)  # Default value

# Hyperparams
batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
exp_name = config['exp_name']
classes_to_include = config['classes_to_include']
class_mapping = config['class_mapping']
weighted = config['weighted']
model_name = config['model']
overfit = config['overfit']
dataset_id = config['dataset_id']
dropout = config['dropout']
relaunch = config['relaunch']
checkpoint = config['checkpoint']

# Set up the output directory
num_classes = len(classes_to_include) if class_mapping is None else len(set(class_mapping.values()))
exp_name = f'{exp_name}_{model_name}_lr_{learning_rate}_epochs_{num_epochs}_bs_{batch_size}__weighted_{weighted}_classes-{"_".join([str(c) for c in classes_to_include])}'
output_dir = f'outputs/{exp_name}'

# See if folder already exists
if os.path.exists(output_dir) and not args.force:
    raise ValueError("Output directory already exists. Please specify a new output directory.")
else:
    os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'used_config.yaml'), 'w') as file:
    yaml.dump(config, file)

# Set up general settings
if not args.no_wandb:
    wandb.init(project="3d-ct-scans")
    wandb.run.name = exp_name
    wandb.run.save()

# Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = DATASETS[dataset_id]
root_dir = ROOT_DIRS[dataset_id]
my_transform = TRANSFORMS[dataset_id]

# Transforms
if args.no_transform:
    train_transforms = None
    val_transforms = None
else:
    train_transforms =my_transform.get_train_transforms()
    val_transforms = my_transform.get_val_transforms()

# Datasets and DataLoaders
train_dataset = dataset(root_dir=root_dir, split='train', transform=train_transforms, include_classes=classes_to_include, class_mapping=class_mapping)
val_dataset = dataset(root_dir=root_dir, split='val', transform=val_transforms, include_classes=classes_to_include, class_mapping=class_mapping)

if args.overfit:
    train_size = 500
    val_size = 50
    train_dataset = torch.utils.data.Subset(train_dataset, list(range(train_size)) + list(range(-train_size, 0)))
    val_dataset = torch.utils.data.Subset(val_dataset, list(range(val_size)) + list(range(-val_size, 0)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
else:
    if args.weighted:
        class_counts = CLASS_TRAIN_COUNTS
        class_counts = {k: v for k, v in class_counts.items() if k in classes_to_include}
        if class_mapping:
            totals = defaultdict(int)
            for k, v in class_counts.items():
                totals[class_mapping[k]] += v
            class_counts = dict(totals)
        num_samples = sum(class_counts.values()) * 64
        labels = np.array([label for _, label in train_dataset])
        weights = [num_samples/class_counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss Function, Optimizer
model = MODELS[model_name](num_classes=num_classes, dropout=dropout)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Metrics
task = "multiclass"
train_accuracy_metric_macro = torchmetrics.Accuracy(task, num_classes=num_classes, average='macro').to(device)
train_accuracy_metric_micro = torchmetrics.Accuracy(task, num_classes=num_classes, average='micro').to(device)
accuracy_metric_macro = torchmetrics.Accuracy(task, num_classes=num_classes, average='macro').to(device)
accuracy_metric_micro = torchmetrics.Accuracy(task, num_classes=num_classes, average='micro').to(device)
confmat_metric = torchmetrics.ConfusionMatrix(task, num_classes=num_classes).to(device)
best_val_loss = float('inf')

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total = 0
    correct = 0

    predictions_file = open(f'{output_dir}/predictions_epoch_{epoch+1}.csv', 'w', newline='')
    predictions_writer = csv.writer(predictions_file)
    predictions_writer.writerow(['True Label', 'Predicted Label'])

    train_accuracy_metric_macro.reset()
    train_accuracy_metric_micro.reset()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_accuracy_metric_macro.update(outputs.argmax(dim=1), labels)
        train_accuracy_metric_micro.update(outputs.argmax(dim=1), labels)

    # Validation Loop
    model.eval()
    total_val_loss = 0
    accuracy_metric_macro.reset()
    accuracy_metric_micro.reset()
    confmat_metric.reset()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            accuracy_metric_macro.update(outputs.argmax(dim=1), labels)
            accuracy_metric_micro.update(outputs.argmax(dim=1), labels)
            confmat_metric.update(outputs.argmax(dim=1), labels)

            for label, pred in zip(labels.tolist(), outputs.argmax(dim=1).tolist()):
                predictions_writer.writerow([label, pred])
    
    # Log Metrics
    train_accuracy_macro = train_accuracy_metric_macro.compute()
    train_accuracy_micro = train_accuracy_metric_micro.compute()
    val_accuracy_macro = accuracy_metric_macro.compute()
    val_accuracy_micro = accuracy_metric_micro.compute()
    confusion_matrix = confmat_metric.compute()

    scheduler.step(total_val_loss)

    # Print Epoch Statistics
    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc (Macro): {train_accuracy_macro:.4f}, Train Acc (Micro): {train_accuracy_micro:.4f}, Val Acc (Macro): {val_accuracy_macro:.4f}, Val Acc (Micro): {val_accuracy_micro:.4f}')
    confusion_matrix = pd.DataFrame(confusion_matrix.cpu().numpy(), columns=[f'pred_{i}' for i in range(num_classes)], index=[f'label_{i}' for i in range(num_classes)])
    # Log to W&B
    if not args.no_wandb:
        wandb.log({
            "epoch": epoch, 
            "val_loss": avg_val_loss, 
            "train_loss": avg_train_loss,
            "train_accuracy_macro": train_accuracy_macro,
            "train_accuracy_micro": train_accuracy_micro,
            "val_accuracy_macro": val_accuracy_macro,
            "val_accuracy_micro": val_accuracy_micro,
            "confusion_matrix": wandb.Table(dataframe=confusion_matrix),
        })

    if epoch % 20 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'best_val_loss': best_val_loss,
        }, f'{output_dir}/checkpoint_epoch_{epoch}.pth')

    # Save Best Model Checkpoint
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        torch.save(model.state_dict(), f'{output_dir}/best_model.pth')

# Save the Model
torch.save(model.state_dict(), f'{output_dir}/model.pth')
