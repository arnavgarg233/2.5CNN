import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from dataloader import DATASETS, TRANSFORMS, ROOT_DIRS
from models_half import MODELS
from utils import load_config
from dataloader_old import CTScansDataset
import torchmetrics
import csv
import numpy as np
from torchmetrics import Accuracy, F1Score, Specificity, Recall, ConfusionMatrix
import seaborn as sns
import matplotlib.pyplot as plt


def load_checkpoint(model, checkpoint_path):
    """
    Load model weights from a checkpoint file.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    return model

def load_checkpoint_for_resnet(model, checkpoint_path):
    """
    Load model weights from a checkpoint file and map them correctly to TwoStageModel.
    Then freeze the loaded parameters.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Create a new state dict for mapping
    new_state_dict = {}
    
    # Map the keys from checkpoint to TwoStageModel format
    for key, value in checkpoint.items():
        if key in ['custom_layers.6.weight', 'custom_layers.6.bias']:
            continue
        new_key = f"resnet.{key}"
        new_state_dict[new_key] = value
    
    # Load the mapped state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    # Freeze all parameters in the resnet
    for name, param in model.named_parameters():
        if name.startswith('resnet.'):
            # if key in ['custom_layers.6.weight', 'custom_layers.6.bias']:
            #     continue
            param.requires_grad = False
            print(f"Freezing parameter: {name}")
    
    return model

# def evaluate_3d(model, val_loader, num_classes, device, ckpt_path):
#     """
#     Evaluate the model on the validation set.
#     """
#     model.eval()  # Set the model to evaluation mode
#     task = "multiclass"
#     accuracy_metric_macro = torchmetrics.Accuracy(task, num_classes=num_classes, average='macro').to(device)
#     accuracy_metric_micro = torchmetrics.Accuracy(task, num_classes=num_classes, average='micro').to(device)
#     confmat_metric = torchmetrics.ConfusionMatrix(task, num_classes=num_classes).to(device)

#     output_dir = os.path.join(ckpt_path, 'eval/')
#     os.makedirs(output_dir, exist_ok=True)
#     predictions_file = open(f'{output_dir}/predictions.csv', 'w', newline='')
#     predictions_writer = csv.writer(predictions_file)
#     predictions_writer.writerow(['True Label', 'Predicted Label'])

#     accuracy_metric_macro.reset()
#     accuracy_metric_micro.reset()
#     confmat_metric.reset()

#     full_predictions = []
#     full_labels = []
#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc=f"Evaluating..."):
#             images, labels = batch
#             print(images.shape)
#             images, labels = images.to(device), labels.to(device)
#             outputs = []
#             for slice in range(images.shape[4]):
#                 repeated_images = images[:, :, :, :, slice]
#                 output = model(repeated_images)
#                 outputs.append(output)

            
#             output_preds = torch.hstack([output.argmax(dim=1, keepdims=True) for output in outputs])
#             full_predictions.append(output_preds)
#             full_labels.append(labels.unsqueeze(1))
#             outputs = torch.hstack([output.argmax(dim=1, keepdims=True) for output in outputs]).sum(axis=1)
#             outputs = outputs > 30
#             outputs = outputs.int()
#             for label, pred in zip(labels.tolist(),outputs.tolist()):
#                 predictions_writer.writerow([label, pred])
#             accuracy_metric_macro.update(outputs, labels)
#             accuracy_metric_micro.update(outputs, labels)
#             confmat_metric.update(outputs, labels)
    
#     full_predictions = torch.vstack(full_predictions) 
#     full_labels = torch.vstack(full_labels)

#     # save full predictions and labels
#     full_predictions_numpy = full_predictions.cpu().numpy() 
#     full_labels_numpy = full_labels.cpu().numpy() 
#     breakpoint()
#     np.savetxt(os.path.join(ckpt_path, 'eval/all_predictions.csv'), full_predictions_numpy.astype(int), delimiter=',')
#     np.savetxt(os.path.join(ckpt_path, 'eval/all_labels.csv'), full_labels_numpy.astype(int), delimiter=',')


#     # Log Metrics
#     val_accuracy_macro = accuracy_metric_macro.compute()
#     val_accuracy_micro = accuracy_metric_micro.compute()
#     confusion_matrix = confmat_metric.compute()

#     print(f'Val Acc (Macro): {val_accuracy_macro:.4f}, Val Acc (Micro): {val_accuracy_micro:.4f}')
#     print(confusion_matrix)

# def evaluate_3d(model, val_loader, num_classes, device, ckpt_path):
#     """
#     Evaluate the model on the validation set.
#     """
#     model.eval()  # Set the model to evaluation mode
#     task = "multiclass"
#     output_dir = os.path.join(ckpt_path, 'eval/')
#     os.makedirs(output_dir, exist_ok=True)

#     full_predictions = []
#     full_labels = []
#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc=f"Evaluating..."):
#             images, labels = batch
#             print(images.shape)
#             images, labels = images.to(device), labels.to(device)
#             outputs = []
#             for slice in range(images.shape[4]):
#                 repeated_images = images[:, :, :, :, slice]
#                 output = model(repeated_images)
#                 outputs.append(output)

            
#             output_preds = torch.hstack([output.argmax(dim=1, keepdims=True) for output in outputs])
#             full_predictions.append(output_preds)
#             full_labels.append(labels.unsqueeze(1))
#             outputs = torch.hstack([output.argmax(dim=1, keepdims=True) for output in outputs]).sum(axis=1)
#             outputs = outputs > 30
#             outputs = outputs.int()
    
#     full_predictions = torch.vstack(full_predictions) 
#     full_labels = torch.vstack(full_labels)

#     # save full predictions and labels
#     full_predictions_numpy = full_predictions.cpu().numpy() 
#     full_labels_numpy = full_labels.cpu().numpy() 
#     np.savetxt(os.path.join(ckpt_path, 'eval/all_predictions.csv'), full_predictions_numpy.astype(int), delimiter=',')
#     np.savetxt(os.path.join(ckpt_path, 'eval/all_labels.csv'), full_labels_numpy.astype(int), delimiter=',')



# checkpoint_path = 'outputs/main_model/'
# root_dir = ROOT_DIRS['3d']
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_path = os.path.join(checkpoint_path, 'best_model.pth')
# config_path = os.path.join(checkpoint_path, 'used_config.yaml')
# config = load_config(config_path)
# classes_to_include = config['classes_to_include']
# class_mapping = config['class_mapping']
# num_classes = len(classes_to_include) if class_mapping is None else len(set(class_mapping.values()))

# batch_size = config['batch_size']
# model = config['model']
# model = MODELS[model](num_classes=num_classes).to(device)
# # val_dataset = DATASETS['3d'](root_dir=root_dir, split='val', transform=TRANSFORMS['3d'].get_val_transforms(), include_classes=classes_to_include, class_mapping=class_mapping)
# val_dataset = CTScansDataset(root_dir=root_dir, split='val', transform=TRANSFORMS['3d'].get_val_transforms(), include_classes=classes_to_include, class_mapping=class_mapping)

# # Load the model weights
# model = load_checkpoint(model, model_path)

# # Prepare the validation data
# val_size = 20
# # val_dataset = torch.utils.data.Subset(val_dataset, list(range(val_size)) + list(range(-val_size, 0)))
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# # Evaluate the model
# evaluate_3d(model, val_loader, num_classes, device, checkpoint_path)


class TwoStageModel(torch.nn.Module):
    """ 
    Two stage model that first runs a resnet on each slice of a 3d volume, then runs a resnet on the 2d slices.
    """
    def __init__(self):
        super(TwoStageModel, self).__init__()
        self.resnet = MODELS['custom_resnet'](num_classes=2)
        in_features = self.resnet.custom_layers[6].in_features
        self.out_features = 2
        self.resnet.custom_layers[6] = nn.Identity()
        
        
        # nn.Linear(in_features, self.out_features)

        self.ffwd = torch.nn.Sequential(
            torch.nn.Linear(in_features * 64, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(2048, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 2)
        )
        # self.identity = torch.nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape 
        outputs = []
        for slice in range(D): 
            repeated_images = x[:, :, :, :, slice] 
            slice_outputs = self.resnet(repeated_images) 
            outputs.append(slice_outputs)

        outputs = torch.hstack(outputs)
        out = self.ffwd(outputs)
        return out

checkpoint_path = 'outputs/main_model/'
root_dir = ROOT_DIRS['3d']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(checkpoint_path, 'best_model.pth')
config_path = os.path.join(checkpoint_path, 'used_config.yaml')
config = load_config(config_path)
classes_to_include = config['classes_to_include']
class_mapping = config['class_mapping']
num_classes = len(classes_to_include) if class_mapping is None else len(set(class_mapping.values()))
batch_size = config['batch_size'] 
model = config['model']
model = TwoStageModel().to(device)
train_dataset = DATASETS['3d'](root_dir=root_dir, split='train', transform=TRANSFORMS['3d'].get_train_transforms(), include_classes=classes_to_include, class_mapping=class_mapping)
val_dataset = DATASETS['3d'](root_dir=root_dir, split='val', transform=TRANSFORMS['3d'].get_val_transforms(), include_classes=classes_to_include, class_mapping=class_mapping)

model = load_checkpoint_for_resnet(model, model_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
correct = 0
total = 0

train = False
eval = True
epochs = 10
best_val_acc = 0

if train:
    # Initialize metrics
    metrics = {
        'accuracy_macro': Accuracy(task="multiclass", num_classes=2, average='macro').to(device),
        'accuracy_micro': Accuracy(task="multiclass", num_classes=2, average='micro').to(device),
        'f1': F1Score(task="multiclass", num_classes=2, average='macro').to(device),
        'sensitivity': Recall(task="multiclass", num_classes=2, average='macro').to(device),
        'specificity': Specificity(task="multiclass", num_classes=2, average='macro').to(device),
        'confusion_matrix': ConfusionMatrix(task="multiclass", num_classes=2).to(device)
    }
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs), desc=f"Training..."):
        # Reset metrics at start of epoch
        for metric in metrics.values():
            metric.reset()
            
        model.train()
        counter = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{10} (Training)"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            predictions = outputs.argmax(dim=1)
            for metric in metrics.values():
                metric.update(predictions, labels)
            
            counter += 1
            
        # Compute training metrics
        train_metrics = {name: metric.compute() for name, metric in metrics.items()}
        
        # Validation loop with same metrics
        model.eval()
        # Reset metrics for validation
        for metric in metrics.values():
            metric.reset()
            
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating..."):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                
                # Update metrics
                for metric in metrics.values():
                    metric.update(predictions, labels)
            
            # Compute validation metrics
            val_metrics = {name: metric.compute() for name, metric in metrics.items()}
            
            # Print metrics
            print(f"\nEpoch {epoch+1} Results:")
            print(f"Loss: {loss:.4f}")
            print("\nTraining Metrics:")
            for name, value in train_metrics.items():
                if name != 'confusion_matrix':
                    print(f"{name}: {value:.4f}")
            print("\nValidation Metrics:")
            for name, value in val_metrics.items():
                if name != 'confusion_matrix':
                    print(f"{name}: {value:.4f}")
            print("\nValidation Confusion Matrix:")
            print(val_metrics['confusion_matrix'])

            # Save best model based on macro F1 score
            if val_metrics['f1'] > best_val_acc:  # Changed from accuracy to F1
                best_val_acc = val_metrics['f1']
                save_path = os.path.join(checkpoint_path, 'best_two_stage_model.pth')
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

    # Save the trained model
    save_path = os.path.join(checkpoint_path, 'two_stage_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Then for evaluation, you can use a simpler load_checkpoint function:
def load_checkpoint_simple(model, checkpoint_path):
    """
    Simple function to load model weights from a checkpoint file.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)
    return model

def plot_confusion_matrix(confusion_matrix, save_path):
    """
    Create and save a pretty confusion matrix plot using seaborn with larger text
    """
    plt.figure(figsize=(10, 8))
    
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,          # Default font size
        'axes.titlesize': 16,     # Title size
        'axes.labelsize': 14,     # Axis label size
        'xtick.labelsize': 12,    # Tick label size
        'ytick.labelsize': 12     # Tick label size
    })
    
    # Create heatmap with larger annotation text
    sns.heatmap(confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar_kws={'label': 'Count'},
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={'size': 16})    # Size of numbers in cells
    
    # Add labels and title with specified font sizes
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    
    # Adjust layout and save with higher DPI for better quality
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if eval:
    # Initialize metrics for evaluation
    metrics = {
        'accuracy_macro': Accuracy(task="multiclass", num_classes=2, average='macro').to(device),
        'accuracy_micro': Accuracy(task="multiclass", num_classes=2, average='micro').to(device),
        'f1': F1Score(task="multiclass", num_classes=2, average='macro').to(device),
        'sensitivity': Recall(task="multiclass", num_classes=2, average='macro').to(device),
        'specificity': Specificity(task="multiclass", num_classes=2, average='macro').to(device),
        'confusion_matrix': ConfusionMatrix(task="multiclass", num_classes=2).to(device)
    }
    
    model = TwoStageModel().to(device)
    model = load_checkpoint_simple(model, 'outputs/main_model/best_two_stage_model.pth')
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating..."):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            
            # Update metrics
            for metric in metrics.values():
                metric.update(predictions, labels)

    # Get confusion matrix and convert to numpy for plotting
    conf_matrix = metrics['confusion_matrix'].compute().cpu().numpy()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(checkpoint_path, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(conf_matrix, os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Print final evaluation metrics
    print("\nFinal Evaluation Results:")
    for name, metric in metrics.items():
        value = metric.compute()
        if name != 'confusion_matrix':
            print(f"{name}: {value:.4f}")