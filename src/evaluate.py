import csv 
from tqdm import tqdm 
from models import MODELS
from utils import load_config
import os
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from dataloader import SliceDataset, Transforms2D, CTScansDataset, TRANSFORMS, DATASETS, ROOT_DIRS
from models import CustomResNet
import torchmetrics


# Import your custom dataset class and model class here

def load_checkpoint(model, checkpoint_path):
    """
    Load model weights from a checkpoint file.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    return model

def evaluate_half(model, val_loader, num_classes, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    task = "multiclass"
    accuracy_metric_macro = torchmetrics.Accuracy(task, num_classes=num_classes, average='macro').to(device)
    accuracy_metric_micro = torchmetrics.Accuracy(task, num_classes=num_classes, average='micro').to(device)
    confmat_metric = torchmetrics.ConfusionMatrix(task, num_classes=num_classes).to(device)

    output_dir = os.path.join(ckpt_path, 'eval/')
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = open(f'{output_dir}/predictions.csv', 'w', newline='')
    predictions_writer = csv.writer(predictions_file)
    predictions_writer.writerow(['True Label', 'Predicted Label'])

    accuracy_metric_macro.reset()
    accuracy_metric_micro.reset()
    confmat_metric.reset()
    threshold = 32
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating..."):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.argmax(dim=1).sum()
            outputs = outputs > threshold
            output = outputs.int().unsqueeze(0)
            label = labels[0].unsqueeze(0)
            assert(all(labels == labels[0]))

            predictions_writer.writerow([label, output])
            accuracy_metric_macro.update(output, label)
            accuracy_metric_micro.update(output, label)
            confmat_metric.update(output, label)
    
    # Log Metrics
    val_accuracy_macro = accuracy_metric_macro.compute()
    val_accuracy_micro = accuracy_metric_micro.compute()
    confusion_matrix = confmat_metric.compute()

    print(f'Val Acc (Macro): {val_accuracy_macro:.4f}, Val Acc (Micro): {val_accuracy_micro:.4f}')
    print(confusion_matrix)

def evaluate_3d(model, val_loader, num_classes, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    task = "multiclass"
    accuracy_metric_macro = torchmetrics.Accuracy(task, num_classes=num_classes, average='macro').to(device)
    accuracy_metric_micro = torchmetrics.Accuracy(task, num_classes=num_classes, average='micro').to(device)
    confmat_metric = torchmetrics.ConfusionMatrix(task, num_classes=num_classes).to(device)

    output_dir = os.path.join(ckpt_path, 'eval/')
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = open(f'{output_dir}/predictions.csv', 'w', newline='')
    predictions_writer = csv.writer(predictions_file)
    predictions_writer.writerow(['True Label', 'Predicted Label'])

    accuracy_metric_macro.reset()
    accuracy_metric_micro.reset()
    confmat_metric.reset()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating..."):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = []
            for slice in range(images.shape[4]):
                repeated_images = images[:, :, :, :, slice]
                output = model(repeated_images)
                outputs.append(output)

            outputs = torch.hstack([output.argmax(dim=1, keepdims=True) for output in outputs]).sum(axis=1)
            outputs = outputs > 30
            outputs = outputs.int()
            for label, pred in zip(labels.tolist(),outputs.tolist()):
                predictions_writer.writerow([label, pred])
            accuracy_metric_macro.update(outputs, labels)
            accuracy_metric_micro.update(outputs, labels)
            confmat_metric.update(outputs, labels)
    
    # Log Metrics
    val_accuracy_macro = accuracy_metric_macro.compute()
    val_accuracy_micro = accuracy_metric_micro.compute()
    confusion_matrix = confmat_metric.compute()

    print(f'Val Acc (Macro): {val_accuracy_macro:.4f}, Val Acc (Micro): {val_accuracy_micro:.4f}')
    print(confusion_matrix)

def evaluate_model(model, val_loader, num_classes, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    task = "multiclass"
    accuracy_metric_macro = torchmetrics.Accuracy(task, num_classes=num_classes, average='macro').to(device)
    accuracy_metric_micro = torchmetrics.Accuracy(task, num_classes=num_classes, average='micro').to(device)
    confmat_metric = torchmetrics.ConfusionMatrix(task, num_classes=num_classes).to(device)

    output_dir = os.path.join(ckpt_path, 'eval/')
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = open(f'{output_dir}/predictions.csv', 'w', newline='')
    predictions_writer = csv.writer(predictions_file)
    predictions_writer.writerow(['True Label', 'Predicted Label'])

    accuracy_metric_macro.reset()
    accuracy_metric_micro.reset()
    confmat_metric.reset()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating..."):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            accuracy_metric_macro.update(outputs.argmax(dim=1), labels)
            accuracy_metric_micro.update(outputs.argmax(dim=1), labels)
            confmat_metric.update(outputs.argmax(dim=1), labels)

            for label, pred in zip(labels.tolist(), outputs.argmax(dim=1).tolist()):
                predictions_writer.writerow([label, pred])
    
    # Log Metrics
    val_accuracy_macro = accuracy_metric_macro.compute()
    val_accuracy_micro = accuracy_metric_micro.compute()
    confusion_matrix = confmat_metric.compute()

    print(f'Val Acc (Macro): {val_accuracy_macro:.4f}, Val Acc (Micro): {val_accuracy_micro:.4f}')
    print(confusion_matrix)

def load_model(checkpoint_path, root_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(checkpoint_path, 'best_model.pth')
    config_path = os.path.join(checkpoint_path, 'used_config.yaml')
    config = load_config(config_path)
    classes_to_include = config['classes_to_include']
    class_mapping = config['class_mapping']
    num_classes = len(classes_to_include) if class_mapping is None else len(set(class_mapping.values()))
    model = config['model']
    model = MODELS[model](num_classes=num_classes).to(device)
    model = load_checkpoint(model, model_path)

    return model


def main_3d(checkpoint_path, root_dir):
    """
    Main function to evaluate the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(checkpoint_path, 'best_model.pth')
    config_path = os.path.join(checkpoint_path, 'used_config.yaml')
    config = load_config(config_path)
    classes_to_include = config['classes_to_include']
    class_mapping = config['class_mapping']
    num_classes = len(classes_to_include) if class_mapping is None else len(set(class_mapping.values()))

    batch_size = config['batch_size']
    model = config['model']
    model = MODELS[model](num_classes=num_classes).to(device)
    val_dataset = DATASETS['3d'](root_dir=root_dir, split='val', transform=TRANSFORMS['3d'].get_val_transforms(), include_classes=classes_to_include, class_mapping=class_mapping)

    # Load the model weights
    model = load_checkpoint(model, model_path)

    # Prepare the validation data
    val_size = 20
    # val_dataset = torch.utils.data.Subset(val_dataset, list(range(val_size)) + list(range(-val_size, 0)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    evaluate_3d(model, val_loader, num_classes, device)

def main_half(checkpoint_path, root_dir):
    """
    Main function to evaluate the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(checkpoint_path, 'best_model.pth')
    config_path = os.path.join(checkpoint_path, 'used_config.yaml')
    config = load_config(config_path)
    classes_to_include = config['classes_to_include']
    class_mapping = config['class_mapping']
    num_classes = len(classes_to_include) if class_mapping is None else len(set(class_mapping.values()))

    # batch_size = config['batch_size']
    batch_size = 64
    model = config['model']
    model = MODELS[model](num_classes=num_classes).to(device)
    val_dataset = DATASETS['2d'](root_dir=root_dir, split='val', transform=TRANSFORMS['2d'].get_val_transforms(), include_classes=classes_to_include, class_mapping=class_mapping)

    # Load the model weights
    model = load_checkpoint(model, model_path)

    # Prepare the validation data
    val_size = batch_size * 20
    # val_dataset = torch.utils.data.Subset(val_dataset, list(range(val_size)) + list(range(-val_size, 0)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    evaluate_half(model, val_loader, num_classes, device)

def main(checkpoint_path, root_dir):
    """
    Main function to evaluate the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(checkpoint_path, 'best_model.pth')
    config_path = os.path.join(checkpoint_path, 'used_config.yaml')
    config = load_config(config_path)
    classes_to_include = config['classes_to_include']
    class_mapping = config['class_mapping']
    num_classes = len(classes_to_include) if class_mapping is None else len(set(class_mapping.values()))

    batch_size = config['batch_size']
    model = config['model']
    model = MODELS[model](num_classes=num_classes).to(device)

    # Load the model weights
    model = load_checkpoint(model, model_path)

    # Prepare the validation data
    val_size = 100
    val_transforms = Transforms2D.get_val_transforms()  # Assuming this is your validation transforms
    val_dataset = SliceDataset(root_dir=root_dir, split='val', transform=val_transforms, include_classes=classes_to_include, class_mapping=class_mapping)
    # val_dataset = torch.utils.data.Subset(val_dataset, list(range(val_size)) + list(range(-val_size, 0)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    evaluate_model(model, val_loader, num_classes, device)

if __name__ == '__main__':
    # ckpt_path = 'outputs/resnet_no_dropout_custom_resnet_lr_0.001_epochs_100_bs_32__weighted_False_classes-0_2_3_4'  # Path to the model checkpoint
    # root_dir = ROOT_DIRS['2d']
    # main(ckpt_path, root_dir)

    # ckpt_path = 'outputs/resnet_no_dropout_custom_resnet_lr_0.001_epochs_100_bs_32__weighted_False_classes-0_1'
    # ckpt_path = 'outputs/resnet_no_dropout_custom_resnet_lr_0.001_epochs_100_bs_32__weighted_False_classes-0_2_3_4'  # Path to the model checkpoint
    ckpt_path = 'outputs/main_model/'
    # ckpt_path = 'outputs/resnet_no_dropout_custom_resnet_lr_0.001_epochs_100_bs_32__weighted_False_classes-0_2_3_4'  # Path to the model checkpoint
    root_dir = ROOT_DIRS['3d']
    main_3d(ckpt_path, root_dir)

    