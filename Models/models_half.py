import torch 
from models import MODELS
import torch.nn.functional as F
from compare_2d_and_3d_data import get_dataloader
import torch.optim as optim
import torch.nn as nn

def load_checkpoint(model, checkpoint_path):
    """
    Load model weights from a checkpoint file and map them correctly to TwoStageModel.
    Then freeze the loaded parameters.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Create a new state dict for mapping
    new_state_dict = {}
    
    # Map the keys from checkpoint to TwoStageModel format
    for key, value in checkpoint.items():
        new_key = f"resnet.{key}"
        new_state_dict[new_key] = value
    
    # Load the mapped state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    # Freeze all parameters in the resnet
    for name, param in model.named_parameters():
        if name.startswith('resnet.'):
            param.requires_grad = False
            print(f"Freezing parameter: {name}")
    
    return model

class TwoStageModel(torch.nn.Module):
    """ 
    Two stage model that first runs a resnet on each slice of a 3d volume, then runs a resnet on the 2d slices.
    """
    def __init__(self):
        super(TwoStageModel, self).__init__()
        self.resnet = MODELS['custom_resnet'](num_classes=2)
        self.identity = torch.nn.Identity()
        # self.num_slices = 64
        # self.relu = torch.nn.ReLU()
        # self.fc1 = torch.nn.Linear(128, 64)
        # self.fc2 = torch.nn.Linear(64, 32)
        # self.fc3 = torch.nn.Linear(32, 2)
        # self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        # Debug prints
        # Apply the same SliceCNN to each slice
        slice_outputs = self.resnet(x)
        out = self.identity(slice_outputs)
        print(out.shape)
        
        # slice_features = slice_outputs.squeeze()  # (N, num_slices, 2)
        
        # slice_features = slice_features.view(-1)  # (N * num_slices * 2)
        
        # fwd_features = F.relu(self.dropout(self.fc1(slice_features)))
        
        # fwd_features = F.relu(self.dropout(self.fc2(fwd_features)))
        
        # out = self.fc3(fwd_features)
        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and load the two-stage model
    model = TwoStageModel().to(device)
    
    # Load the checkpoint, map the weights, and freeze the parameters
    model = load_checkpoint(model, 'outputs/main_model/best_model.pth')
    
    # Verify which parameters are trainable
    for name, param in model.named_parameters():
        print(f"{name}: trainable={param.requires_grad}")
    
    
    # Define class mapping for binary classification
    include_classes = [0, 1, 2, 3, 4]  # Include all classes
    class_mapping = {
        0: 0,  # Mild cases
        1: 1,  # Mild cases
        2: 1,  # Severe cases
        3: 1,  # Severe cases
        4: 1   # Severe cases
    }
    
    # Get the dataloader with class mapping
    train_loader = get_dataloader(
        '2d', 
        batch_size=64, 
        shuffle_blocks=True, 
        mode='train',
        include_classes=include_classes,
        class_mapping=class_mapping
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # assert all values in labels are 0 or 1
            if not (torch.all(labels == 0) or torch.all(labels == 1) or torch.all(labels == 2) or torch.all(labels == 3) or torch.all(labels == 4)):
                pass
            try:
                # Forward + backward + optimize
                outputs = model(inputs)
                argmax_outputs = torch.argmax(outputs, dim=1)
                print((argmax_outputs == labels).float().mean())
                loss = criterion(outputs, labels)
                # loss = criterion(outputs.unsqueeze(0), labels[0].unsqueeze(0))
                # loss.backward()
                # optimizer.step()
            except Exception as e:
                print(f"Error on batch {i + 1}: {e}")

            # Print statistics
            if i % 10 == 9:    # Print every 10 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {loss.item():.3f}')

    print('Finished Training')

    # Save the model
    torch.save(model.state_dict(), 'two_stage_model.pth')

    # Optional: Test a single batch
    inputs = torch.randn(8, 3, 128, 128, 64).to(device)
    out = model(inputs)
    print(out)
