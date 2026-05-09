from models2d import *
from models3d import *
import torch.optim as optim


MODELS = {
    'simple': Simple3DCNN,
    'simplebn': Simple3DCNNBN,
    'simple_modular': Simple3DCNNBN_Modular,
    'attn': CNN3DWithAttention,
    '2d_cnn': Simple2DCNN,
    'volume_cnn': VolumeCNNSharedWeights,
    'slice_cnn': SliceCNN,
    'custom_resnet': CustomResNet,
}

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 5 
    dataset_id = '2d'
    model = 'custom_resnet'
    model = MODELS[model](num_classes=num_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    my_transform = TRANSFORMS[dataset_id]
    root_dir = ROOT_DIRS[dataset_id]
    dataset = DATASETS[dataset_id]

    # Run through a full batch
    dataset = dataset(root_dir=root_dir, split='train', transform=my_transform.get_val_transforms())
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for batch in tqdm(dataloader, desc=f"Epoch  (Validation)"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        labels = labels.to(device)
        loss = F.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Do memory footprinting
    # inputs = torch.randn(16, 1, 128, 128, 64).to(device)
    # out, activations = model.forward_with_memory(inputs)
    # memory = memory_utilization(activations)
    # print(memory)