import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def exp_name_from_config(config_path):
    config = load_config(config_path)
    batch_size = config.get('batch_size', 16)  # Default value if not in YAML
    learning_rate = config.get('learning_rate', 0.001)  # Default value
    num_epochs = config.get('num_epochs', 400)  # Default value
    exp_name = config.get('exp_name', '3d-cnn')  # Default value
    classes_to_include = config.get('classes_to_include', [0, 1, 2, 3, 4])  # Default value
    class_mapping = config.get('class_mapping', None)  # Default value
    weighted = config.get('weighted', False)  # Default value

    # Set up the output directory
    num_classes = len(classes_to_include) if class_mapping is None else len(set(class_mapping.values()))
    exp_name = f'{exp_name}_lr_{learning_rate}_epochs_{num_epochs}_bs_{batch_size}__weighted_{weighted}_classes-{"_".join([str(c) for c in classes_to_include])}'
    return exp_name