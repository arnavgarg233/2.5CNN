import yaml
import os
from scheduler import get_next_job_id, get_next_dont_increment

def generate_yaml(new_yaml_id, batch_size=None, class_mapping=None, classes_to_include=None, 
                  exp_name=None, learning_rate=None, num_epochs=None, weighted = None, default_yaml_path='multiplexer/default.yaml'):
    
    # Load defaults from the provided YAML file
    with open(default_yaml_path, 'r') as file:
        default_config = yaml.safe_load(file)

    # Create new configuration based on defaults and provided arguments
    new_config = {
        'batch_size': batch_size or default_config['batch_size'],
        'class_mapping': class_mapping or default_config['class_mapping'],
        'classes_to_include': classes_to_include or default_config['classes_to_include'],
        'exp_name': exp_name or default_config['exp_name'],
        'learning_rate': learning_rate or default_config['learning_rate'],
        'num_epochs': num_epochs or default_config['num_epochs'],
        'weighted': weighted or default_config['weighted']
    }

    # Generate a new YAML file with the new configuration
    os.makedirs('multiplexer/launch_configs', exist_ok=True)
    yaml_path = f'multiplexer/launch_configs/{new_yaml_id}.yaml'
    with open(yaml_path, 'w') as file:
        yaml.dump(new_config, file, default_flow_style=False)

    print("Generated new configuration YAML file: {new_config}.yaml")
    return yaml_path

if __name__ == '__main__':
    # job_id_file = 'multiplexer/job_id.txt'
    # next_id = get_next_dont_increment(job_id_file) + 1
    for i in range(1,5):
        classes_to_include = [j for j in range(i+1)]
        class_mapping = {}
        for j in range(i+1):
            class_mapping[j] = 1
        class_mapping[0] = 0


        yaml_name = f'0_vs_{i}'
        yaml_path = generate_yaml(yaml_name, classes_to_include=classes_to_include, class_mapping=class_mapping)

