
permuted_mnist = {
    'input_dim':  784,
    'output_dim': 10,
    'hidden_size': [100, 100],
    'single_head': True,
    'data_name': 'permuted_mnist',
}

split_mnist = {
    'input_dim': 784,
    'output_dim': 10,
    'hidden_size': [256, 256],
    'single_head': False,
    'data_name': 'split_mnist',
}

no_mnist = {
    'input_dim': 784,
    'output_dim': 10, 
    'hidden_size': [150, 150, 150, 150],
    'single_head': False,
    'data_name': 'no_mnist',
}

split_cifar = {
    'input_dim': 784,
    'output_dim': 10, 
    'hidden_size': [150, 150, 150, 150],
    'single_head': False,
    'data_name': 'split_cifar',
}

permuted_mnist_mh = {
    'input_dim':  784,
    'output_dim': 10,
    'hidden_size': [100, 100],
    'single_head': False,
    'data_name': 'permuted_mnist'
}

valid_configs = [config_name for config_name in globals() if isinstance(globals()[config_name], dict)]

def load_task_config(task_config: str = ''):
    if task_config in valid_configs:
        return tuple(globals()[task_config].values())
    else:
        raise ValueError(f"Invalid config_name give. Either create a new config or select a valid config from {valid_configs}")
