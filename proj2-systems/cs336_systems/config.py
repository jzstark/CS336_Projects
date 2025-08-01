from pathlib import Path
import torch

base_config = {
    # Fixed hyperparameters
        'vocab_size': 10000,
        'batch_size': 4,
        # Other hyperparameters
        'learning_rate': 1e-4,
        'use_rope': True,
        'theta': 10000, 
        'max_seq_len': 512,
        
        # Paths
        'experiment_name': 'runs/tmodel',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

config_small = {
        'd_model': 768,
        'd_ff': 3072,
        'num_layers': 12,
        'num_heads': 12,
} | base_config


config_medium = {
        'd_model': 1024,
        'd_ff': 4096,
        'num_layers': 24,
        'num_heads': 16,
} | base_config

config_large = {
        'd_model': 1280,
        'd_ff': 5120,
        'num_layers': 36,
        'num_heads': 20,
} | base_config

config_xl = {
        'd_model': 1600,
        'd_ff': 6400,
        'num_layers': 48,
        'num_heads': 25,
} | base_config

config_2_7b= {
        'd_model': 2560,
        'd_ff': 10240,
        'num_layers': 32,
        'num_heads': 32,
} | base_config

configs = {
    'config_small': config_small,
    'config_medium': config_medium,
    'config_large': config_large,
    'config_xl': config_xl,
    'config_2_7b': config_2_7b,
}

def legal_config_name(name: str) -> bool:
    return name in configs

def get_config(name):
    return configs[name]
