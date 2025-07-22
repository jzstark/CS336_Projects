from pathlib import Path
import torch

def get_config():
    config = {
        'batch_size': 8,
        'num_epochs': 300,
        'lr': 1e-4,
        'seq_len': 350,
        'd_model': 512,
        'context_length' : 512,
        'num_layers': 16,
        'num_heads': 8,
        'd_ff': 2048,
        'max_seq_len': 2048,
        'vocab_size': 1000,
        'dropout': 0.1,
        'theta': 0.5,
        'use_rope': True,
        'lang_src': 'en',
        'lang_tgt': 'it',
        'model_folder': './weights',
        'model_basename': 'tmodel_',
        'preload': None, 
        'tokenizer_file': 'tmp_token/tokenizer.json',
        'token_training_data_path': '../tests/fixtures/corpus.en',
        'experiment_name': 'runs/tmodel',
        'datasource': 'opus_books',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    return config


def get_weights_file_path(config, epoch : str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    #model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    print(f"Found {len(weights_files)} weights files in {model_folder}")
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])