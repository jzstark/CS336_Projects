from pathlib import Path
import torch

def get_config():
    config = {
        # Fixed hyperparameters
        'vocab_size': 10000,
        'context_length' : 256,
        'd_model': 512,
        'd_ff': 1344,
        'num_layers': 4,
        'num_heads': 16,
        # learning rate scheduling
        'learning_rate': 1e-4,
        'min_learning_rate': 1e-5,
        'warmup_iters': 2000,
        'cosine_cycle_iters': 100000, 
        #RoPE
        'use_rope': True,
        'theta': 10000, 

        # Other 
        'batch_size': 64,
        'num_epochs': 5,
        'validation_batch_size': 128,
        #'seq_len': 350,
        'max_seq_len': 512,
        'save_interval': 1,
        'log_interval': 1,
        'dropout': 0.1,

        # Paths
        'tokenizer_file': 'tmp_token/tokenizer.json',
        'training_text_file': '../data/TinyStoriesV2-GPT4-train.txt',
        'training_data_path': 'tmp_data/tinystory_training_tokens.npy',
        'validation_text_file': '../data/TinyStoriesV2-GPT4-valid.txt',
        'validation_data_path': 'tmp_data/tinystory_validation_tokens.npy',
        'tensorboard_logdir': 'tb_log/',
        'experiment_name': 'runs/tmodel',
        'datasource': 'opus_books',
        'model_folder': './weights',
        'model_basename': 'tmodel_',
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