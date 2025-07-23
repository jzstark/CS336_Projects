"""
Now, it's finally time to put all of the components you implemented together into your main training script.
It will pay off to make it easy to start training runs with different hyperparameters (e.g., by taking them
as command-line arguments), since you will be doing these many times later to study how different choices
impact training.


Deliverable: Write a script that runs a training loop to train your model on user-provided input.
In particular, we recommend that your training script allow for (at least) the following:
• Ability to configure and control the various model and optimizer hyperparameters.
• Memory-efficient loading of training and validation large datasets with np.memmap.
• Serializing checkpoints to a user-provided path.
• Periodically logging training and validation performance (e.g., to console and/or an external
service like Weights and Biases).a

"""
import torch 
from pathlib import Path
from config import get_config
from transformer import get_model
import numpy as np
from tqdm import tqdm
from datetime import datetime

from jaxtyping import Int
from typing import List
from torch import Tensor
import numpy.typing as npt

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from learning import learning_rate_schedule, gradient_clipping, AdamW

special_tokens = ["<|endoftext|>"]
pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_tokenizer(config):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        vocab_size = config['vocab_size']

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern, behavior="isolated", invert=False) #type: ignore
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens) #type: ignore

        tokenizer.train([config['token_training_data_path']], trainer)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer



def get_batch(tokens: npt.NDArray, 
              batch_size: int, 
              context_length: int, 
              device: str) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    
    """
    Create a batch of input and target sequences from the given tokens. 
    Args:
        tokens: A numpy array of token IDs.
        batch_size: The number of sequences in the batch.
        context_length: The length of each sequence.
        device: The device to which the tensors should be moved (e.g., 'cpu' or 'cuda').        
    Returns:
        A tuple containing:
        - input_tensor: A tensor of shape (batch_size, context_length) containing the input
        - target_tensor: A tensor of shape (batch_size, context_length) containing the target
    """

    n_tokens = tokens.shape[0]
    max_start = n_tokens - context_length - 1
    assert max_start > 0, "Not enough tokens for the requested context_length"
    # Build input and target batches
    
    start_indices = np.random.randint(0, max_start + 1, size=batch_size)
    input_batch = np.stack([tokens[i : i + context_length] for i in start_indices])
    target_batch = np.stack([tokens[i + 1 : i + 1 + context_length] for i in start_indices])
    
    # Convert to torch tensors and move to device
    input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_batch, dtype=torch.long, device=device)
    return input_tensor, target_tensor


def temperature_softmax(logits: Tensor, dim=-1, temperature: float = 1.0) -> Tensor:
    """
    Apply temperature scaling to logits and return the softmax probabilities.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=dim)


#TODO: Implementation not exactly checked yet, but looks good.  
def top_p_sampling(probs: Tensor, p: float = 0.9) -> Tensor:
    """
    Apply top-p sampling to logits and return the sampled token.
    """
    assert torch.allclose(torch.sum(probs), torch.tensor(1.0), atol=1e-6)
    
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
    
    # Filter out tokens with cumulative probability above p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0  # Keep the first token
    filtered_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))
    
    # Sample from the filtered logits
    sampled_idx_in_sorted = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1).squeeze()
    # Map back to original token index
    sampled_token = sorted_indices[sampled_idx_in_sorted]
    return sampled_token


def generate(model, tokenizer, prompt, device, max_len = 100):
    model.eval()
    input_tokens : List[Int] = tokenizer.encode(prompt).ids
    print("Input tokens:", input_tokens)
    special_tokens_id = [tokenizer.token_to_id(token) for token in special_tokens]
    
    with torch.no_grad():
        while len(input_tokens) < max_len:
            input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
            output = model(input_tensor)
            # Get the last token's logits
            last_probs = temperature_softmax(output[0, -1], dim=-1)
            next_token = top_p_sampling(last_probs, p=0.9).item()
            #next_token = int(torch.argmax(last_probs).item())
            if next_token in special_tokens_id:
                break
            input_tokens.append(next_token)

    generated_text = tokenizer.decode(input_tokens)
    print("Generated text:", generated_text)


def prepare_data(config, tokenizer, training=True, update_data=False, chunk_size=1024*1024):
    """
    Prepare the training/validation data by tokenizing the input text and saving it as a memory-mapped file.
    Processes the file in chunks to reduce memory usage.
    """
    if training:
        data_path = Path(config['training_data_path'])
        text_file = Path(config['training_text_file'])
    else:
        data_path = Path(config['validation_data_path'])
        text_file = Path(config['validation_text_file'])
    if update_data or not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        # First pass: count total tokens
        total_tokens = 0
        with open(text_file, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total_tokens += len(tokenizer.encode(chunk).ids)
        # Second pass: tokenize and write to memmap
        memmap_arr = np.memmap(data_path, dtype=np.int32, mode='w+', shape=(total_tokens,))
        idx = 0
        with open(text_file, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                token_ids = tokenizer.encode(chunk).ids
                memmap_arr[idx:idx+len(token_ids)] = token_ids
                idx += len(token_ids)
        memmap_arr.flush()
        print(f"Data saved to {data_path}")
    else:
        print(f"Data already exists at {data_path}")


def evaluate(model, tokens, batch_size, context_length, device):
    model.eval()
    losses = []
    with torch.no_grad():
        num_batches = len(tokens) // batch_size
        for _ in range(num_batches):
            # print(f"Evaluating batch {_+1}/{num_batches}")
            input_tensor, target_tensor = get_batch(tokens, batch_size, context_length, device)
            output = model(input_tensor)
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), target_tensor.view(-1))
            losses.append(loss.item())
    model.train()
    return np.mean(losses)


def train(config, model):
    device = config['device']
    model.train()
    
    # Load training and validation data
    tokens = np.memmap(Path(config['training_data_path']), dtype=np.int32, mode='r')
    val_tokens = np.memmap(Path(config['validation_data_path']), dtype=np.int32, mode='r')
    
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    
    for epoch in range(config['num_epochs']):
        num_batches = len(tokens) // config['batch_size']

        val_loss = evaluate(model, val_tokens, config['validation_batch_size'], config['context_length'], device)
        print(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")
        

        with tqdm(range(0, len(tokens), config['batch_size']), total=num_batches, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx in pbar:
                input_tensor, target_tensor = get_batch(tokens, config['batch_size'], config['context_length'], device)
                
                optimizer.zero_grad()
                output = model(input_tensor)
                loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), target_tensor.view(-1))
                loss.backward()

                ## Gradient clipping (before optimizer.step)
                #gradient_clipping(model.parameters(), config['max_grad_norm'])
                #
                ## Learning rate scheduling (update optimizer's lr)
                #t = epoch * (len(tokens) // config['batch_size']) + (batch_idx // config['batch_size'])
                #lr = learning_rate_schedule(
                #    t,
                #    config['learning_rate'],
                #    config['min_learning_rate'],
                #    config['warmup_iters'],
                #    config['cosine_cycle_iters']
                #)
                #for param_group in optimizer.param_groups:
                #    param_group['lr'] = lr

                optimizer.step()
                pbar.set_postfix(loss=loss.item())
        

        # Save the model
        if epoch % config['save_interval'] == 0 or epoch == config['num_epochs'] - 1:
            save_path = Path(config['model_folder'] + f"/{config['model_basename']}{epoch+1}.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saving model to {save_path}")
        
       
        

if __name__ == "__main__":
    config = get_config()
    model = get_model(config, vocab_size=config['vocab_size']).to(config['device'])
    tokenizer = get_tokenizer(config)
    
    print(f"Preparing training data...{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    prepare_data(config, tokenizer, training=True, update_data=False)
    print(f"Preparing validation data...{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    prepare_data(config, tokenizer, training=False, update_data=False)

    print(f"Start training...{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    train(config, model)

    print(f"Finish Training, a simple check ...{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    prompt = "in a larger sense, we can not dedicate"
    generate(model, tokenizer, prompt, config['device'], max_len=100)

