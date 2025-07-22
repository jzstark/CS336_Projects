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

from jaxtyping import Int
from typing import List
from torch import Tensor
import numpy.typing as npt


from tokenizers import Tokenizer, models, trainers, pre_tokenizers

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


def generate(model, tokenizer, prompt, device, max_len = 100):
    model.eval()
    input_tokens : List[Int] = tokenizer.encode(prompt).ids
    print("Input tokens:", input_tokens)
    special_tokens_id = [tokenizer.token_to_id(token) for token in special_tokens]
    
    with torch.no_grad():
        while len(input_tokens) < max_len:
            input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
            output = model(input_tensor)
            last_probs = torch.softmax(output[0, -1], dim=-1)
            next_token = int(torch.argmax(last_probs).item())
            if next_token in special_tokens_id:
                break

            input_tokens.append(next_token)

    generated_text = tokenizer.decode(input_tokens)
    print("Generated text:", generated_text)


if __name__ == "__main__":
    config = get_config()
    model = get_model(config, vocab_size=config['vocab_size']).to(config['device'])
    tokenizer = get_tokenizer(config)

    # train(config, model, tokenizer)

    prompt = "in a larger sense, we can not dedicate"
    generate(model, tokenizer, prompt, config['device'], max_len=100)

