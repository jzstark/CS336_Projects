import torch
import numpy as np
from jaxtyping import Float, Int, Bool
from torch import Tensor
import numpy.typing as npt

import typing, os 


#When sampling from your dataset (i.e., a numpy array) during training, be sure load the dataset in memory-mapped mode

def get_batch(tokens: npt.NDArray, 
              batch_size: int, 
              context_length: int, 
              device: str) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    
    n_tokens = tokens.shape[0]
    # Sample random start indices so that each sequence fits in tokens
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

"""
should dump all the state from the
first three parameters into the file-like object out. You can use the state_dict method of both
the model and the optimizer to get their relevant states and use torch.save(obj, out) to dump
obj into out (PyTorch supports either a path or a file-like object here). A typical choice is to
have obj be a dictionary, but you can use whatever format you want as long as you can load your
checkpoint later.
"""

def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer) -> int:
    """
    Load a checkpoint from the given source and restore the model and optimizer states.
    
    Args:
        src: The source from which to load the checkpoint.
        model: The model to restore the state into.
        optimizer: The optimizer to restore the state into.
    
    Returns:
        The iteration number from the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']