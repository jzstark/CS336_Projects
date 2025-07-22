import torch
import numpy as np
from jaxtyping import Float, Int, Bool
from torch import Tensor
import numpy.typing as npt

import typing, os 


#When sampling from your dataset (i.e., a numpy array) during training, be sure load the dataset in memory-mapped mode



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


