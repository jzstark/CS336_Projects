import sys
from pathlib import Path

# Add the proj1_basic directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "cs336-basics"))

from cs336_basics import model #type: ignore
from config import get_config, legal_config_name
import timeit
import torch
import numpy as np

import torch.cuda.nvtx as nvtx


def build_transformer_model(config, context_length=128):

    transformer = model.BasicsTransformerLM(
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        vocab_size=config['vocab_size'],
        rope_theta=config['theta'],
        context_length=context_length,
    ).to(config['device'])
    return transformer


def time_model(transformer, config, context_length=128, backward=True, n_steps=10):
    batch_size = config['batch_size']
    context_length = transformer.context_length
    input_tensor = torch.randint(
        0, transformer.vocab_size, (batch_size, context_length), 
        device=config['device']
    )
    if backward:
        transformer.train()
    else:
        transformer.eval()
        
    # Warm-up
    print("Warming up the model...")
    for _ in range(5):
        output_tensor = transformer(input_tensor)
        if backward:
            loss = torch.nn.functional.cross_entropy(output_tensor.view(-1, transformer.vocab_size), input_tensor.view(-1))
            loss.backward()
        torch.cuda.synchronize()
    
    
    # Measure time
    forward_time = []
    backward_time = []
    for _ in range(n_steps):
        print(f"Step {_ + 1}/{n_steps}")
        start_time = timeit.default_timer()
        _output = transformer(input_tensor)
        end_time1 = timeit.default_timer()
        if backward:
            _loss = torch.nn.functional.cross_entropy(_output.view(-1, transformer.vocab_size), input_tensor.view(-1))
            _loss.backward()
        torch.cuda.synchronize()
        end_time2 = timeit.default_timer()

        forward_time.append(end_time1 - start_time)
        if backward:
            backward_time.append(end_time2 - end_time1)

    fwd_avg = np.mean(forward_time)
    fwd_dev = np.std(forward_time)
    bwd_avg = np.mean(backward_time) 
    bwd_dev = np.std(backward_time) 
    
    print(f"Forward time: {fwd_avg:.6f} ± {fwd_dev:.6f} seconds")
    print(f"Backward time: {bwd_avg:.6f} ± {bwd_dev:.6f} seconds")

    return fwd_avg, bwd_avg, fwd_dev, bwd_dev


@nvtx.range("time_model_ntx")
def time_model_ntx(transformer, config, context_length=128, backward=True, n_steps=10):
    batch_size = config['batch_size']
    context_length = transformer.context_length
    input_tensor = torch.randint(
        0, transformer.vocab_size, (batch_size, context_length), 
        device=config['device']
    )
    if backward:
        transformer.train()
    else:
        transformer.eval()
        
    # Warm-up
    with nvtx.range("Warm-up"):
        for _ in range(5):
            output_tensor = transformer(input_tensor)
            if backward:
                loss = torch.nn.functional.cross_entropy(output_tensor.view(-1, transformer.vocab_size), input_tensor.view(-1))
                loss.backward()
            torch.cuda.synchronize()
    
    
    # Measure time
    for _ in range(n_steps):
        print(f"Step {_ + 1}/{n_steps}")
        with nvtx.range(f"Timing Step {_ + 1}/{n_steps} with backward={backward}"):
            _output = transformer(input_tensor)
            if backward:
                _loss = torch.nn.functional.cross_entropy(_output.view(-1, transformer.vocab_size), input_tensor.view(-1))
                _loss.backward()
            torch.cuda.synchronize()


config_name = "config_large"
assert legal_config_name(config_name), f"Invalid config name: {config_name}"
config = get_config(config_name)

context_length = 256

model = build_transformer_model(config, context_length)
#time_model(model, config, context_length)

#print("forward pass timing")
#time_model_ntx(model, config, context_length, backward=False)

print("backward pass timing")
time_model_ntx(model, config, context_length)


"""
# Direct timing 

128 context length

Small: 
Forward time: 0.025184 ± 0.004708 seconds
Backward time: 0.025723 ± 0.003331 seconds

Medium:
Forward time: 0.041769 ± 0.002313 seconds
Backward time: 0.045823 ± 0.001156 seconds

Large:
Warming up the model...
Forward time: 0.040173 ± 0.000735 seconds
Backward time: 0.045097 ± 0.001265 seconds

xl:
Forward time: 0.079619 ± 0.003536 seconds
Backward time: 0.167457 ± 0.000817 seconds


256 context length:

Large:

Forward time: 0.119940 ± 0.006997 seconds
Backward time: 0.154517 ± 0.002929 seconds


512 context length:

small:
Forward time: 0.021785 ± 0.001899 seconds
Backward time: 0.061706 ± 0.001953 seconds

medium:
Forward time: 0.037306 ± 0.001294 seconds
Backward time: 0.212388 ± 0.010809 seconds

large:
Just too long... 


# Timing with Nsys 


(1) Inference time

Different passes shows very different timing results... 
Warm-up: 891.3ms 
1st 10 forward pass: 95ms 
2nd 10 forward pass: 4.351s 
3rd: 87.33ms 
6th: 7.98s 

Something is wrong with the timing... 

Total time: 
Forward test: 42.7s
Backward test: 64.3s

(2) 

"""