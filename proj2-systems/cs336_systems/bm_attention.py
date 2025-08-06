import sys
from pathlib import Path
import timeit
from tracemalloc import start
import torch
import numpy as np

import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

# Add the proj1_basic directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "cs336-basics"))
from cs336_basics import model #type: ignore
from config import get_config, legal_config_name

BATCH = 8
THETA = 10000
d_head = 1

MEM_RECORD = False
d_models = [16,32, 64,128]
context_lengths = [256, 1024, 4096, 8192]#, 16384

#d_models = [128]
#context_lengths = [8192]

def run_forward_backward(model, x):
    nvtx.range_push("forward")
    t0 = timeit.default_timer()
    output = model(x)
    torch.cuda.synchronize()
    t1 = timeit.default_timer()
    nvtx.range_pop()

    nvtx.range_push("dummy_loss & backward")
    loss = output.sum()  # Dummy loss
    loss.backward()
    torch.cuda.synchronize()
    t2 = timeit.default_timer()
    nvtx.range_pop()
    return t1 - t0, t2 - t1


#check the input shape of attention model
def build_model(d_model, context_length):
    position_encoding = model.RotaryEmbedding(
                context_length=context_length,
                dim = d_model // d_head,
                theta = THETA,
    )
    attention_model = model.CausalMultiHeadSelfAttention(
        d_model=d_model ,
        num_heads=d_head,
        positional_encoder=position_encoding,
    ).cuda()
    return attention_model
 

def run(d_model, context_length, n_steps):
    #print(f"Testing d_model={d_model}, context_length={context_length}")
    forward_times = []
    backward_times = []
    for n in range(n_steps):
        #with torch.autocast(device_type='cuda', dtype=torch.float16):
        attention_model = build_model(d_model, context_length)

        # Compile the model
        attention_model = torch.compile(attention_model)

        input_tensor = torch.randn(BATCH, context_length, d_model).cuda()
        tforward, tbackward = run_forward_backward(attention_model, input_tensor)
        #print(f"Step {n+1} - Forward time: {tforward:.4f}s, Backward time: {tbackward:.4f}s")
        forward_times.append(tforward)
        backward_times.append(tbackward)
    avg_forward_time = np.mean(forward_times)
    std_forward_time = np.std(forward_times)
    avg_backward_time = np.mean(backward_times)
    std_backward_time = np.std(backward_times)     
    return avg_forward_time, std_forward_time, avg_backward_time, std_backward_time
    

if __name__ == "__main__":
    if MEM_RECORD:
        torch.cuda.memory._record_memory_history(max_entries=1000000, stacks='all')
    
    for d_model in d_models:
        for context_length in context_lengths:
            # warm up 
            run(d_model, context_length, n_steps=3)
            avg_forward_time, std_forward_time, avg_backward_time, std_backward_time = run(d_model, context_length, n_steps=10)
    
            print(f"Results for d_model={d_model}, context_length={context_length}:")
            print(f"  Average Forward Time: {avg_forward_time:.4f}s ± { std_forward_time:.4f}s")
            print(f"  Average Backward Time: {avg_backward_time:.4f}s ± {std_backward_time:.4f}s")
            print("-" * 50)
    if MEM_RECORD:
        torch.cuda.memory._dump_snapshot("memory_snapshot_attention.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)


"""
Results for d_model=16, context_length=256:
  Average Forward Time: 0.0014s ± 0.0001s
  Average Backward Time: 0.0009s ± 0.0000s
--------------------------------------------------
Results for d_model=16, context_length=1024:
  Average Forward Time: 0.0013s ± 0.0000s
  Average Backward Time: 0.0013s ± 0.0000s
--------------------------------------------------
Results for d_model=16, context_length=4096:
  Average Forward Time: 0.0101s ± 0.0003s
  Average Backward Time: 0.0199s ± 0.0005s
--------------------------------------------------
Results for d_model=16, context_length=8192:
  Average Forward Time: 0.0361s ± 0.0007s
  Average Backward Time: 0.0773s ± 0.0007s
--------------------------------------------------
Results for d_model=32, context_length=256:
  Average Forward Time: 0.0027s ± 0.0002s
  Average Backward Time: 0.0024s ± 0.0002s
--------------------------------------------------
Results for d_model=32, context_length=1024:
  Average Forward Time: 0.0025s ± 0.0003s
  Average Backward Time: 0.0023s ± 0.0000s
--------------------------------------------------
Results for d_model=32, context_length=4096:
  Average Forward Time: 0.0106s ± 0.0007s
  Average Backward Time: 0.0203s ± 0.0005s
--------------------------------------------------
Results for d_model=32, context_length=8192:
  Average Forward Time: 0.0355s ± 0.0008s
  Average Backward Time: 0.0776s ± 0.0005s
--------------------------------------------------
Results for d_model=64, context_length=256:
  Average Forward Time: 0.0025s ± 0.0001s
  Average Backward Time: 0.0025s ± 0.0003s
--------------------------------------------------
Results for d_model=64, context_length=1024:
  Average Forward Time: 0.0026s ± 0.0002s
  Average Backward Time: 0.0023s ± 0.0001s
--------------------------------------------------
Results for d_model=64, context_length=4096:
  Average Forward Time: 0.0106s ± 0.0006s
  Average Backward Time: 0.0201s ± 0.0006s
--------------------------------------------------
Results for d_model=64, context_length=8192:
  Average Forward Time: 0.0356s ± 0.0008s
  Average Backward Time: 0.0774s ± 0.0005s
--------------------------------------------------
Results for d_model=128, context_length=256:
  Average Forward Time: 0.0121s ± 0.0274s
  Average Backward Time: 0.0024s ± 0.0001s
--------------------------------------------------
Results for d_model=128, context_length=1024:
  Average Forward Time: 0.0027s ± 0.0002s
  Average Backward Time: 0.0023s ± 0.0001s
--------------------------------------------------
Results for d_model=128, context_length=4096:
  Average Forward Time: 0.0109s ± 0.0004s
  Average Backward Time: 0.0206s ± 0.0006s
--------------------------------------------------
Results for d_model=128, context_length=8192:
  Average Forward Time: 0.0384s ± 0.0019s
  Average Backward Time: 0.0831s ± 0.0034s
--------------------------------------------------



Memory usage for d_model=128, context_length=8192: max 13GB; Only forward pass: max 9GB

Actually the peak usage comes from softmax in attention at leas in forward pass. About 4.5G in forward pass.
"""




"""Compiled version with torch.compile



"""