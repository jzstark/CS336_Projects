
import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    #def forward(self, x):
    #    x = self.relu(self.fc1(x))
    #   x = self.ln(x)
    #    x = self.fc2(x)
    #    return x
    
    def forward(self, x):
        print("Input dtype:", x.dtype)
        x = self.relu(self.fc1(x))
        print("After fc1 dtype:", x.dtype)
        print("fc1 weight dtype:", self.fc1.weight.dtype)
        x = self.ln(x)
        print("After ln dtype:", x.dtype)
        print("ln weight dtype:", self.ln.weight.dtype)
        x = self.fc2(x)
        print("After fc2 dtype:", x.dtype)
        print("fc2 weight dtype:", self.fc2.weight.dtype)
        return x
    
with torch.autocast(device_type='cuda', dtype=torch.float16):
    model = ToyModel(10, 5).cuda()
    input_tensor = torch.randn(2, 10).cuda()
    print("Input dtype:", input_tensor.dtype)
    output_tensor = model(input_tensor)
    print("Output with mixed precision:", output_tensor)