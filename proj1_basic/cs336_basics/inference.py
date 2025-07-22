import torch
from config import get_config
from transformer import get_model
from bpe import bpe_train
from tokenizer import Tokenizer

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
config = get_config()
vocab_size = 100  # Example vocab size, adjust as needed
model = get_model(config, vocab_size).to(device)

# Load the pretrained weights

input_string = "in a larger sense, we can not dedicate"

# prepare the tokenizer

vocab, merge = bpe_train('../tests/fixtures/address.txt', vocab_size, ['<|endoftext|>'])
tokenizer = Tokenizer(vocab, merge, special_tokens=['<|endoftext|>'])

input_tokens = tokenizer.encode(input_string)
print("Input tokens:", input_tokens)
print("Input shape: ", len(input_tokens))

model.eval()
with torch.no_grad():
    input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0) 
    output = model(input_tensor)
    # output = torch.softmax(output, dim=-1)
    last_probs = torch.softmax(output[0, -1], dim=-1)
    next_token = int(torch.argmax(last_probs).item())
    next_word = tokenizer.decode([next_token])
    print("Predicted next word:", next_word)