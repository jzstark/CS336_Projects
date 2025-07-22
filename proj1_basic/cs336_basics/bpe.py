from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.pre_tokenizers import Split
import time
import os


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def load_merges_as_bytes(filepath: str) -> list[tuple[bytes, bytes]]:
    merges = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):  # skip header
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))
    return merges


def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]) -> \
    tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Create a Byte-Pair Encoding model
    tokenizer = Tokenizer(models.BPE())
    pattern = PAT
    tokenizer.pre_tokenizer = Split(pattern, behavior="isolated", invert=False)

    # Trainer
    trainer = trainers.BpeTrainer(vocab_size = vocab_size, special_tokens= special_tokens)

    #start = time.time()
    tokenizer.train([input_path], trainer)
    #end = time.time()
    #print(f"Training time: {end - start:.2f} seconds")

     # Save to disk temporarily
    model_dir = "./tmp_bpe_model"
    os.makedirs(model_dir, exist_ok=True)
    _vocab_file, merges_file = tokenizer.model.save(model_dir, "my_bpe")

    # Build vocab dict[int, bytes]
    vocab = tokenizer.get_vocab()
    token_bytes_vocab: dict[int, bytes] = {
        token_id: token_str.encode('utf-8') for token_str, token_id in vocab.items()
    }

    # Load merges from merges.txt
    merges_list = load_merges_as_bytes(merges_file)
    
    #print(token_bytes_vocab)
    #print(merges_list)
    print(len(token_bytes_vocab), "tokens in the vocabulary")

    return token_bytes_vocab, merges_list

#start = time.time()
#vocab, merge = bpe_train('test.txt', 1000, ['<|endoftext|>'])
#end = time.time()
#print(merge)


"""
HUGE performance Gap (give the same arguments (1000 vocab, one special token)):
on a small 5M file, my implementation takes around 6 seconds (1.5 Pre-tokenization, 4.5 Merging), 
while this off-the-shelf HuggingFace implementation takes around 2.38 seconds. 

Also, this implementation cannot pass the tests 
"""