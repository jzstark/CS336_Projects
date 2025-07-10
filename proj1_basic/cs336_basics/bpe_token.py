import regex as re
from typing import Callable
from collections import defaultdict

import cProfile
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _convert_token(word: re.Match[str]) -> tuple[bytes, ...]:
    text = word.group(0)
    return tuple(bytes([b]) for b in text.encode('utf-8'))
    
def _replace_pair(A: tuple[bytes, ...], B: tuple[bytes, ...]) -> \
    tuple[tuple[bytes, ...], dict[tuple[bytes, bytes], int]]: # actually B is a pair of bytes
    result = []
    update = defaultdict(int)
    i = 0
    while i < len(A):
        if i < len(A) - 1 and A[i] == B[0] and A[i + 1] == B[1]:
            result.append(B[0] + B[1])  # Join the two matched chars

            if i > 0 : 
                update[(A[i - 1], A[i])] -= 1
                update[(A[i - 1], B[0] + B[1])] += 1
            if i < len(A) - 2:
                update[(A[i+1], A[i+2])] -= 1
                update[(B[0] + B[1], A[i + 2])] += 1
            
            i += 2
            
        else:
            result.append(A[i])
            i += 1
    return tuple(result), update


def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a Byte Pair Encoding (BPE) model on the input data.

    Args:
        input_path (str): Path to the input file containing training data.
        vocal_size (int): Size of the vocabulary to be created.
        special_tokens (list[str]): List of special tokens to be included in the vocabulary.

    Returns:
        tuple: A tuple containing:
            - A dictionary mapping token IDs to byte sequences.
            - A list of tuples representing the BPE merges.
    """
    #TODO: Change this fixed iteration parameter
    MERGE_ITER = vocab_size - len(special_tokens) - 256  # Number of merge iterations
    pattern = b"|".join(re.escape(tok.encode("utf-8")) for tok in special_tokens)
    regex = re.compile(pattern)

    #profiler = cProfile.Profile()

    # Step #1: Vocabulary initialization 
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges = [] # list[tuple[bytes, bytes]]
    freq_table : dict[tuple[bytes,...], int] = {} 

    start = time.time()
    # Step #2: pre-tokenization 
    # read the input file
    with open(input_path, 'rb') as f:
        data = f.read()
    # remove special tokens and split on them into segments
    segments = regex.split(data)

    for s in segments:
        tokens  = re.finditer(PAT, s.decode('utf-8'))
        tokens  = [_convert_token(match) for match in tokens]
        for t in tokens:
            if t not in freq_table:
                freq_table[t] = 0
            freq_table[t] += 1
    end1 = time.time()

    # Step #3: Merging

    # remove () as key from freq_table:
    freq_table = {k: v for k, v in freq_table.items() if len(k) > 1}

    pair_table : dict[tuple[bytes, bytes], int] = {}
    for (char_tuple, cnt) in freq_table.items():
        for i in range(len(char_tuple) - 1):
            pair = (char_tuple[i], char_tuple[i + 1])
            if pair not in pair_table:
                pair_table[pair] = 0
            pair_table[pair] += cnt

    for _ in range(MERGE_ITER):
        max_comb = max(pair_table, key = lambda k: (pair_table[k], k))
        merges.append(max_comb)

        # tmp corpus/freq_table 
        freq_table2 : dict[tuple[bytes,...], int]  = {}
        for (char_tuple, cnt) in freq_table.items():
            updated_tuple, pair_updates = _replace_pair(char_tuple, max_comb)
            freq_table2[updated_tuple] = freq_table[char_tuple]
            
            # keep updating the pair_table 
            pair_table.pop(max_comb, None)
            for (pair, update_cnt) in pair_updates.items():
                if pair not in pair_table:
                    pair_table[pair] = 0
                pair_table[pair] += update_cnt * cnt          

        freq_table = freq_table2

    # Update the vocabulary with merges 
    for bpair in merges:
        vocab[len(vocab)] = bpair[0] + bpair[1]

    for special_token in special_tokens:
        if special_token not in vocab.values():
            vocab[len(vocab)] = bytes(special_token, 'utf-8')
    end2 = time.time()

    print(f"Pre-tokenization time: {end1 - start:.2f} seconds")
    print(f"Merge time: {end2 - end1:.2f} seconds")
    
    # print(merges)        
    return vocab, merges


#vocab, merge = bpe_train('test.txt', 1000, ['<|endoftext|>'])
#print(merge)