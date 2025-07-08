import regex as re
from typing import Callable

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _convert_token(word: re.Match[str]) -> tuple[bytes, ...]:
    return tuple(
        bytes([b]) for b in bytes(word.group(0).strip(), 'utf-8'))


def _replace_pair(A: tuple[bytes, ...], B: tuple[bytes, ...]) -> tuple[bytes, ...]: # actually B is a pair of bytes
    result = []
    i = 0
    while i < len(A):
        if i < len(A) - 1 and A[i] == B[0] and A[i + 1] == B[1]:
            result.append(B[0] + B[1])  # Join the two matched chars
            i += 2
        else:
            result.append(A[i])
            i += 1
    return tuple(result)


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

    # Step #1: Vocabulary initialization 
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges = [] # list[tuple[bytes, bytes]]
    freq_table : dict[tuple[bytes,...], int] = {} 

    # Step #2: pre-tokenization 
    # read the input file
    with open(input_path, 'rb') as f:
        data = f.read()

    for line in data.splitlines():
        tokens  = re.finditer(PAT, line.decode('utf-8'))
        tokens  = [_convert_token(match) for match in tokens]
        for t in tokens:
            if t not in freq_table:
                freq_table[t] = 0
            freq_table[t] += 1

    # Step #3: Merging

    for _ in range(MERGE_ITER):
    
        freq_table2 : dict[tuple[bytes,...], int]  = {}

        for (char_tuple, cnt) in freq_table.items():
            for i in range(len(char_tuple) - 1):
                pair = (char_tuple[i], char_tuple[i + 1])
                if pair not in freq_table2:
                    freq_table2[pair] = 0
                freq_table2[pair] += cnt
        
        max_comb = max(freq_table2, key = lambda k: (freq_table2[k], k))
        merges.append(max_comb)

        freq_table2 = {}
        for (char_tuple, cnt) in freq_table.items():
            updated_tuple = _replace_pair(char_tuple, max_comb)
            freq_table2[updated_tuple] = freq_table[char_tuple]
        freq_table = freq_table2

    # Update the vocabulary with merges 
    for bpair in merges:
        vocab[len(vocab)] = bpair[0] + bpair[1]

    for special_token in special_tokens:
        if special_token not in vocab.values():
            vocab[len(vocab)] = bytes(special_token, 'utf-8')

    # print(merges)        
    return vocab, merges


#bpe_train('test.txt', 1000, ['<|endoftext|>'])