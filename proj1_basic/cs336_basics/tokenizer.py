from collections.abc import Iterable, Iterator
#from bpe_token import bpe_train, PAT
from .bpe_token import bpe_train, PAT
import regex as re

def custom_split(text: str, special_tokens: list[str], default_pattern=PAT):
    # Sort special tokens by length, longest first
    escaped = sorted((re.escape(tok) for tok in special_tokens), key=len, reverse=True)
    # Pattern to split and keep delimiters (special tokens)
    split_pat = f"({'|'.join(escaped)})"
    parts = re.split(split_pat, text)
    tokens = []
    for part in parts:
        if not part:
            continue
        if part in special_tokens:
            tokens.append(part)
        else:
            tokens.extend([m.group(0) for m in re.finditer(default_pattern, part) if m.group(0)])
    return tokens


def convert_token(text: str) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in text.encode('utf-8'))


class Tokenizer :

    def __init__(self, vocab: dict[int, bytes], 
                 merges : list[tuple[bytes, bytes]], 
                 special_tokens : list[str] | None = None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        if special_tokens is None:
            self.special_tokens_byte = []
        else:
            self.special_tokens_byte = [tuple([bytes([b]) for b in token.encode('utf-8')]) for token in special_tokens]
        self.reverse_vocab = {v: k for k, v in vocab.items()}   
    

    def _merge_token(self, token : tuple[bytes, ...], ) -> tuple[bytes, ...]:
        if token in self.special_tokens_byte:
            return (b''.join(token),)
        
        while len(token) > 1: 
            pairs = [(token[i], token[i+1]) for i in range(len(token)-1)]
            merge_candidate = None 
            for pair in pairs:
                if pair[0] + pair[1] in  self.reverse_vocab:
                    merge_candidate = pair
                    break
            if merge_candidate is None: break

            # merge the pair
            new_t : list[bytes] = []
            j = 0
            while j < len(token):
                if j < len(token) - 1 and (token[j], token[j+1]) == merge_candidate:
                    new_t.append(token[j] + token[j+1])  # 合并
                    j += 2
                else:
                    new_t.append(token[j])
                    j += 1
            token = tuple(new_t)
        return token


    def encode(self, text : str) -> list[int]:
        #matches  = re.finditer(PAT, text)
        matches = custom_split(text, self.special_tokens)

        # parallelizable
        tokens  = [convert_token(match) for match in matches]
        tokens = [self._merge_token(token) for token in tokens]
        
        token_to_ints = lambda t: [self.reverse_vocab[b] for b in t] 
        token_id = []
        for ints in tokens:
            token_id.extend(token_to_ints(ints))
        return token_id
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    

    def decode(self, ids: list[int]) -> str:
        ret = b''
        for id in ids:
            ret = ret + self.vocab.get(id, b'')
        #print(ret)
        return ret.decode('utf-8', errors='replace')


def from_file(vocab_filepath: str, merges_filepath: str, 
              special_tokens: list[str] | None = None) -> Tokenizer:
    
    return Tokenizer(
        vocab= {},  # Load vocab from file
        merges=[],  # Load merges from file
        special_tokens=special_tokens
    )

"""
vocab, merge = bpe_train('test.txt', 1000, ['<|endoftext|>'])
tokenizer = Tokenizer(vocab, merge, special_tokens=['<|endoftext|>'])

print(merge)

with open('test.txt', 'r') as f:
    for line in f: 
        encoded = tokenizer.encode(line)
        print(f"Encoded: {encoded}")
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")
"""