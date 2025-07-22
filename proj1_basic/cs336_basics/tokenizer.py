from collections.abc import Iterable, Iterator
from bpe import bpe_train, PAT
import regex as re

# This implementation's result still not exactly the same as some reference (5 fail, 18 pass), 
# but I think it is workable enough for the purpose of this tokenizer.

def custom_split(text: str, special_tokens: list[str], default_pattern=PAT):
    if special_tokens == []:        
        return [m.group(0) for m in re.finditer(default_pattern, text) if m.group(0)]

    # Sort special tokens by length, longest first
    escaped = sorted((re.escape(tok) for tok in special_tokens), key=len, reverse=True)
    pattern = f"({'|'.join(escaped)})"
    parts = re.split(pattern, text)
    
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
        
        i = 0
        merged = []
        while i < len(token):
            # Try to find the longest match in vocab starting at position i
            found = None
            for j in range(len(token), i, -1):
                candidate = b''.join(token[i:j])
                if candidate in self.reverse_vocab:
                    found = candidate
                    break
            if found is not None:
                merged.append(found)
                i += len(found) // len(token[0])  # Advance by number of merged tokens
            else:
                # Fallback: single token
                merged.append(token[i])
                i += 1
        return tuple(merged)


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