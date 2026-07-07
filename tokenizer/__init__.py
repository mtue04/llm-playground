import os
from tokenizers import Tokenizer


class LLMTokenizer:
    def __init__(self, tokenizer_path="tokenizer/vocab/tokenizer.json"):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.bos_id = self.tok.token_to_id("<bos>")
        self.eos_id = self.tok.token_to_id("<eos>")
        self.pad_id = self.tok.token_to_id("<pad>")
        for name, tid in [("bos", self.bos_id), ("eos", self.eos_id), ("pad", self.pad_id)]:
            if tid is None:
                raise ValueError(
                    f"Special token <{name}> not found in tokenizer vocab at {tokenizer_path}. "
                    f"Make sure the tokenizer was trained with this special token."
                )
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts, add_bos=False, add_eos=False):
        encodings = self.tok.encode_batch(texts)
        results = []
        for enc in encodings:
            ids = enc.ids
            if add_bos:
                ids = [self.bos_id] + ids
            if add_eos:
                ids = ids + [self.eos_id]
            results.append(ids)
        return results
    
    def decode(self, ids):
        ids = [id for id in ids if id not in (self.bos_id, self.eos_id, self.pad_id)]
        return self.tok.decode(ids)
    

if __name__ == "__main__":
    tokenizer = LLMTokenizer()
    ids = tokenizer.encode("Hello, how are you?")
    print("Text: Hello, how are you?")
    print("Encoded IDs:", ids)
    print("Number of tokens:", len(ids))
    print("Decoded text:", tokenizer.decode(ids))