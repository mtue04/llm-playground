import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from tokenizer import LLMTokenizer
 
 
def tokenize_corpus(input_path, tokenizer_path, output_dir, val_ratio=0.02):
    tok = LLMTokenizer(tokenizer_path)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split corpus into documents to enable parallel batch tokenization
    documents = [doc.strip() for doc in text.split("\n\n") if doc.strip()]
    print(f"Tokenizing {len(documents)} documents using fast batch encoding...")
    
    batch_ids = tok.encode_batch(documents, add_bos=True, add_eos=True)
    
    # Flatten the token list
    flat_ids = [tok_id for doc_ids in batch_ids for tok_id in doc_ids]
    ids = np.array(flat_ids, dtype=np.uint32)

    n = len(ids)
    n_val = int(n * val_ratio)
    if n_val == 0:
        train_ids = ids
        val_ids = np.array([], dtype=ids.dtype)
    else:
        train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    os.makedirs(output_dir, exist_ok=True)
    train_ids.tofile(os.path.join(output_dir, "train.bin"))
    val_ids.tofile(os.path.join(output_dir, "val.bin"))

    print(f"Tokenized {n} tokens: {len(train_ids)} for training, {len(val_ids)} for validation.")
    print(f"Saved to {os.path.join(output_dir, 'train.bin')} and {os.path.join(output_dir, 'val.bin')}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/corpus_clean.txt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/vocab/tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="data/processed/tokenized")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    args = parser.parse_args()

    tokenize_corpus(args.input, args.tokenizer, args.output_dir, args.val_ratio)