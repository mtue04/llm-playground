import argparse
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def train_bbpe_tokenizer(corpus_path, vocab_size, output_path):
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<|user|>", "<|assistant|>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train([corpus_path], trainer)
    
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, "tokenizer.json")
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file for training the tokenizer.")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size for the tokenizer.")
    parser.add_argument("--output", type=str, default="tokenizer/vocab", help="Path to save the trained tokenizer.")
    args = parser.parse_args()

    train_bbpe_tokenizer(args.input, args.vocab_size, args.output)