import argparse
import os
from datasets import load_dataset


def download_subset(target_mb, output_path, dataset_name="HuggingFaceFW/fineweb", split="train"):
    ds = load_dataset(dataset_name, name="sample-10BT", split=split, streaming=True)
 
    target_bytes = target_mb * 1024 * 1024
    written_bytes = 0
    n_docs = 0
 
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example["text"].strip()
            if not text:
                continue
            f.write(text + "\n\n")
            written_bytes += len(text.encode("utf-8"))
            n_docs += 1
 
            if n_docs % 1000 == 0:
                print(f"  ...{written_bytes / (1024*1024):.1f}MB, {n_docs} documents")
 
            if written_bytes >= target_bytes:
                break

    print(f"Downloaded {n_docs} documents, total size: {written_bytes / (1024*1024):.1f}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_mb", type=int, required=True, help="Target size in MB for the downloaded subset.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the downloaded subset.")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb", help="Name of the dataset to download from Hugging Face.")
    args = parser.parse_args()

    download_subset(args.target_mb, args.output, args.dataset_name)