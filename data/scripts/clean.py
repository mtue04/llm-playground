import argparse
import hashlib
import os
import re
 
 
def is_low_quality(doc, min_words=20, max_symbol_ratio=0.3):
    words = doc.split()
    if len(words) < min_words:
        return True
 
    n_symbols = len(re.findall(r"[^\w\s]", doc))
    if n_symbols / max(len(doc), 1) > max_symbol_ratio:
        return True
 
    return False
 
 
def clean_corpus(input_path, output_path, min_words=20):
    seen_hashes = set()
    n_total, n_kept, n_dup, n_low_quality = 0, 0, 0, 0
 
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
 
        buffer = []
        for line in fin:
            if line.strip() == "" and buffer:
                doc = " ".join(buffer).strip()
                buffer = []
                n_total += 1
 
                if is_low_quality(doc, min_words=min_words):
                    n_low_quality += 1
                    continue
 
                doc_hash = hashlib.md5(doc.encode("utf-8")).hexdigest()
                if doc_hash in seen_hashes:
                    n_dup += 1
                    continue
                seen_hashes.add(doc_hash)
 
                fout.write(doc + "\n\n")
                n_kept += 1
            else:
                buffer.append(line.strip())
 
        if buffer:
            doc = " ".join(buffer).strip()
            n_total += 1
            if not is_low_quality(doc, min_words=min_words):
                doc_hash = hashlib.md5(doc.encode("utf-8")).hexdigest()
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    fout.write(doc + "\n\n")
                    n_kept += 1
                else:
                    n_dup += 1
            else:
                n_low_quality += 1
 
    print(f"Total: {n_total} | Kept: {n_kept} | Low quality: {n_low_quality} | Duplicates: {n_dup}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/corpus.txt")
    parser.add_argument("--output", type=str, default="data/processed/corpus_clean.txt")
    parser.add_argument("--min_words", type=int, default=20)
    args = parser.parse_args()
 
    clean_corpus(args.input, args.output, args.min_words)
