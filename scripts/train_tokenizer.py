"""
LIVO Tokenizer Training Script
Train a BPE tokenizer on any dataset, then save it for model training.

Usage:
    # Train on TinyStories (default)
    python scripts/train_tokenizer.py

    # Train on any HuggingFace dataset
    python scripts/train_tokenizer.py --dataset wikitext --config wikitext-2-raw-v1

    # Train on local text files
    python scripts/train_tokenizer.py --local-dir ./my_texts/

    # Custom vocab size
    python scripts/train_tokenizer.py --vocab-size 8192 --num-samples 50000
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tokenizer import livorator
from utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train livorator BPE tokenizer")
    parser.add_argument(
        "--dataset",
        default="roneneldan/TinyStories",
        help="HuggingFace dataset name (default: roneneldan/TinyStories)",
    )
    parser.add_argument("--config", default=None, help="HuggingFace dataset config")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--text-column", default="text", help="Column name for text")
    parser.add_argument("--local-dir", default=None, help="Path to local files (.txt, .md, .json, .jsonl, .csv, .parquet)")
    parser.add_argument("--vocab-size", type=int, default=16384, help="Target vocabulary size")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of texts to train on")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "configs" / "tokenizer.json"),
        help="Output path for trained tokenizer",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    return parser.parse_args()


def load_texts(args, logger) -> list:
    """Load text corpus from HuggingFace or local files (.txt, .md, .json, .jsonl, .csv, .parquet)."""
    if args.local_dir:
        import json as json_module

        logger.info("Loading local texts from: %s", args.local_dir)
        path = Path(args.local_dir)
        col = args.text_column

        SUPPORTED = {".txt", ".md", ".json", ".jsonl", ".csv", ".parquet"}

        if path.is_file():
            files = [path]
        else:
            files = sorted(f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED and f.is_file())

        if not files:
            raise FileNotFoundError(
                f"No supported files in: {args.local_dir}\n"
                f"Supported: {', '.join(SUPPORTED)}"
            )

        texts = []
        for f in files:
            ext = f.suffix.lower()
            try:
                if ext in (".txt", ".md"):
                    content = f.read_text(encoding="utf-8", errors="replace")
                    texts.extend(p.strip() for p in content.split("\n\n") if p.strip())

                elif ext == ".json":
                    raw = json_module.loads(f.read_text(encoding="utf-8", errors="replace"))
                    if isinstance(raw, list):
                        for item in raw:
                            if isinstance(item, dict) and col in item:
                                texts.append(str(item[col]))
                            elif isinstance(item, str):
                                texts.append(item)
                    elif isinstance(raw, dict) and col in raw:
                        if isinstance(raw[col], list):
                            texts.extend(str(x) for x in raw[col])
                        else:
                            texts.append(str(raw[col]))

                elif ext == ".jsonl":
                    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        obj = json_module.loads(line)
                        if isinstance(obj, dict) and col in obj:
                            texts.append(str(obj[col]))
                        elif isinstance(obj, str):
                            texts.append(obj)

                elif ext == ".csv":
                    import csv
                    with open(f, "r", encoding="utf-8", errors="replace", newline="") as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            if row.get(col):
                                texts.append(row[col])

                elif ext == ".parquet":
                    import pandas as pd
                    df = pd.read_parquet(f)
                    if col in df.columns:
                        texts.extend(str(v) for v in df[col].dropna().tolist())

            except Exception as e:
                logger.warning("Skipping %s: %s", f.name, e)
                continue

            if len(texts) >= args.num_samples:
                break

        texts = texts[: args.num_samples]
        logger.info("Loaded %d texts from local files", len(texts))
        return texts

    else:
        logger.info("Loading dataset: %s (split=%s)", args.dataset, args.split)
        from datasets import load_dataset

        kwargs = {"split": args.split}
        if args.config:
            kwargs["name"] = args.config

        dataset = load_dataset(args.dataset, **kwargs)

        # Validate column
        if args.text_column not in dataset.column_names:
            available = ", ".join(dataset.column_names)
            raise ValueError(
                f"Column '{args.text_column}' not found. Available: {available}"
            )

        n = min(args.num_samples, len(dataset))
        texts = [dataset[i][args.text_column] for i in range(n)]
        logger.info("Loaded %d texts from %s", len(texts), args.dataset)
        return texts


def main() -> None:
    args = parse_args()
    logger = get_logger("livo")

    # 1. Load corpus
    texts = load_texts(args, logger)

    # 2. Train tokenizer
    logger.info("Training tokenizer (vocab_size=%d)...", args.vocab_size)
    tokenizer = livorator(vocab_size=args.vocab_size, max_length=args.max_length)
    tokenizer.train(texts, verbose=True)

    # 3. Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    logger.info("Tokenizer saved to: %s", output_path)

    # 4. Quick test
    test_text = texts[0][:100] if texts else "Hello world"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    logger.info("Test encode/decode:")
    logger.info("  Input:   %s", test_text)
    logger.info("  Tokens:  %d ids", len(tokens))
    logger.info("  Decoded: %s", decoded)


if __name__ == "__main__":
    main()
