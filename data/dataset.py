"""
Generic text dataset for LIVO — works with any HuggingFace dataset or local text files.

Supports:
  - Any HuggingFace dataset: "roneneldan/TinyStories", "wikitext", "openwebtext", etc.
  - Local text files: .txt files from a directory
  - Pre-trained tokenizer loading from JSON
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from data.tokenizer import livorator


class TextDataset(Dataset):
    """
    Generic PyTorch Dataset for causal language modeling.
    
    Works with any text source — HuggingFace datasets or local files.
    Tokenizes on-the-fly using livorator.

    Examples:
        # HuggingFace dataset
        ds = TextDataset(source="roneneldan/TinyStories", split="train")

        # Different HuggingFace dataset
        ds = TextDataset(source="wikitext", config="wikitext-2-raw-v1", split="train")

        # Local text files
        ds = TextDataset(source="./my_data/", local=True)

        # With a trained tokenizer
        ds = TextDataset(source="roneneldan/TinyStories", tokenizer_path="tokenizer.json")
    """

    def __init__(
        self,
        source: str = "roneneldan/TinyStories",
        config: Optional[str] = None,
        split: str = "train",
        text_column: str = "text",
        max_length: int = 512,
        vocab_size: int = 16384,
        tokenizer: Optional[livorator] = None,
        tokenizer_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        local: bool = False,
    ):
        """
        Args:
            source: HuggingFace dataset name OR path to local directory of .txt files
            config: HuggingFace dataset config name (e.g., "wikitext-2-raw-v1")
            split: Dataset split — "train", "validation", "test"
            text_column: Name of the text column in the dataset
            max_length: Maximum sequence length after tokenization
            vocab_size: Vocabulary size (ignored if tokenizer/tokenizer_path provided)
            tokenizer: Pre-built livorator instance (optional)
            tokenizer_path: Path to saved tokenizer JSON (optional)
            max_samples: Limit number of samples (for debugging)
            local: If True, treat `source` as a local directory path
        """
        self.max_length = max_length
        self.text_column = text_column

        # Setup tokenizer (priority: instance > path > new)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_path is not None:
            self.tokenizer = livorator.load(tokenizer_path)
        else:
            self.tokenizer = livorator(vocab_size=vocab_size, max_length=max_length)

        # Load data
        if local:
            self.data = self._load_local(source, max_samples)
        else:
            self.data = self._load_huggingface(source, config, split, max_samples)

    def _load_huggingface(
        self,
        dataset_name: str,
        config: Optional[str],
        split: str,
        max_samples: Optional[int],
    ) -> list:
        """Load from any HuggingFace dataset."""
        from datasets import load_dataset

        kwargs = {"split": split}
        if config:
            kwargs["name"] = config

        dataset = load_dataset(dataset_name, **kwargs)

        # Validate text column exists
        if self.text_column not in dataset.column_names:
            available = ", ".join(dataset.column_names)
            raise ValueError(
                f"Column '{self.text_column}' not found in dataset '{dataset_name}'. "
                f"Available columns: {available}. "
                f"Set text_column to the correct column name."
            )

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return dataset

    def _load_local(self, directory: str, max_samples: Optional[int]) -> List[Dict[str, str]]:
        """
        Load from local files in a directory.

        Supported formats:
            .txt, .md    — plain text, split by paragraphs (double newline)
            .json        — JSON array of objects OR single object with a text field
            .jsonl       — JSON Lines (one JSON object per line)
            .csv         — CSV with a text column header
            .parquet     — Parquet files with a text column
        """
        import json as json_module

        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {directory}")

        SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".csv", ".parquet"}

        # Collect files
        if path.is_file():
            files = [path]
        else:
            files = sorted(
                f for f in path.rglob("*")
                if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
            )

        if not files:
            ext_list = ", ".join(SUPPORTED_EXTENSIONS)
            raise FileNotFoundError(
                f"No supported files found in: {directory}\n"
                f"Supported extensions: {ext_list}"
            )

        texts: List[Dict[str, str]] = []
        col = self.text_column

        for file_path in files:
            ext = file_path.suffix.lower()

            try:
                if ext in (".txt", ".md"):
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                    for para in paragraphs:
                        texts.append({col: para})

                elif ext == ".json":
                    raw = json_module.loads(
                        file_path.read_text(encoding="utf-8", errors="replace")
                    )
                    if isinstance(raw, list):
                        # Array of objects: [{"text": "..."}, ...]
                        for item in raw:
                            if isinstance(item, dict) and col in item:
                                texts.append({col: str(item[col])})
                            elif isinstance(item, str):
                                texts.append({col: item})
                    elif isinstance(raw, dict):
                        # Single object with a text field or a field containing a list
                        if col in raw and isinstance(raw[col], str):
                            texts.append({col: raw[col]})
                        elif col in raw and isinstance(raw[col], list):
                            for item in raw[col]:
                                texts.append({col: str(item)})

                elif ext == ".jsonl":
                    for line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        obj = json_module.loads(line)
                        if isinstance(obj, dict) and col in obj:
                            texts.append({col: str(obj[col])})
                        elif isinstance(obj, str):
                            texts.append({col: obj})

                elif ext == ".csv":
                    import csv
                    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
                        reader = csv.DictReader(f)
                        if col not in (reader.fieldnames or []):
                            available = ", ".join(reader.fieldnames or [])
                            raise ValueError(
                                f"Column '{col}' not in CSV '{file_path.name}'. "
                                f"Available: {available}"
                            )
                        for row in reader:
                            if row.get(col):
                                texts.append({col: row[col]})

                elif ext == ".parquet":
                    import pandas as pd
                    df = pd.read_parquet(file_path)
                    if col not in df.columns:
                        available = ", ".join(df.columns.tolist())
                        raise ValueError(
                            f"Column '{col}' not in Parquet '{file_path.name}'. "
                            f"Available: {available}"
                        )
                    for val in df[col].dropna().tolist():
                        texts.append({col: str(val)})

            except (json_module.JSONDecodeError, ValueError) as e:
                # Skip malformed files with a warning
                import warnings
                warnings.warn(f"Skipping {file_path.name}: {e}")
                continue

            if max_samples and len(texts) >= max_samples:
                texts = texts[:max_samples]
                break

        if not texts:
            raise ValueError(
                f"No texts extracted from {len(files)} file(s) in '{directory}'. "
                f"Check that your files contain a '{col}' field/column."
            )

        return texts

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx][self.text_column]

        # Tokenize with BOS/EOS
        token_ids = self.tokenizer.encode(text, add_special_tokens=True, truncate=True)

        # Pad to max_length
        pad_len = self.max_length - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * max(0, pad_len)
        token_ids = token_ids + [self.tokenizer.pad_token] * max(0, pad_len)

        # Truncate to exactly max_length
        token_ids = token_ids[: self.max_length]
        attention_mask = attention_mask[: self.max_length]

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask_t,
            "labels": input_ids.clone(),
        }


# Keep backward compatibility
TinyStoriesDataset = TextDataset
