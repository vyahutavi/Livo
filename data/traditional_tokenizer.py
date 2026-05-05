"""
Traditional Tokenizer for LIVO — Word-level & Character-level tokenization.

Implements classic NLP tokenization approaches:
  - WordTokenizer: whitespace + punctuation splitting with frequency-based vocabulary
  - CharTokenizer: single-character tokenization

Both follow the same interface as livorator for drop-in compatibility
with LIVO's TextDataset, training pipeline, and model.

Usage:
    from data.traditional_tokenizer import WordTokenizer, CharTokenizer

    # Word-level
    tok = WordTokenizer(vocab_size=50000)
    tok.build_vocab(["Hello world!", "The cat sat on the mat."])
    ids = tok.encode("Hello world!")

    # Character-level
    tok = CharTokenizer()
    ids = tok.encode("Hello world!")
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Set

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ------------------------------------------------------------------ #
#  Base class — shared interface for all traditional tokenizers        #
# ------------------------------------------------------------------ #

class BaseTraditionalTokenizer:
    """
    Abstract base for traditional tokenizers.

    Provides the same public API as livorator so it can be used
    as a drop-in replacement in TextDataset and training scripts.
    """

    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }

    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 1024,
        lowercase: bool = False,
        verbose: bool = False,
    ):
        min_vocab = len(self.SPECIAL_TOKENS)
        self.vocab_size = max(int(vocab_size), min_vocab)
        self.max_length = max_length
        self.lowercase = lowercase
        self.verbose = verbose

        # Special token IDs
        self.pad_token = self.SPECIAL_TOKENS["<pad>"]
        self.unk_token = self.SPECIAL_TOKENS["<unk>"]
        self.bos_token = self.SPECIAL_TOKENS["<bos>"]
        self.eos_token = self.SPECIAL_TOKENS["<eos>"]

        # Vocab maps (built by subclass)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.actual_vocab_size: int = len(self.SPECIAL_TOKENS)

        # Always register special tokens
        for name, idx in self.SPECIAL_TOKENS.items():
            self.token_to_id[name] = idx
            self.id_to_token[idx] = name

    # ---- public interface (matches livorator) ---- #

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncate: bool = True,
    ) -> List[int]:
        """Encode text to token IDs."""
        if text is None:
            raise ValueError("text cannot be None.")
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text).__name__}.")

        tokens = self._tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_token) for t in tokens]

        if add_special_tokens:
            ids = [self.bos_token] + ids + [self.eos_token]

        if truncate and len(ids) > self.max_length:
            ids = ids[: self.max_length - 1] + [self.eos_token]

        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if token_ids is None:
            raise ValueError("token_ids cannot be None.")
        if not isinstance(token_ids, list):
            raise ValueError(f"token_ids must be a list, got {type(token_ids).__name__}.")

        special = {self.pad_token, self.bos_token, self.eos_token}
        parts: List[str] = []

        for tid in token_ids:
            if tid == self.unk_token:
                parts.append("?" if skip_special_tokens else "<unk>")
                continue
            if tid in special:
                if skip_special_tokens:
                    continue
                parts.append(self.id_to_token.get(tid, "<unk>"))
                continue
            parts.append(self.id_to_token.get(tid, "?"))

        return self._detokenize(parts)

    def encode_to_tensor(self, text: str, device: str = "cpu") -> "torch.Tensor":
        """Encode text to a padded tensor of shape (1, max_length)."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for tensor encoding.")

        ids = self.encode(text)
        if len(ids) < self.max_length:
            ids += [self.pad_token] * (self.max_length - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=device)

    # ---- persistence ---- #

    def save(self, path: str) -> None:
        """Save tokenizer vocabulary to JSON."""
        data = {
            "type": self.__class__.__name__,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "lowercase": self.lowercase,
            "vocab": {tok: idx for tok, idx in self.token_to_id.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if self.verbose:
            print(f"[OK] Tokenizer saved to {path} ({self.actual_vocab_size:,} tokens)")

    @classmethod
    def load(cls, path: str, **kwargs) -> "BaseTraditionalTokenizer":
        """Load a saved tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        class_name = data.get("type", cls.__name__)

        # Resolve subclass
        subclass_map = {
            "WordTokenizer": WordTokenizer,
            "CharTokenizer": CharTokenizer,
        }
        target_cls = subclass_map.get(class_name, cls)

        obj = target_cls(
            vocab_size=kwargs.pop("vocab_size", data["vocab_size"]),
            max_length=kwargs.pop("max_length", data["max_length"]),
            lowercase=kwargs.pop("lowercase", data.get("lowercase", False)),
            **kwargs,
        )

        # Rebuild vocab maps
        obj.token_to_id = {}
        obj.id_to_token = {}
        for tok, idx in data["vocab"].items():
            obj.token_to_id[tok] = int(idx)
            obj.id_to_token[int(idx)] = tok
        obj.actual_vocab_size = len(obj.token_to_id)

        return obj

    # ---- abstract methods (subclass must implement) ---- #

    def _tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    def _detokenize(self, tokens: List[str]) -> str:
        raise NotImplementedError


# ------------------------------------------------------------------ #
#  Word-level Tokenizer                                               #
# ------------------------------------------------------------------ #

_WORD_SPLIT_RE = re.compile(r"""
    (?:[A-Za-z]+'[a-z]+)     |   # contractions: don't, it's
    (?:\d+[.,]?\d*)          |   # numbers: 42, 3.14, 1,000
    (?:[A-Za-z]+)            |   # alphabetic words
    (?:[^\s])                    # individual punctuation / symbols
""", re.VERBOSE | re.UNICODE)


class WordTokenizer(BaseTraditionalTokenizer):
    """
    Traditional word-level tokenizer.

    Splits text on whitespace and punctuation boundaries, builds
    a frequency-based vocabulary from a training corpus. Unknown
    words map to <unk>.

    Example:
        tok = WordTokenizer(vocab_size=30000)
        tok.build_vocab(["The cat sat.", "The dog ran."])
        ids = tok.encode("The cat ran.")
        print(tok.decode(ids))  # "The cat ran ."
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 1024,
        lowercase: bool = False,
        min_freq: int = 1,
        verbose: bool = False,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_length=max_length,
            lowercase=lowercase,
            verbose=verbose,
        )
        self.min_freq = min_freq

    def build_vocab(self, corpus: List[str], vocab_size: Optional[int] = None) -> None:
        """
        Build vocabulary from a list of text strings.

        Args:
            corpus: List of training texts
            vocab_size: Override vocabulary size (optional)
        """
        if vocab_size is not None:
            self.vocab_size = max(int(vocab_size), len(self.SPECIAL_TOKENS))

        if self.verbose:
            print(f"[VOCAB] Building vocabulary from {len(corpus):,} texts...")

        # Count word frequencies
        counter: Counter = Counter()
        for text in corpus:
            tokens = self._tokenize(text)
            counter.update(tokens)

        # Filter by min_freq and take top vocab_size
        available_slots = self.vocab_size - len(self.SPECIAL_TOKENS)
        most_common = [
            (word, freq) for word, freq in counter.most_common()
            if freq >= self.min_freq
        ][:available_slots]

        # Build maps
        self.token_to_id = {}
        self.id_to_token = {}

        for name, idx in self.SPECIAL_TOKENS.items():
            self.token_to_id[name] = idx
            self.id_to_token[idx] = name

        next_id = len(self.SPECIAL_TOKENS)
        for word, _ in most_common:
            if word not in self.token_to_id:
                self.token_to_id[word] = next_id
                self.id_to_token[next_id] = word
                next_id += 1

        self.actual_vocab_size = len(self.token_to_id)

        if self.verbose:
            print(f"[VOCAB] Built vocabulary: {self.actual_vocab_size:,} tokens")
            print(f"[VOCAB] Unique words in corpus: {len(counter):,}")
            print(f"[VOCAB] Coverage: {self.actual_vocab_size - len(self.SPECIAL_TOKENS):,}"
                  f" / {len(counter):,} words")

    def _tokenize(self, text: str) -> List[str]:
        """Split text into word tokens using regex."""
        if self.lowercase:
            text = text.lower()
        # Normalize unicode (e.g. accented chars)
        text = unicodedata.normalize("NFC", text)
        return _WORD_SPLIT_RE.findall(text)

    def _detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from word tokens with smart spacing."""
        if not tokens:
            return ""

        # Simple heuristic: no space before punctuation
        no_space_before = set(".,!?;:)]}\"'-")
        no_space_after = set("([{\"'-")

        parts = [tokens[0]]
        for i in range(1, len(tokens)):
            tok = tokens[i]
            prev = tokens[i - 1]

            if tok in no_space_before or prev in no_space_after:
                parts.append(tok)
            else:
                parts.append(" " + tok)

        return "".join(parts)


# ------------------------------------------------------------------ #
#  Character-level Tokenizer                                          #
# ------------------------------------------------------------------ #

class CharTokenizer(BaseTraditionalTokenizer):
    """
    Traditional character-level tokenizer.

    Each Unicode character becomes one token. Vocabulary is built
    automatically from all printable ASCII + any characters seen
    in the training corpus.

    Example:
        tok = CharTokenizer()
        tok.build_vocab(["Hello world!"])
        ids = tok.encode("Hello!")
        print(tok.decode(ids))  # "Hello!"
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 2048,
        lowercase: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_length=max_length,
            lowercase=lowercase,
            verbose=verbose,
        )
        # Auto-build with printable ASCII by default
        self._build_ascii_vocab()

    def _build_ascii_vocab(self) -> None:
        """Initialize with printable ASCII characters."""
        next_id = len(self.SPECIAL_TOKENS)
        for code in range(32, 127):  # printable ASCII
            char = chr(code)
            if char not in self.token_to_id and next_id < self.vocab_size:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

        # Add common whitespace
        for char in ["\n", "\t", "\r"]:
            if char not in self.token_to_id and next_id < self.vocab_size:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

        self.actual_vocab_size = len(self.token_to_id)

    def build_vocab(self, corpus: List[str], vocab_size: Optional[int] = None) -> None:
        """
        Extend vocabulary with characters from corpus.

        Args:
            corpus: List of training texts
            vocab_size: Override vocabulary size (optional)
        """
        if vocab_size is not None:
            self.vocab_size = max(int(vocab_size), len(self.SPECIAL_TOKENS))

        if self.verbose:
            print(f"[VOCAB] Scanning {len(corpus):,} texts for characters...")

        # Collect all unique characters
        char_freq: Counter = Counter()
        for text in corpus:
            if self.lowercase:
                text = text.lower()
            char_freq.update(text)

        # Add new characters by frequency
        next_id = max(self.id_to_token.keys(), default=-1) + 1
        for char, _ in char_freq.most_common():
            if char not in self.token_to_id and next_id < self.vocab_size:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

        self.actual_vocab_size = len(self.token_to_id)

        if self.verbose:
            print(f"[VOCAB] Character vocabulary: {self.actual_vocab_size:,} tokens")

    def _tokenize(self, text: str) -> List[str]:
        """Split text into individual characters."""
        if self.lowercase:
            text = text.lower()
        return list(text)

    def _detokenize(self, tokens: List[str]) -> str:
        """Join characters back into text."""
        return "".join(tokens)


# ------------------------------------------------------------------ #
#  Factory function                                                    #
# ------------------------------------------------------------------ #

def get_traditional_tokenizer(
    tokenizer_type: str = "word",
    tokenizer_path: Optional[str] = None,
    **kwargs,
) -> BaseTraditionalTokenizer:
    """
    Get a traditional tokenizer instance.

    Args:
        tokenizer_type: "word" or "char"
        tokenizer_path: Path to saved tokenizer JSON (loads existing vocab)
        **kwargs: Passed to tokenizer constructor

    Returns:
        WordTokenizer or CharTokenizer instance
    """
    tokenizer_type = tokenizer_type.lower()

    if tokenizer_path is not None:
        return BaseTraditionalTokenizer.load(tokenizer_path, **kwargs)

    if tokenizer_type in {"word", "words", "word_level"}:
        return WordTokenizer(**kwargs)

    if tokenizer_type in {"char", "character", "char_level"}:
        return CharTokenizer(**kwargs)

    raise ValueError(
        f"Unsupported tokenizer_type '{tokenizer_type}'. "
        "Expected: 'word' or 'char'."
    )


# ------------------------------------------------------------------ #
#  Quick test                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 60)
    print("Traditional Tokenizer Test")
    print("=" * 60)

    corpus = [
        "Once upon a time there was a little girl named Lily.",
        "She loved to play in the garden with her friends.",
        "The sun was shining and the birds were singing.",
        "Lily picked a beautiful flower and gave it to her mom.",
        "Her mom smiled and said thank you very much.",
    ]

    # --- Word Tokenizer ---
    print("\n[Word Tokenizer]")
    wt = WordTokenizer(vocab_size=500, verbose=True)
    wt.build_vocab(corpus)

    text = "Lily loved the garden."
    ids = wt.encode(text)
    decoded = wt.decode(ids)
    print(f"  Input:   {text}")
    print(f"  Tokens:  {ids} ({len(ids)} ids)")
    print(f"  Decoded: {decoded}")

    # Save / load round-trip
    wt.save("_test_word_tok.json")
    wt2 = WordTokenizer.load("_test_word_tok.json")
    ids2 = wt2.encode(text)
    assert ids == ids2, f"Round-trip failed: {ids} != {ids2}"
    print("  ✓ Save/load round-trip passed!")

    # --- Char Tokenizer ---
    print("\n[Character Tokenizer]")
    ct = CharTokenizer(verbose=True)
    ct.build_vocab(corpus)

    ids = ct.encode(text)
    decoded = ct.decode(ids)
    print(f"  Input:   {text}")
    print(f"  Tokens:  {ids} ({len(ids)} ids)")
    print(f"  Decoded: {decoded}")

    # Save / load round-trip
    ct.save("_test_char_tok.json")
    ct2 = CharTokenizer.load("_test_char_tok.json")
    ids2 = ct2.encode(text)
    assert ids == ids2, f"Round-trip failed: {ids} != {ids2}"
    print("  ✓ Save/load round-trip passed!")

    # Cleanup
    import os
    os.remove("_test_word_tok.json")
    os.remove("_test_char_tok.json")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
