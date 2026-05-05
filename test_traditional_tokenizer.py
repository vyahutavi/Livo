"""
Pytest unit tests for the Traditional Tokenizer (Word-level & Character-level).

Covers:
  - Initialization & defaults
  - Encoding & decoding round-trips
  - Special tokens handling
  - Vocabulary building
  - Truncation & padding
  - Tensor encoding
  - Save / load persistence
  - Factory function routing
  - Edge cases & error handling
"""
import json
import os
import pytest

from data.traditional_tokenizer import (
    BaseTraditionalTokenizer,
    WordTokenizer,
    CharTokenizer,
    get_traditional_tokenizer,
)
from data.tokenizer import get_tokenizer


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

SAMPLE_CORPUS = [
    "Once upon a time there was a little girl named Lily.",
    "She loved to play in the garden with her friends.",
    "The sun was shining and the birds were singing.",
    "Lily picked a beautiful flower and gave it to her mom.",
    "Her mom smiled and said thank you very much.",
]


@pytest.fixture
def word_tok():
    """Word tokenizer with vocab built from sample corpus."""
    tok = WordTokenizer(vocab_size=500, max_length=128, verbose=False)
    tok.build_vocab(SAMPLE_CORPUS)
    return tok


@pytest.fixture
def char_tok():
    """Character tokenizer with vocab built from sample corpus."""
    tok = CharTokenizer(vocab_size=500, max_length=256, verbose=False)
    tok.build_vocab(SAMPLE_CORPUS)
    return tok


@pytest.fixture
def tmp_path_json(tmp_path):
    """Provide a temporary JSON file path."""
    return str(tmp_path / "tokenizer.json")


# ================================================================== #
#  WORD TOKENIZER TESTS                                                #
# ================================================================== #


class TestWordTokenizerDefaults:
    """Test initialization and default values."""

    def test_default_vocab_size(self):
        tok = WordTokenizer()
        assert tok.vocab_size == 50000

    def test_default_max_length(self):
        tok = WordTokenizer()
        assert tok.max_length == 1024

    def test_special_tokens_registered(self):
        tok = WordTokenizer()
        assert tok.pad_token == 0
        assert tok.unk_token == 1
        assert tok.bos_token == 2
        assert tok.eos_token == 3

    def test_custom_vocab_size(self):
        tok = WordTokenizer(vocab_size=1000)
        assert tok.vocab_size == 1000

    def test_min_vocab_floor(self):
        tok = WordTokenizer(vocab_size=1)
        assert tok.vocab_size >= len(WordTokenizer.SPECIAL_TOKENS)


class TestWordTokenizerEncodeDecode:
    """Test encoding and decoding."""

    def test_round_trip(self, word_tok):
        text = "Lily loved the garden."
        ids = word_tok.encode(text)
        decoded = word_tok.decode(ids)
        assert decoded == text

    def test_bos_eos_added(self, word_tok):
        ids = word_tok.encode("hello")
        assert ids[0] == word_tok.bos_token
        assert ids[-1] == word_tok.eos_token

    def test_no_special_tokens(self, word_tok):
        ids = word_tok.encode("hello", add_special_tokens=False)
        assert ids[0] != word_tok.bos_token
        assert ids[-1] != word_tok.eos_token

    def test_unknown_word(self, word_tok):
        ids = word_tok.encode("xylophone")
        assert word_tok.unk_token in ids

    def test_unk_decoded_as_question_mark(self, word_tok):
        ids = word_tok.encode("xylophone")
        decoded = word_tok.decode(ids)
        assert "?" in decoded

    def test_empty_string(self, word_tok):
        ids = word_tok.encode("")
        assert ids == [word_tok.bos_token, word_tok.eos_token]

    def test_decode_skip_special(self, word_tok):
        ids = [word_tok.bos_token, word_tok.pad_token, word_tok.eos_token]
        decoded = word_tok.decode(ids, skip_special_tokens=True)
        assert "<bos>" not in decoded
        assert "<eos>" not in decoded
        assert "<pad>" not in decoded

    def test_decode_keep_special(self, word_tok):
        ids = [word_tok.bos_token, word_tok.eos_token]
        decoded = word_tok.decode(ids, skip_special_tokens=False)
        assert "<bos>" in decoded
        assert "<eos>" in decoded


class TestWordTokenizerTruncation:
    """Test truncation behavior."""

    def test_truncation(self):
        tok = WordTokenizer(vocab_size=500, max_length=10)
        tok.build_vocab(SAMPLE_CORPUS)
        long_text = " ".join(["word"] * 100)
        ids = tok.encode(long_text)
        assert len(ids) <= 10
        assert ids[-1] == tok.eos_token

    def test_no_truncation(self):
        tok = WordTokenizer(vocab_size=500, max_length=10)
        tok.build_vocab(SAMPLE_CORPUS)
        long_text = " ".join(["word"] * 100)
        ids = tok.encode(long_text, truncate=False)
        assert len(ids) > 10


class TestWordTokenizerVocab:
    """Test vocabulary building."""

    def test_vocab_built(self, word_tok):
        assert word_tok.actual_vocab_size > len(WordTokenizer.SPECIAL_TOKENS)

    def test_vocab_respects_limit(self):
        tok = WordTokenizer(vocab_size=10)
        tok.build_vocab(SAMPLE_CORPUS)
        assert word_tok_actual_size_ok(tok)

    def test_min_freq_filter(self):
        tok = WordTokenizer(vocab_size=500, min_freq=99)
        tok.build_vocab(SAMPLE_CORPUS)
        # With min_freq=99, only very frequent words survive
        assert tok.actual_vocab_size <= len(WordTokenizer.SPECIAL_TOKENS) + 5

    def test_lowercase_mode(self):
        tok = WordTokenizer(vocab_size=500, lowercase=True)
        tok.build_vocab(["Hello WORLD hello"])
        ids = tok.encode("HELLO")
        decoded = tok.decode(ids)
        assert "hello" in decoded

    def test_vocab_override_in_build(self):
        tok = WordTokenizer(vocab_size=500)
        tok.build_vocab(SAMPLE_CORPUS, vocab_size=10)
        assert tok.vocab_size == 10


def word_tok_actual_size_ok(tok):
    return tok.actual_vocab_size <= tok.vocab_size


class TestWordTokenizerErrors:
    """Test error handling."""

    def test_encode_none_raises(self, word_tok):
        with pytest.raises(ValueError, match="cannot be None"):
            word_tok.encode(None)

    def test_encode_non_string_raises(self, word_tok):
        with pytest.raises(ValueError, match="must be a string"):
            word_tok.encode(42)

    def test_decode_none_raises(self, word_tok):
        with pytest.raises(ValueError, match="cannot be None"):
            word_tok.decode(None)

    def test_decode_non_list_raises(self, word_tok):
        with pytest.raises(ValueError, match="must be a list"):
            word_tok.decode("not a list")


# ================================================================== #
#  CHARACTER TOKENIZER TESTS                                           #
# ================================================================== #


class TestCharTokenizerDefaults:
    """Test CharTokenizer initialization."""

    def test_default_vocab_size(self):
        tok = CharTokenizer()
        assert tok.vocab_size == 50000

    def test_ascii_pre_populated(self):
        tok = CharTokenizer()
        # Printable ASCII should be in vocab
        assert "A" in tok.token_to_id
        assert "z" in tok.token_to_id
        assert "0" in tok.token_to_id
        assert "!" in tok.token_to_id

    def test_default_max_length(self):
        tok = CharTokenizer()
        assert tok.max_length == 2048


class TestCharTokenizerEncodeDecode:
    """Test CharTokenizer encoding and decoding."""

    def test_round_trip(self, char_tok):
        text = "Lily loved the garden."
        ids = char_tok.encode(text)
        decoded = char_tok.decode(ids)
        assert decoded == text

    def test_bos_eos(self, char_tok):
        ids = char_tok.encode("A")
        assert ids[0] == char_tok.bos_token
        assert ids[-1] == char_tok.eos_token
        assert len(ids) == 3  # BOS + 'A' + EOS

    def test_each_char_is_token(self, char_tok):
        text = "abc"
        ids = char_tok.encode(text, add_special_tokens=False)
        assert len(ids) == 3

    def test_empty_string(self, char_tok):
        ids = char_tok.encode("")
        assert ids == [char_tok.bos_token, char_tok.eos_token]

    def test_spaces_preserved(self, char_tok):
        text = "a b"
        ids = char_tok.encode(text)
        decoded = char_tok.decode(ids)
        assert decoded == text

    def test_unicode_chars(self, char_tok):
        # Unicode chars not in vocab should become <unk>
        char_tok_small = CharTokenizer(vocab_size=100)
        ids = char_tok_small.encode("日本")
        # Should still produce valid output
        assert len(ids) > 0


class TestCharTokenizerVocab:
    """Test CharTokenizer vocabulary building."""

    def test_corpus_extends_vocab(self):
        tok = CharTokenizer(vocab_size=500)
        size_before = tok.actual_vocab_size
        tok.build_vocab(["αβγδ"])  # Greek chars
        assert tok.actual_vocab_size > size_before

    def test_lowercase_mode(self):
        tok = CharTokenizer(vocab_size=500, lowercase=True)
        tok.build_vocab(["HELLO"])
        ids = tok.encode("HELLO")
        decoded = tok.decode(ids)
        assert decoded == "hello"


# ================================================================== #
#  PERSISTENCE TESTS                                                   #
# ================================================================== #


class TestSaveLoad:
    """Test save/load for both tokenizers."""

    def test_word_save_load(self, word_tok, tmp_path_json):
        text = "Lily loved the garden."
        ids_before = word_tok.encode(text)
        word_tok.save(tmp_path_json)

        loaded = WordTokenizer.load(tmp_path_json)
        ids_after = loaded.encode(text)
        assert ids_before == ids_after

    def test_char_save_load(self, char_tok, tmp_path_json):
        text = "Hello world!"
        ids_before = char_tok.encode(text)
        char_tok.save(tmp_path_json)

        loaded = CharTokenizer.load(tmp_path_json)
        ids_after = loaded.encode(text)
        assert ids_before == ids_after

    def test_save_creates_file(self, word_tok, tmp_path_json):
        word_tok.save(tmp_path_json)
        assert os.path.exists(tmp_path_json)

    def test_saved_json_valid(self, word_tok, tmp_path_json):
        word_tok.save(tmp_path_json)
        with open(tmp_path_json, "r") as f:
            data = json.load(f)
        assert "type" in data
        assert "vocab_size" in data
        assert "vocab" in data

    def test_load_preserves_type(self, tmp_path_json):
        tok = WordTokenizer(vocab_size=100)
        tok.build_vocab(["hello world"])
        tok.save(tmp_path_json)

        loaded = BaseTraditionalTokenizer.load(tmp_path_json)
        assert isinstance(loaded, WordTokenizer)

    def test_load_char_preserves_type(self, tmp_path_json):
        tok = CharTokenizer(vocab_size=200)
        tok.save(tmp_path_json)

        loaded = BaseTraditionalTokenizer.load(tmp_path_json)
        assert isinstance(loaded, CharTokenizer)


# ================================================================== #
#  TENSOR ENCODING TESTS                                               #
# ================================================================== #


class TestTensorEncoding:
    """Test encode_to_tensor for both tokenizers."""

    def test_word_tensor_shape(self, word_tok):
        tensor = word_tok.encode_to_tensor("hello world")
        assert tensor.shape == (1, word_tok.max_length)

    def test_char_tensor_shape(self, char_tok):
        tensor = char_tok.encode_to_tensor("hello")
        assert tensor.shape == (1, char_tok.max_length)

    def test_tensor_dtype(self, word_tok):
        import torch
        tensor = word_tok.encode_to_tensor("hello")
        assert tensor.dtype == torch.long

    def test_tensor_padding(self, word_tok):
        tensor = word_tok.encode_to_tensor("hi")
        # Most of the tensor should be padding
        assert (tensor[0] == word_tok.pad_token).sum().item() > 0


# ================================================================== #
#  FACTORY FUNCTION TESTS                                              #
# ================================================================== #


class TestFactoryFunction:
    """Test get_traditional_tokenizer and get_tokenizer routing."""

    def test_word_type(self):
        tok = get_traditional_tokenizer("word")
        assert isinstance(tok, WordTokenizer)

    def test_char_type(self):
        tok = get_traditional_tokenizer("char")
        assert isinstance(tok, CharTokenizer)

    def test_word_aliases(self):
        for alias in ["word", "words", "word_level"]:
            tok = get_traditional_tokenizer(alias)
            assert isinstance(tok, WordTokenizer)

    def test_char_aliases(self):
        for alias in ["char", "character", "char_level"]:
            tok = get_traditional_tokenizer(alias)
            assert isinstance(tok, CharTokenizer)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            get_traditional_tokenizer("invalid")

    def test_get_tokenizer_routes_word(self):
        tok = get_tokenizer("word")
        assert isinstance(tok, WordTokenizer)

    def test_get_tokenizer_routes_char(self):
        tok = get_tokenizer("char")
        assert isinstance(tok, CharTokenizer)

    def test_get_tokenizer_auto_still_works(self):
        from data.tokenizer import livorator
        tok = get_tokenizer("auto")
        assert isinstance(tok, livorator)

    def test_kwargs_passed_through(self):
        tok = get_traditional_tokenizer("word", vocab_size=999, max_length=64)
        assert tok.vocab_size == 999
        assert tok.max_length == 64
