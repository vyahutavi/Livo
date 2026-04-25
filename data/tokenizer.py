from collections import Counter
import heapq
from typing import Any, Dict, List, Optional, Tuple


try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False 

class livorator:
    """
    Custom BPE Tokenizer for livorator.
    
    Implements Byte-Pair Encoding (BPE) algorithm with:
    - 16384 token vocabulary 
    - Character + subword tokenization
    - Efficient encoding/decoding
    - Spike neural encoding support
    - Designed for small, efficient models with strong generalization.
    """

    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }
    
    def __init__(
        self, 
        vocab_size: int = 16384,
        max_length: int = 512,
        bpe_merges: Optional[Dict[Tuple[str, str], int]] = None,
        model: Optional[str] = None,
        verbose: bool = False,
        **_
    ):
        """
        Initialize livorator custom tokenizer.
        
        Args:
            vocab_size: Total vocabulary size (default: 16,384 to match model config)
            max_length: Maximum sequence length (default: 512 to match model config)
            bpe_merges: Pre-computed BPE merge operations (optional)
        """
        min_vocab_size = len(self.SPECIAL_TOKENS) + 256
        self.vocab_size = max(int(vocab_size), min_vocab_size)
        self.max_length = max_length
        self.model = model
        self.verbose = verbose
        
        # Special tokens - MUST initialize first before _build_token_map()
        self.pad_token = self.SPECIAL_TOKENS["<pad>"]
        self.unk_token = self.SPECIAL_TOKENS["<unk>"]
        self.bos_token = self.SPECIAL_TOKENS["<bos>"]
        self.eos_token = self.SPECIAL_TOKENS["<eos>"]
        self.special_token_count = len(self.SPECIAL_TOKENS)
        self.byte_token_offset = self.special_token_count
        self.merge_token_offset = self.byte_token_offset + 256
        
        # Initialize vocabulary - character level (256 bytes) + merges
        self._init_vocab()
        
        # BPE merge operations
        if bpe_merges is None:
            self.bpe_merges = self._init_default_merges()
        else:
            self.bpe_merges = self._normalize_bpe_merges(bpe_merges)
        
        # Build token ID mapping
        self._build_token_map()
        
        if self.verbose:
            print("[OK] livorator Custom Tokenizer initialized")
            print(f"   Vocabulary: {self.vocab_size:,} tokens")
            print(f"   Max Length: {self.max_length}")
            print(f"   BPE Merges: {len(self.bpe_merges):,}")
            print(f"   Actual Vocabulary: {self.actual_vocab_size:,}")
    
    def _init_vocab(self):
        """Initialize base vocabulary with 256 bytes."""
        # Start with all 256 byte values
        self.byte_encoder = {
            i: bytes([i]) for i in range(256)
        }
        self.byte_decoder = {
            bytes([i]): i for i in range(256)
        }
        
        # Base vocabulary = 256 bytes + special tokens + merges
        self.base_vocab_size = 256
        if self.verbose:
            print(f"   Base vocabulary: {self.base_vocab_size} bytes")

    def _normalize_token_bytes(self, token: Any) -> bytes:
        """Normalize caller-provided merge entries into byte tokens."""
        if isinstance(token, bytes):
            return token
        if isinstance(token, bytearray):
            return bytes(token)
        if isinstance(token, int):
            if 0 <= token <= 255:
                return bytes([token])
            raise ValueError(f"Byte token integers must be in [0, 255], got {token}")
        if isinstance(token, str):
            return token.encode("utf-8")
        raise TypeError(f"Unsupported token type for BPE merge: {type(token)._name_}")

    def _normalize_bpe_merges(
        self,
        bpe_merges: Dict[Tuple[Any, Any], int]
    ) -> Dict[Tuple[bytes, bytes], int]:
        """Normalize externally supplied BPE merges into a deterministic byte map."""
        normalized: Dict[Tuple[bytes, bytes], int] = {}

        for pair, priority in bpe_merges.items():
            if len(pair) != 2:
                raise ValueError(f"Each BPE merge must contain exactly 2 tokens, got {pair!r}")
            a, b = pair
            normalized_pair = (
                self._normalize_token_bytes(a),
                self._normalize_token_bytes(b),
            )
            normalized[normalized_pair] = int(priority)

        return dict(sorted(normalized.items(), key=lambda item: item[1]))

    def _init_default_merges(self) -> Dict[Tuple[bytes, bytes], int]:
        """Build a deterministic default merge table that fills the configured vocab."""
        merge_capacity = max(0, self.vocab_size - self.merge_token_offset)
        if merge_capacity == 0:
            return {}

        seed_corpus = (
            " the and of to in a is that for it on with as are this be or by from "
            "language model transformer attention neural network data training text "
            "hello world generation reasoning sparse encoder token byte pair merge "
        ).encode("utf-8")

        ordered_pairs: List[Tuple[bytes, bytes]] = []
        pair_counts = Counter(zip(seed_corpus, seed_corpus[1:]))
        for (left, right), _ in pair_counts.most_common():
            ordered_pairs.append((bytes([left]), bytes([right])))

        preferred_bytes = list(dict.fromkeys(
            seed_corpus +
            b" etaoinshrdlcumwfgypbvkjxqETAOINSHRDL0123456789.,!?;:-_()[]{}'\"/\n\t"
        ))
        for left in preferred_bytes:
            for right in preferred_bytes:
                ordered_pairs.append((bytes([left]), bytes([right])))

        merges: Dict[Tuple[bytes, bytes], int] = {}
        seen_merged_tokens = set(self.byte_decoder.keys())

        for left, right in ordered_pairs:
            merged_token = left + right
            if merged_token in seen_merged_tokens:
                continue
            pair = (left, right)
            if pair in merges:
                continue
            merges[pair] = len(merges)
            seen_merged_tokens.add(merged_token)
            if len(merges) >= merge_capacity:
                return merges

        for left in range(256):
            for right in range(256):
                pair = (bytes([left]), bytes([right]))
                merged_token = pair[0] + pair[1]
                if merged_token in seen_merged_tokens:
                    continue
                if pair in merges:
                    continue
                merges[pair] = len(merges)
                seen_merged_tokens.add(merged_token)
                if len(merges) >= merge_capacity:
                    return merges

        return merges
    
    def _build_token_map(self):
        """Build bidirectional token-to-string mappings."""
        self.token_to_string = {}
        self.string_to_token = {}
        self.token_to_bytes = {}
        self.bytes_to_token = {}
        
        # Add special tokens
        for token_name, token_id in self.SPECIAL_TOKENS.items():
            self.token_to_string[token_id] = token_name
            self.string_to_token[token_name] = token_id

        # Add byte tokens after the reserved specials.
        for byte_value in range(self.base_vocab_size):
            token_id = self.byte_token_offset + byte_value
            token_bytes = self.byte_encoder[byte_value]
            token_text = token_bytes.decode("latin1")
            self.token_to_bytes[token_id] = token_bytes
            self.bytes_to_token[token_bytes] = token_id
            self.token_to_string[token_id] = token_text
            self.string_to_token[token_text] = token_id
        
        # Add BPE-merged tokens
        next_id = self.merge_token_offset
        for (a, b), _ in sorted(self.bpe_merges.items(), key=lambda x: x[1]):
            if next_id >= self.vocab_size:
                break
            merged = a + b
            if merged in self.bytes_to_token:
                continue
            merged_text = merged.decode("latin1")
            self.token_to_bytes[next_id] = merged
            self.bytes_to_token[merged] = next_id
            self.token_to_string[next_id] = merged_text
            self.string_to_token[merged_text] = next_id
            next_id += 1

        self.actual_vocab_size = len(self.token_to_string)
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncate: bool = True
    ) -> List[int]:
        """
        Encode text to token IDs using custom BPE.
        
        Args:
            text: Input text (must be a string)
            add_special_tokens: Whether to add BOS/EOS tokens
            truncate: Whether to enforce max_length
            
        Returns:
            List of token IDs
            
        Raises:
            ValueError: If text is None or not a string
            
        Example:
            python
            tokenizer = Tokenizer()
            tokens = tokenizer.encode("Hello, world!")
            
        """
        if text is None:
            raise ValueError("text cannot be None. Provide a string to encode.")
        
        if not isinstance(text, str):
            raise ValueError(
                f"text must be a string, got {type(text)._name_}. "
                f"Convert with: str(text)"
            )
        
        # Start with UTF-8 byte-level encoding so all Unicode text round-trips.
        tokens = [bytes([byte_val]) for byte_val in text.encode("utf-8")]
        
        # Apply BPE merges
        tokens = self._apply_bpe(tokens)
        
        # Convert to token IDs
        token_ids = [self.bytes_to_token.get(token, self.unk_token) for token in tokens]
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token] + token_ids + [self.eos_token]
        
        # Truncate to max length
        if truncate and len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length - 1] + [self.eos_token]
        
        return token_ids
    
    def _apply_bpe(self, tokens: List[bytes]) -> List[bytes]:
        """Optimized BPE application using a heap to find merges in O(N log N)."""
        if len(tokens) <= 1:
            return tokens

        # 1. Find all possible merge pairs and their 'rank' (priority)
        queue = []
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            rank = self.bpe_merges.get(pair)
            if rank is not None:
                heapq.heappush(queue, (rank, i, pair))

        while queue:
            rank, i, pair = heapq.heappop(queue)
            
            # Check if the pair is still valid
            if i >= len(tokens) - 1 or (tokens[i], tokens[i+1]) != pair:
                continue
                
            # 2. Perform the merge
            merged_token = pair[0] + pair[1]
            tokens[i] = merged_token
            tokens.pop(i + 1)
            
            # 3. Check NEW pairs created (neighbors)
            if i > 0:
                prev_pair = (tokens[i-1], tokens[i])
                prev_rank = self.bpe_merges.get(prev_pair)
                if prev_rank is not None:
                    heapq.heappush(queue, (prev_rank, i-1, prev_pair))
            
            if i < len(tokens) - 1:
                next_pair = (tokens[i], tokens[i+1])
                next_rank = self.bpe_merges.get(next_pair)
                if next_rank is not None:
                    heapq.heappush(queue, (next_rank, i, next_pair))

        # THIS LINE MUST BE OUTSIDE THE WHILE LOOP (Correctly indented)
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs (must be a list of integers)
            skip_special_tokens: Whether to skip BOS/EOS/PAD tokens
            
        Returns:
            Decoded text string
            
        Raises:
            ValueError: If token_ids is None or invalid
            
        Example:
            python
            tokenizer = Tokenizer()
            text = tokenizer.decode([1, 2, 3, 4])
            
        """
        if token_ids is None:
            raise ValueError("token_ids cannot be None. Provide a list of token IDs.")
        
        if not isinstance(token_ids, list):
            raise ValueError(
                f"token_ids must be a list, got {type(token_ids)._name_}. "
                f"Convert with: list(token_ids)"
            )
        
        if not all(isinstance(t, int) for t in token_ids):
            raise ValueError(
                "token_ids must contain only integers. "
                f"Found types: {set(type(t)._name_ for t in token_ids)}"
            )
        
        special = {self.pad_token, self.bos_token, self.eos_token}
        parts: List[str] = []
        byte_buffer = bytearray()

        def flush_byte_buffer() -> None:
            if byte_buffer:
                parts.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in token_ids:
            if token_id == self.unk_token:
                flush_byte_buffer()
                parts.append("?" if skip_special_tokens else "<unk>")
                continue
            if token_id in special:
                if skip_special_tokens:
                    continue
                flush_byte_buffer()
                parts.append(self.token_to_string[token_id])
                continue

            token_bytes = self.token_to_bytes.get(token_id)
            if token_bytes is None:
                flush_byte_buffer()
                parts.append("?" if skip_special_tokens else "<unk>")
                continue

            byte_buffer.extend(token_bytes)

        flush_byte_buffer()
        return ''.join(parts)
    
    def encode_to_tensor(
        self, 
        text: str, 
        device: str = 'cpu'
    ) -> 'torch.Tensor':
        """
        Encode text to padded tensor.
        
        Args:
            text: Input text (must be a string)
            device: Target device ('cpu' or 'cuda')
            
        Returns:
            Tensor of shape (1, max_length)
            
        Raises:
            ValueError: If text is None or invalid
            RuntimeError: If PyTorch is not available
            
        Example:
            python
            tokenizer = Tokenizer()
            tensor = tokenizer.encode_to_tensor("Hello, world!", device='cuda')
            
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch required for tensor encoding. "
                "Install with: pip install torch"
            )
        
        if text is None:
            raise ValueError("text cannot be None. Provide a string to encode.")
        
        if not isinstance(text, str):
            raise ValueError(
                f"text must be a string, got {type(text)._name_}. "
                f"Convert with: str(text)"
            )
        
        tokens = self.encode(text)
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.pad_token] * (self.max_length - len(tokens))
        
        tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        return tensor
    
    def encode_to_spikes(
        self, 
        text: str, 
        spike_dim: int = 720,
        device: str = 'cpu',
        threshold: float = 0.5
    ) -> 'torch.Tensor':
        """
        Encode text to a deterministic binary spike representation.
        
        Args:
            text: Input text
            spike_dim: Width of the spike feature vector
            device: Target device
            threshold: Spike threshold in [0, 1]
            
        Returns:
            Binary tensor of shape (1, seq_len, spike_dim)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for spike encoding")
        if spike_dim <= 0:
            raise ValueError(f"spike_dim must be positive, got {spike_dim}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        
        tokens = self.encode(text)
        
        # Pad
        if len(tokens) < self.max_length:
            tokens = tokens + [self.pad_token] * (self.max_length - len(tokens))
        tokens = tokens[:self.max_length]
        
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        feature_ids = torch.arange(spike_dim, dtype=torch.long, device=device)

        # Hash-based spikes keep the tokenizer independent from model width.
        hashed = (
            token_tensor.unsqueeze(-1) * 1103515245 +
            feature_ids.unsqueeze(0) * 12345 +
            1013904223
        ) & 0x7FFFFFFF
        normalized = hashed.float() / float(0x7FFFFFFF)
        spikes = (normalized >= threshold).float()
        spikes[token_tensor == self.pad_token] = 0.0

        return spikes.unsqueeze(0)  # (1, seq_len, spike_dim)

    def _sample_token_ids_from_logits(
        self,
        logits: 'torch.Tensor',
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> List[int]:
        """Sample token IDs from model logits after validation and filtering."""
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch required for logits decoding. "
                "Install with: pip install torch"
            )

        if logits is None:
            raise ValueError("logits cannot be None. Provide a PyTorch tensor.")

        if temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {temperature}. "
                f"Typical values: 0.1-2.0"
            )

        if top_k < 0:
            raise ValueError(
                f"top_k must be non-negative, got {top_k}. "
                f"Typical values: 0 (disabled), 10-100"
            )

        if not (0 <= top_p <= 1):
            raise ValueError(
                f"top_p must be between 0 and 1, got {top_p}. "
                f"Typical values: 0.5-0.95"
            )

        if torch.isnan(logits).any():
            raise RuntimeError(
                "logits contains NaN values. "
                "Check model output for numerical issues."
            )

        if torch.isinf(logits).any():
            raise RuntimeError(
                "logits contains Inf values. "
                "Check model output for numerical issues."
            )

        if len(logits.shape) == 3:
            logits = logits[:, -1, :]  # Use last position
        elif len(logits.shape) != 2:
            raise ValueError(
                f"logits must be 2D or 3D, got shape {tuple(logits.shape)}"
            )

        # Clone before filtering to avoid mutating caller-owned tensors.
        logits = logits.clone()

        # Ensure valid vocab range
        if logits.shape[-1] > self.vocab_size:
            logits = logits[..., :self.vocab_size]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return token_ids.tolist()
    
    def decode_from_logits(
        self, 
        logits: 'torch.Tensor',
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """
        Decode text from model output logits.
        
        Args:
            logits: Model output of shape (batch, seq_len, vocab_size) or (batch, vocab_size)
            temperature: Sampling temperature (must be positive)
            top_k: Top-k filtering (must be non-negative)
            top_p: Nucleus sampling threshold (0-1)
            
        Returns:
            Decoded text
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If PyTorch is not available or logits contain NaN/Inf
            
        Example:
            python
            tokenizer = Tokenizer()
            logits = model(input_ids)  # (batch, vocab_size)
            text = tokenizer.decode_from_logits(logits, temperature=0.8)
            
        """
        token_ids = self._sample_token_ids_from_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        return self.decode(token_ids)

    def generate_text(
        self,
        model: 'torch.nn.Module',
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        device: str = 'cuda'
    ) -> str:
        """
        Generate text using the model.
        
        Args:
            model: The neural network model
            prompt: Starting text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling
            device: Device to use
            
        Returns:
            Generated text
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for generation")

        # Use the raw prompt context without EOS/padding so generation remains autoregressive.
        context_ids = self.encode(prompt)
        if context_ids and context_ids[-1] == self.eos_token:
            context_ids = context_ids[:-1]
        if not context_ids:
            context_ids = [self.bos_token]

        generated_ids = []
        
        model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_ids = torch.tensor(
                    [context_ids[-self.max_length:]],
                    dtype=torch.long,
                    device=device
                )

                # Forward pass
                outputs = model(input_ids)
                
                # Get logits for next token
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                next_token_id = self._sample_token_ids_from_logits(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )[0]

                if next_token_id == self.eos_token:
                    break

                context_ids.append(next_token_id)
                generated_ids.append(next_token_id)

        return prompt + self.decode(generated_ids)

    # ------------------------------------------------------------------ #
    #  BPE Training — learn merges from real corpus data                  #
    # ------------------------------------------------------------------ #

    def train(
        self,
        corpus: List[str],
        vocab_size: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """
        Train BPE merges on real corpus data.
        
        This replaces the default heuristic merges with statistically
        optimal ones learned from the actual training text.

        Args:
            corpus: List of text strings to learn merges from
            vocab_size: Override vocabulary size (optional)
            verbose: Print progress during training
            
        Example:
            tokenizer = livorator(vocab_size=16384)
            tokenizer.train(["Hello world", "The cat sat on the mat", ...])
            tokenizer.save("tokenizer.json")
        """
        if vocab_size is not None:
            min_vocab_size = len(self.SPECIAL_TOKENS) + 256
            self.vocab_size = max(int(vocab_size), min_vocab_size)

        merge_capacity = max(0, self.vocab_size - self.merge_token_offset)
        if merge_capacity == 0:
            return

        if verbose:
            print(f"[BPE] Training {merge_capacity:,} merges from {len(corpus):,} texts...")

        # Step 1: Convert all texts to byte-token sequences
        byte_sequences: List[List[bytes]] = []
        for text in corpus:
            seq = [bytes([b]) for b in text.encode("utf-8")]
            if seq:
                byte_sequences.append(seq)

        if not byte_sequences:
            raise ValueError("Corpus is empty — no texts to train on.")

        # Step 2: Iteratively find and merge the most frequent pair
        merges: Dict[Tuple[bytes, bytes], int] = {}

        for merge_idx in range(merge_capacity):
            # Count all adjacent pairs
            pair_counts: Counter = Counter()
            for seq in byte_sequences:
                for j in range(len(seq) - 1):
                    pair_counts[(seq[j], seq[j + 1])] += 1

            if not pair_counts:
                if verbose:
                    print(f"[BPE] No more pairs to merge at step {merge_idx}. Stopping early.")
                break

            # Find the most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            merged_token = best_pair[0] + best_pair[1]
            merges[best_pair] = merge_idx

            # Apply this merge across all sequences
            for seq_idx in range(len(byte_sequences)):
                new_seq: List[bytes] = []
                j = 0
                old_seq = byte_sequences[seq_idx]
                while j < len(old_seq):
                    if j < len(old_seq) - 1 and (old_seq[j], old_seq[j + 1]) == best_pair:
                        new_seq.append(merged_token)
                        j += 2
                    else:
                        new_seq.append(old_seq[j])
                        j += 1
                byte_sequences[seq_idx] = new_seq

            if verbose and (merge_idx + 1) % 500 == 0:
                best_text = (best_pair[0] + best_pair[1]).decode("utf-8", errors="replace")
                print(
                    f"[BPE] {merge_idx + 1:,}/{merge_capacity:,} merges — "
                    f"latest: '{best_text}' (count: {pair_counts[best_pair]:,})"
                )

        self.bpe_merges = merges
        self._build_token_map()

        if verbose:
            print(f"[BPE] Training complete! {len(merges):,} merges learned.")
            print(f"[BPE] Actual vocabulary size: {self.actual_vocab_size:,}")

    # ------------------------------------------------------------------ #
    #  Save / Load — persist trained tokenizer to disk                    #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """
        Save trained tokenizer to a JSON file.

        Args:
            path: Output file path (e.g., "tokenizer.json")
            
        Example:
            tokenizer.train(corpus)
            tokenizer.save("tokenizer.json")
        """
        import json

        data = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "merges": [
                {
                    "left": list(left),
                    "right": list(right),
                    "priority": priority,
                }
                for (left, right), priority in sorted(
                    self.bpe_merges.items(), key=lambda x: x[1]
                )
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"[OK] Tokenizer saved to {path} ({len(self.bpe_merges):,} merges)")

    @classmethod
    def load(cls, path: str, **kwargs) -> "livorator":
        """
        Load a trained tokenizer from a JSON file.

        Args:
            path: Path to the tokenizer JSON file
            **kwargs: Override any init parameter (e.g., max_length=1024)

        Returns:
            livorator instance with trained merges
            
        Example:
            tokenizer = livorator.load("tokenizer.json")
            tokens = tokenizer.encode("Hello world")
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        merges: Dict[Tuple[bytes, bytes], int] = {}
        for entry in data["merges"]:
            pair = (bytes(entry["left"]), bytes(entry["right"]))
            merges[pair] = entry["priority"]

        return cls(
            vocab_size=kwargs.pop("vocab_size", data["vocab_size"]),
            max_length=kwargs.pop("max_length", data["max_length"]),
            bpe_merges=merges,
            **kwargs,
        )


def get_tokenizer(
    tokenizer_type: str = "auto",
    tokenizer_path: Optional[str] = None,
    **kwargs,
):
    """
    Get a tokenizer instance.
    
    Args:
        tokenizer_type: "auto", "custom", "bpe", "livo", or "livorator"
        tokenizer_path: Path to a saved tokenizer JSON (loads trained merges)
        **kwargs: Additional arguments for tokenizer

    Returns:
        Tokenizer instance.
    """
    tokenizer_type = tokenizer_type.lower()

    if tokenizer_type in {"auto", "custom", "bpe", "livo", "livorator"}:
        if tokenizer_path is not None:
            return livorator.load(tokenizer_path, **kwargs)
        return livorator(**kwargs)

    raise ValueError(
        f"Unsupported tokenizer_type '{tokenizer_type}'. "
        "Expected one of: auto, custom, bpe, livo, livorator."
    )


# Quick test
if __name__ == "__main__":
    print("=" * 50)
    print("livorator Tokenizer Test")
    print("=" * 50)

    # 1. Default (seed merges)
    tok = livorator(vocab_size=16384)
    text = "Hello, World Brain!"
    print(f"\n[Default Merges]")
    print(f"  Input:   {text}")
    tokens = tok.encode(text)
    print(f"  Tokens:  {tokens} ({len(tokens)} ids)")
    decoded = tok.decode(tokens)
    print(f"  Decoded: {decoded}")

    # 2. Trained on sample data
    sample_corpus = [
        "Once upon a time there was a little girl named Lily.",
        "She loved to play in the garden with her friends.",
        "The sun was shining and the birds were singing.",
        "Lily picked a beautiful flower and gave it to her mom.",
        "Her mom smiled and said thank you very much.",
    ] * 20  # repeat for better statistics

    tok2 = livorator(vocab_size=512)
    tok2.train(sample_corpus, verbose=True)

    text2 = "Lily loved the garden."
    print(f"\n[Trained Merges]")
    print(f"  Input:   {text2}")
    tokens2 = tok2.encode(text2)
    print(f"  Tokens:  {tokens2} ({len(tokens2)} ids)")
    decoded2 = tok2.decode(tokens2)
    print(f"  Decoded: {decoded2}")

    # 3. Save and reload
    tok2.save("_test_tokenizer.json")
    tok3 = livorator.load("_test_tokenizer.json")
    tokens3 = tok3.encode(text2)
    print(f"\n[Loaded from disk]")
    print(f"  Tokens:  {tokens3}")
    assert tokens2 == tokens3, "Round-trip failed!"
    print("  ✓ Save/load round-trip passed!")

    import os
    os.remove("_test_tokenizer.json")