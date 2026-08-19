"""
Byte-level BPE Tokenizer — GPT-2 style.

Three training algorithms behind a single interface:
  algo="naive"       (V1): recount all pair frequencies each round — O(corpus) per merge
  algo="incremental" (V2): maintain a running pair-count table, update only affected pairs
  num_workers>1      (V3): multiprocessing pre-tokenisation, then V2 merge loop on aggregated counts

All three produce byte-identical vocab/merges for the same corpus.
"""

from __future__ import annotations

import json
import multiprocessing
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import regex  # pip install regex  (supports \p{L} etc.)

# ---------------------------------------------------------------------------
# GPT-2 pre-tokenisation pattern (unchanged from the original paper / tiktoken)
# ---------------------------------------------------------------------------
GPT2_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)|"""      # contractions: 's 'd 'm 't 'll 've 're
    r"""[^\r\n\p{L}\p{N}]?\p{L}+|"""  # optional leading non-letter/non-digit, then letters
    r"""\p{N}{1,3}|"""                 # up to 3 digit run
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|""" # punctuation / symbols (with optional trailing newlines)
    r"""\s*[\r\n]+|"""                 # newlines / whitespace ending in newline
    r"""\s+(?!\S)|"""                  # trailing whitespace (end of string)
    r"""\s+"""                         # remaining whitespace
)

# ---------------------------------------------------------------------------
# Helpers shared across all three algorithm variants
# ---------------------------------------------------------------------------

def _pretokenise(text: str) -> list[bytes]:
    """Split text with GPT-2 regex, return each pre-token as a bytes object."""
    return [m.group(0).encode("utf-8") for m in GPT2_PATTERN.finditer(text)]


def _bytes_to_token_ids(token_bytes: bytes) -> list[int]:
    """Expand a bytes object into a list of single-byte integer ids (0–255)."""
    return list(token_bytes)


def _count_pairs(word_freqs: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    """Full recount of every adjacent pair across all words (naive / V1)."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += freq
    return counts


def _initial_id_bytes() -> dict[int, bytes]:
    """id -> byte sequence mapping, seeded with the 256 base bytes.

    SPEC §5 tie-breaking compares the *byte sequences* a pair represents,
    not the raw int ids — merged ids (256+) are ordered by creation time,
    which is NOT lexicographic byte order. Both training loops must extend
    this map as merges are created and use it in their max() key.
    """
    return {i: bytes([i]) for i in range(256)}


# ---------------------------------------------------------------------------
# Worker function for multiprocessing (must be a module-level def so it's
# picklable under macOS spawn start method)
# ---------------------------------------------------------------------------

def _worker_pretokenise(args: tuple[str, int, int, str]) -> dict[tuple[int, ...], int]:
    """
    Read a byte-slice [start, end) from file_path, pre-tokenise the text
    (excluding special tokens), and return word-frequency counts.

    The caller guarantees that [start, end) is aligned to special-token
    boundaries, so we never see a split special token here.
    """
    file_path, start, end, special_token_pattern = args
    with open(file_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    text = raw.decode("utf-8", errors="replace")

    # Split on special tokens; only the non-special segments get pre-tokenised.
    word_freqs: dict[tuple[int, ...], int] = defaultdict(int)
    for segment in re.split(special_token_pattern, text):
        for tok_bytes in _pretokenise(segment):
            word = tuple(tok_bytes)
            if word:
                word_freqs[word] += 1
    return dict(word_freqs)


# ---------------------------------------------------------------------------
# Chunk-boundary alignment (V3)
# ---------------------------------------------------------------------------

def _find_aligned_boundaries(
    file_path: str,
    num_workers: int,
    special_token_bytes: bytes,
) -> list[int]:
    """
    Split the file into num_workers roughly-equal chunks, but snap each
    boundary *forward* to just after the next occurrence of special_token_bytes
    (so no special token straddles a boundary).

    Returns a sorted list of [start, ..., end] offsets of length num_workers+1.
    """
    file_size = Path(file_path).stat().st_size
    if file_size == 0:
        return [0, 0]

    nominal_chunk = max(1, file_size // num_workers)
    boundaries = [0]
    window = 1 << 20  # 1MB forward-search window; avoids loading whole file

    with open(file_path, "rb") as f:
        for i in range(1, num_workers):
            nominal = i * nominal_chunk
            if nominal >= file_size:
                break
            # Search forward from nominal position for the end of a special
            # token, reading windowed blocks with overlap so a special token
            # spanning a window edge is still found.
            pos = nominal
            boundary = None
            while pos < file_size:
                f.seek(pos)
                block = f.read(window + len(special_token_bytes) - 1)
                idx = block.find(special_token_bytes)
                if idx != -1:
                    boundary = pos + idx + len(special_token_bytes)
                    break
                pos += window
            if boundary is None:
                # No more special token; remaining bytes go to last chunk.
                break
            boundaries.append(min(boundary, file_size))

    boundaries.append(file_size)
    # Deduplicate while preserving order.
    seen: set[int] = set()
    unique: list[int] = []
    for b in boundaries:
        if b not in seen:
            seen.add(b)
            unique.append(b)
    return unique


# ---------------------------------------------------------------------------
# Core training — naive (V1)
# ---------------------------------------------------------------------------

def _train_naive(
    word_freqs: dict[tuple[int, ...], int],
    vocab_size: int,
    n_special: int,
) -> list[tuple[int, int]]:
    """
    V1: each round, recount all pairs from scratch.
    Simple and obviously correct — used as the reference.
    """
    # Number of merges = target vocab beyond initial 256 bytes + specials.
    n_merges = vocab_size - 256 - n_special
    merges: list[tuple[int, int]] = []
    next_id = 256  # merged tokens get ids 256, 257, ...
    id_bytes = _initial_id_bytes()

    for _ in range(n_merges):
        counts = _count_pairs(word_freqs)
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], (id_bytes[p[0]], id_bytes[p[1]])))
        merges.append(best)
        id_bytes[next_id] = id_bytes[best[0]] + id_bytes[best[1]]

        # Apply the merge: replace every occurrence of best in every word.
        new_word_freqs: dict[tuple[int, ...], int] = {}
        for word, freq in word_freqs.items():
            new_word = _apply_merge(word, best, next_id)
            new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq
        word_freqs = new_word_freqs
        next_id += 1

    return merges


# ---------------------------------------------------------------------------
# Core training — incremental (V2)
# ---------------------------------------------------------------------------

def _train_incremental(
    word_freqs: dict[tuple[int, ...], int],
    vocab_size: int,
    n_special: int,
) -> list[tuple[int, int]]:
    """
    V2: maintain a running pair-count dict; after each merge, only update the
    counts for pairs that were adjacent to the merged pair.

    Complexity per merge: O(occurrences of chosen pair) instead of O(corpus).
    """
    n_merges = vocab_size - 256 - n_special
    merges: list[tuple[int, int]] = []
    next_id = 256
    id_bytes = _initial_id_bytes()

    # Initial full count.
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq

    for _ in range(n_merges):
        if not pair_counts:
            break
        # Filter out zero-count entries that accumulate from decrements.
        best = max(
            (p for p, c in pair_counts.items() if c > 0),
            key=lambda p: (pair_counts[p], (id_bytes[p[0]], id_bytes[p[1]])),
            default=None,
        )
        if best is None:
            break
        merges.append(best)
        id_bytes[next_id] = id_bytes[best[0]] + id_bytes[best[1]]

        # Apply merge and incrementally update pair counts.
        new_word_freqs: dict[tuple[int, ...], int] = {}
        for word, freq in word_freqs.items():
            if best[0] not in word and best[1] not in word:
                # Quick skip: neither token in word, merge can't apply.
                new_word_freqs[word] = new_word_freqs.get(word, 0) + freq
                continue

            new_word = _apply_merge_incremental(
                word, best, next_id, freq, pair_counts
            )
            new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq

        word_freqs = new_word_freqs
        # Remove the merged pair's entry (count should now be 0 or close).
        pair_counts.pop(best, None)
        next_id += 1

    return merges


def _apply_merge(
    word: tuple[int, ...],
    pair: tuple[int, int],
    new_id: int,
) -> tuple[int, ...]:
    """Replace every non-overlapping occurrence of pair in word with new_id."""
    result: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


def _apply_merge_incremental(
    word: tuple[int, ...],
    pair: tuple[int, int],
    new_id: int,
    freq: int,
    pair_counts: dict[tuple[int, int], int],
) -> tuple[int, ...]:
    """
    Apply merge in-place and update pair_counts to reflect:
      - pairs that disappear because they included one of the merged symbols
      - pairs that appear because of the new merged symbol's neighbors
    """
    a, b = pair
    result: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            # This pair will be replaced.
            # Left context: (word[i-1], a) disappears; (word[i-1], new_id) appears.
            if result:
                left = result[-1]
                pair_counts[(left, a)] -= freq
                pair_counts[(left, new_id)] += freq
            # Right context: (b, word[i+2]) disappears; (new_id, word[i+2]) appears.
            if i + 2 < len(word):
                right = word[i + 2]
                # But only if word[i+2] is not also part of a match starting at i+1.
                # (Overlapping pairs: handled naturally since we advance by 2.)
                pair_counts[(b, right)] -= freq
                pair_counts[(new_id, right)] += freq
            result.append(new_id)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


# ---------------------------------------------------------------------------
# BPETokenizer class
# ---------------------------------------------------------------------------

class BPETokenizer:
    """Byte-level BPE, GPT-2 style."""

    def __init__(
        self,
        vocab: dict[int, bytes],        # id -> byte sequence
        merges: list[tuple[int, int]],  # ordered list of (a, b) pairs
        special_tokens: dict[str, int], # token_str -> id
    ) -> None:
        self._vocab = vocab             # id -> bytes
        self._merges = merges           # ordered merge list
        self._special_tokens = special_tokens
        self._special_id_to_str = {v: k for k, v in special_tokens.items()}

        # Reverse vocab: bytes -> id (for encode).
        self._bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        # Merge lookup: (a, b) -> merged_id (for fast encode).
        self._merge_lookup: dict[tuple[int, int], int] = {}
        next_id = 256
        for pair in merges:
            self._merge_lookup[pair] = next_id
            next_id += 1

        # Regex for splitting on special tokens (longest match first).
        if special_tokens:
            sorted_specials = sorted(special_tokens.keys(), key=len, reverse=True)
            pattern = "|".join(re.escape(s) for s in sorted_specials)
            self._special_split_re = re.compile(f"({pattern})")
        else:
            self._special_split_re = None

    # ------------------------------------------------------------------
    # Public interface — SPEC §5
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        input_path: str | Path,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        num_workers: int = 1,
        algo: Literal["naive", "incremental"] = "incremental",
    ) -> "BPETokenizer":
        """Train a BPE tokenizer from a raw UTF-8 text file."""
        input_path = str(input_path)
        special_tokens = special_tokens or []

        # Step 1: Build word-frequency table via pre-tokenisation.
        if num_workers > 1:
            word_freqs = _parallel_pretokenise(input_path, num_workers, special_tokens)
        else:
            word_freqs = _serial_pretokenise(input_path, special_tokens)

        # Step 2: Run the merge loop.
        n_special = len(special_tokens)
        if algo == "naive":
            merges = _train_naive(word_freqs, vocab_size, n_special)
        else:
            merges = _train_incremental(word_freqs, vocab_size, n_special)

        # Step 3: Build vocab.
        vocab, special_token_map = _build_vocab(merges, special_tokens)
        return cls(vocab, merges, special_token_map)

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token ids."""
        if not text:
            return []
        ids: list[int] = []

        # Split on special tokens first; alternating non-special / special segments.
        if self._special_split_re:
            parts = self._special_split_re.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if part in self._special_tokens:
                ids.append(self._special_tokens[part])
            else:
                ids.extend(self._encode_ordinary(part))

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to a string. Unknown ids are skipped."""
        byte_chunks: list[bytes] = []
        for id_ in ids:
            if id_ in self._special_id_to_str:
                byte_chunks.append(self._special_id_to_str[id_].encode("utf-8"))
            elif id_ in self._vocab:
                byte_chunks.append(self._vocab[id_])
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, dirpath: str | Path) -> None:
        """Save vocab and merges to dirpath as JSON files."""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # vocab.json: {str(id): base64 or list-of-ints representation}
        # We store as {str(id): [b0, b1, ...]} for human readability.
        vocab_data = {str(k): list(v) for k, v in sorted(self._vocab.items())}
        (dirpath / "vocab.json").write_text(
            json.dumps(vocab_data, sort_keys=False, ensure_ascii=False), encoding="utf-8"
        )

        # merges.json: [[a, b], ...]
        merges_data = {
            "merges": [[a, b] for a, b in self._merges],
            "special_tokens": {k: v for k, v in sorted(self._special_tokens.items())},
        }
        (dirpath / "merges.json").write_text(
            json.dumps(merges_data, sort_keys=True, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, dirpath: str | Path) -> "BPETokenizer":
        """Load a tokenizer saved with save()."""
        dirpath = Path(dirpath)
        vocab_data = json.loads((dirpath / "vocab.json").read_text(encoding="utf-8"))
        merges_data = json.loads((dirpath / "merges.json").read_text(encoding="utf-8"))

        vocab = {int(k): bytes(v) for k, v in vocab_data.items()}
        merges = [tuple(pair) for pair in merges_data["merges"]]
        special_tokens = merges_data.get("special_tokens", {})
        return cls(vocab, merges, special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + len(self._special_tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode a text segment that contains no special tokens."""
        ids: list[int] = []
        for pre_token_bytes in _pretokenise(text):
            # Start with byte-level ids.
            token_ids = list(pre_token_bytes)
            # Iteratively apply merges in training order.
            token_ids = self._apply_merges(token_ids)
            ids.extend(token_ids)
        return ids

    def _apply_merges(self, ids: list[int]) -> list[int]:
        """Apply all learned merges to a sequence of ids."""
        # Keep looping until no more merges apply.
        while len(ids) >= 2:
            # Find the merge with the lowest index (earliest learned) among
            # all adjacent pairs in ids.
            best_idx = None
            best_merge_rank = len(self._merges)  # sentinel
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                if pair in self._merge_lookup:
                    # Use the position in _merges as the rank.
                    # We need to find the rank; store it for efficiency.
                    # Actually _merge_lookup gives us the merged id,
                    # and merged ids are assigned in order from 256 upward,
                    # so rank = merged_id - 256.
                    rank = self._merge_lookup[pair] - 256
                    if rank < best_merge_rank:
                        best_merge_rank = rank
                        best_idx = i
            if best_idx is None:
                break
            pair = (ids[best_idx], ids[best_idx + 1])
            new_id = self._merge_lookup[pair]
            ids = ids[:best_idx] + [new_id] + ids[best_idx + 2:]
        return ids


# ---------------------------------------------------------------------------
# Pre-tokenisation helpers
# ---------------------------------------------------------------------------

def _serial_pretokenise(
    file_path: str,
    special_tokens: list[str],
) -> dict[tuple[int, ...], int]:
    """Read entire file, split on special tokens, pre-tokenise the rest."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    word_freqs: dict[tuple[int, ...], int] = defaultdict(int)

    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(s) for s in sorted_specials)
        segments = re.split(pattern, text)
    else:
        segments = [text]

    for segment in segments:
        for tok_bytes in _pretokenise(segment):
            word = tuple(tok_bytes)
            if word:
                word_freqs[word] += 1

    return dict(word_freqs)


def _parallel_pretokenise(
    file_path: str,
    num_workers: int,
    special_tokens: list[str],
) -> dict[tuple[int, ...], int]:
    """
    V3: Chunk the file at special-token boundaries, dispatch to worker pool,
    then aggregate the word-frequency dicts from all workers.
    """
    # Use the first special token as the document boundary marker.
    # If multiple exist, we use the first one for chunking alignment.
    if special_tokens:
        primary_special = special_tokens[0].encode("utf-8")
    else:
        primary_special = b"\n"  # fallback; no real boundary needed

    boundaries = _find_aligned_boundaries(file_path, num_workers, primary_special)

    # Build pattern string for splitting on special tokens inside workers.
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "|".join(re.escape(s) for s in sorted_specials)
    else:
        special_pattern = r"(?!)"  # never matches

    tasks = [
        (file_path, boundaries[i], boundaries[i + 1], special_pattern)
        for i in range(len(boundaries) - 1)
        if boundaries[i] < boundaries[i + 1]
    ]

    # Use spawn context explicitly for macOS compatibility.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=min(num_workers, len(tasks))) as pool:
        results = pool.map(_worker_pretokenise, tasks)

    # Aggregate word frequencies from all workers.
    aggregated: dict[tuple[int, ...], int] = defaultdict(int)
    for wf in results:
        for word, freq in wf.items():
            aggregated[word] += freq
    return dict(aggregated)


# ---------------------------------------------------------------------------
# Vocab construction
# ---------------------------------------------------------------------------

def _build_vocab(
    merges: list[tuple[int, int]],
    special_tokens: list[str],
) -> tuple[dict[int, bytes], dict[str, int]]:
    """
    Build the id->bytes vocab from the base 256 bytes + merge results.
    Special tokens get ids starting after all regular vocab entries.
    """
    # Base vocab: byte 0..255 each maps to a single byte.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # Add merged tokens in order.
    for new_id, (a, b) in enumerate(merges, start=256):
        vocab[new_id] = vocab[a] + vocab[b]

    # Special tokens come after regular vocab.
    regular_vocab_size = 256 + len(merges)
    special_token_map: dict[str, int] = {}
    for i, tok in enumerate(special_tokens):
        special_token_map[tok] = regular_vocab_size + i

    return vocab, special_token_map
