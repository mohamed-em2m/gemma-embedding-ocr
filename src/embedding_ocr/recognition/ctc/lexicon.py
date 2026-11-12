"""Lexicon rescoring for CTC decoding.

The acoustic model has no notion of words: it happily emits `qog` or `hlo`.
A word list built from the training transcripts fixes most of those errors at
almost no cost. The scorer is a prefix-set trie over the lexicon, plugged into
:func:`embedding_ocr.recognition.ctc.decode.prefix_beam_search` through the
``PrefixScorer`` protocol.

Scoring is character-level. The scorer tracks the current partial word (the
characters after the last space) and returns:

* ``0.0`` while the partial word can still extend to a lexicon word,
* ``complete_bonus`` once it is exactly a lexicon word,
* ``-penalty`` as soon as it can no longer become one.

Because every beam hypothesis keeps a single best prefix, the penalty prunes
non-words without an explicit word-level lattice.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from embedding_ocr.recognition.ctc.charset import CharSet
from embedding_ocr.recognition.ctc.normalize import normalize_text

SPACE = " "


@dataclass
class WordLexicon:
    """A set of known words plus every prefix of those words.

    Attributes:
        words: The normalized lexicon words.
        prefixes: Every proper prefix of every word, used for incremental
            viability checks.
        max_word_length: Longest word, used to bound the partial-word buffer.
    """

    words: frozenset[str]
    prefixes: frozenset[str] = field(default_factory=frozenset)
    max_word_length: int = 0

    def __post_init__(self) -> None:
        if not self.prefixes and self.words:
            prefixes = {
                word[:length] for word in self.words for length in range(1, len(word))
            }
            self.prefixes = frozenset(prefixes)
        if self.max_word_length == 0 and self.words:
            self.max_word_length = max(len(word) for word in self.words)

    @classmethod
    def from_texts(cls, texts: Iterable[str], *, min_count: int = 1) -> WordLexicon:
        """Build a lexicon from transcript lines.

        Args:
            texts: Raw transcript lines; each is normalized first.
            min_count: Drop words that occur fewer times than this.
        """
        counts: dict[str, int] = {}
        for text in texts:
            for word in normalize_text(text).split():
                counts[word] = counts.get(word, 0) + 1
        kept = {word for word, count in counts.items() if count >= min_count}
        return cls(frozenset(kept))

    @classmethod
    def from_file(cls, path: Path, *, min_count: int = 1) -> WordLexicon:
        """Build a lexicon from a UTF-8 file with one transcript per line."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls.from_texts(lines, min_count=min_count)

    def __len__(self) -> int:
        return len(self.words)

    def is_word(self, word: str) -> bool:
        return word in self.words

    def is_prefix(self, prefix: str) -> bool:
        return prefix in self.prefixes or prefix in self.words

    def extend(self, partial: str, char: str) -> str:
        """Return the partial word after appending ``char``."""
        if char == SPACE:
            return ""
        return partial + char


class LexiconScorer:
    """A :class:`~embedding_ocr.recognition.ctc.decode.PrefixScorer` over a lexicon.

    Attributes:
        charset: Vocabulary used to turn logit indices back into characters.
        lexicon: Known words.
        penalty: Log-domain penalty for a prefix that cannot become a word.
        complete_bonus: Log-domain bonus for a completed known word.
    """

    def __init__(
        self,
        charset: CharSet,
        lexicon: WordLexicon,
        *,
        penalty: float = 2.0,
        complete_bonus: float = 0.0,
    ) -> None:
        if penalty < 0:
            message = "penalty must be non-negative"
            raise ValueError(message)
        self.charset = charset
        self.lexicon = lexicon
        self.penalty = penalty
        self.complete_bonus = complete_bonus
        self._partials: dict[tuple[int, ...], str] = {}

    def _char(self, index: int) -> str | None:
        if 1 <= index <= len(self.charset.chars):
            return self.charset.chars[index - 1]
        return None

    def reset(self) -> None:
        """Drop memoized prefixes. Call once per decoded line."""
        self._partials.clear()

    def partial_for(self, prefix: Sequence[int]) -> str:
        """Return the current partial word for a prefix, memoized."""
        key = tuple(prefix)
        cached = self._partials.get(key)
        if cached is not None:
            return cached
        partial = ""
        for index in key:
            char = self._char(index)
            if char is None:
                continue
            partial = self.lexicon.extend(partial, char)
        self._partials[key] = partial
        return partial

    def score(self, prefix: Sequence[int], next_index: int) -> float:
        """Score appending ``next_index`` to ``prefix``.

        Args:
            prefix: Emitted logit indices so far.
            next_index: Candidate next index.

        Returns:
            ``0.0`` for a viable continuation, ``complete_bonus`` for a
            completed word, ``-penalty`` when no lexicon word can follow.
        """
        char = self._char(next_index)
        if char is None:
            return 0.0
        partial = self.partial_for(prefix)
        if char == SPACE:
            if partial and not self.lexicon.is_word(partial):
                return -self.penalty
            return 0.0
        extended = self.lexicon.extend(partial, char)
        if self.lexicon.is_word(extended):
            return self.complete_bonus
        if self.lexicon.is_prefix(extended):
            return 0.0
        return -self.penalty
