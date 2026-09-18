"""
Text Analysis Tool — Execution Layer.

Stage 3: Lightweight free-text profiling (length, vocabulary, top tokens)
for columns the profiler flagged as prose rather than category labels
(DatasetProfile.text_cols). Pure pandas/regex — no NLP dependency beyond
what's already in requirements.txt.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

_TOKEN_RE = re.compile(r"[a-zA-Z']+")

#: Minimum average word count for a column to be treated as free text when
#: the caller doesn't name one explicitly (mirrors profiler.FREE_TEXT_AVG_WORDS).
_MIN_AVG_WORDS = 3.0

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "this",
    "that", "for", "on", "with", "as", "was", "were", "are", "be", "by",
    "at", "from", "i", "you", "he", "she", "we", "they", "but", "not",
    "have", "has", "had", "so", "if", "then", "there", "their", "its",
    "my", "your", "our", "just", "very", "also", "can", "will", "would",
})


def _autodetect_text_column(df: Any) -> str | None:
    best_col: str | None = None
    best_avg = 0.0
    for col in df.select_dtypes(exclude="number").columns:
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            continue
        avg_words = float(sample.str.split().str.len().mean())
        if avg_words > best_avg:
            best_avg = avg_words
            best_col = str(col)
    return best_col if best_avg >= _MIN_AVG_WORDS else None


class TextAnalysisTool(BaseTool):
    """Length, vocabulary, and top-token profiling for a free-text column."""

    name = "text_analysis"
    description = (
        "Profile a free-text column: non-null count, average character/word length, "
        "vocabulary size, and the most frequent tokens (stopwords excluded). "
        "Use when the data profile lists text_cols. Auto-detects the column when omitted."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 1.0 if profile is not None and profile.text_cols else 0.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        text_column: str | None = None,
        top_n: int = 15,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)

        if text_column is None:
            text_column = _autodetect_text_column(df)
        if text_column is None or text_column not in df.columns:
            raise ToolExecutionError(
                "No free-text column found. Pass text_column explicitly."
            )

        series = df[text_column].dropna().astype(str)
        if series.empty:
            raise ToolExecutionError(f"Column '{text_column}' has no non-null values.")

        char_lengths = series.str.len()
        word_lengths = series.str.split().str.len()

        tokens: Counter[str] = Counter()
        for text in series:
            for tok in _TOKEN_RE.findall(text.lower()):
                if tok not in _STOPWORDS and len(tok) > 1:
                    tokens[tok] += 1

        top_tokens = dict(tokens.most_common(top_n))

        return {
            "summary": (
                f"Text column '{text_column}': {len(series):,} non-null values, "
                f"avg {word_lengths.mean():.1f} words / {char_lengths.mean():.0f} chars, "
                f"vocabulary of {len(tokens):,} distinct token(s)."
            ),
            "text_column": text_column,
            "non_null_count": len(series),
            "missing_pct": round(100.0 * (1 - len(series) / max(len(df), 1)), 2),
            "avg_char_length": round(float(char_lengths.mean()), 2),
            "avg_word_count": round(float(word_lengths.mean()), 2),
            "vocab_size": len(tokens),
            "top_tokens": top_tokens,
        }

    def findings(  # type: ignore[override]
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        results: list[Finding] = []

        top_tokens = output.get("top_tokens")
        non_null = output.get("non_null_count")
        text_column = output.get("text_column")
        if top_tokens and non_null:
            top_word, top_count = next(iter(top_tokens.items()))
            occurrence_rate = top_count / non_null
            if occurrence_rate > 0.05:
                results.append(
                    Finding(
                        finding_id=f"{self.name}_top_token",
                        kind="text",
                        headline=(
                            f"'{top_word}' is the most frequent term in '{text_column}' "
                            f"({top_count:,} occurrences across {non_null:,} entries)"
                        ),
                        detail=(
                            f"Vocabulary of {output.get('vocab_size', 0):,} distinct token(s); "
                            f"top {len(top_tokens)} shown."
                        ),
                        evidence={"top_tokens": top_tokens, "non_null_count": non_null},
                        measure="token_frequency",
                        dimension=text_column,
                        level=top_word,
                        effect=round(min(occurrence_rate, 1.0), 4),
                        effect_kind="pct",
                        confidence=0.5,
                        surprise=0.2,
                    )
                )

        missing_pct = output.get("missing_pct")
        if missing_pct is not None and missing_pct > 20.0:
            results.append(
                Finding(
                    finding_id=f"{self.name}_missing",
                    kind="method_fit",
                    headline=f"'{text_column}' is missing in {missing_pct:.1f}% of rows",
                    evidence={"missing_pct": missing_pct},
                    confidence=0.8,
                    surprise=0.0,
                    caveats=[f"{missing_pct:.1f}% of rows have no value in '{text_column}'."],
                )
            )

        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the (cleaned) dataset.", "required": True},
            "text_column": {
                "type": "string",
                "description": "Free-text column to profile. Auto-detected if omitted.",
                "required": False,
            },
            "top_n": {
                "type": "int",
                "description": "Number of top tokens to return. Default: 15.",
                "required": False,
            },
        }
