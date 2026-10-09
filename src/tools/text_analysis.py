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

# Unit-bearing, compact strings are specifications rather than prose.  This
# deliberately names measurement units rather than products or domains, so it
# generalises to devices, lab results, engineering parts, and product catalogs.
_STRUCTURED_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:"
    r"(?:milli|centi|kilo)?(?:meter|metre|gram|liter|litre|second|minute|hour|volt|watt|hertz|byte)|"
    r"mm|cm|km|mg|kg|ml|l|mah|ah|wh|kwh|mv|v|kv|mw|w|kw|hz|khz|mhz|ghz|"
    r"mb|gb|tb|mp|dpi|fps|rpm|ohm|db|°c|°f|%"
    r")\b",
    re.IGNORECASE,
)
_MAX_WORDS_FOR_COMPACT_SPEC = 14
_STRUCTURED_SPEC_FRACTION = 0.65
_PROSE_NAME_HINTS = frozenset({
    "comment", "comments", "description", "feedback", "message", "messages",
    "narrative", "note", "notes", "review", "reviews", "summary", "text",
    "transcript", "verbatim",
})
_STRUCTURED_NAME_HINTS = frozenset({
    "battery", "camera", "capacity", "charging", "clock", "cpu", "dimension",
    "display", "frequency", "memory", "network", "operating", "platform",
    "power", "processor", "ram", "resolution", "screen", "spec", "specification",
    "storage",
})
_PROSE_WITHOUT_NAME_MIN_WORDS = 15.0

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


def _is_structured_spec_series(series: Any) -> bool:
    """Whether a text-like series is dominated by compact unit specifications.

    A genuine prose field can mention measurements, so a single unit is not
    enough.  The column is excluded only when a clear majority of its observed
    cells are short specification-like values containing one or more units.
    """
    values = series.dropna().astype(str).head(200)
    if values.empty:
        return False
    structured = 0
    for value in values:
        unit_matches = len(_STRUCTURED_UNIT_RE.findall(value))
        word_count = len(_TOKEN_RE.findall(value))
        if unit_matches >= 2 or (unit_matches >= 1 and word_count <= _MAX_WORDS_FOR_COMPACT_SPEC):
            structured += 1
    return structured / len(values) >= _STRUCTURED_SPEC_FRACTION


def _autodetect_text_column(df: Any) -> str | None:
    best_col: str | None = None
    best_avg = 0.0
    for col in df.select_dtypes(exclude="number").columns:
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            continue
        if _is_structured_spec_series(sample):
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
        if profile is None:
            return 0.0
        for column in profile.columns:
            if column.kind != "text":
                continue
            name_tokens = set(re.findall(r"[a-z]+", column.name.lower()))
            average_words = float(column.stats.get("avg_word_count") or 0.0)
            if name_tokens & _PROSE_NAME_HINTS:
                return 1.0
            if name_tokens & _STRUCTURED_NAME_HINTS:
                continue
            if average_words >= _PROSE_WITHOUT_NAME_MIN_WORDS:
                return 1.0
        return 0.0

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

        if _is_structured_spec_series(df[text_column]):
            raise ToolExecutionError(
                f"Column '{text_column}' is dominated by compact unit-bearing specifications, not prose; "
                "use structured feature extraction instead of text analysis."
            )

        series = df[text_column].dropna().astype(str)
        if series.empty:
            raise ToolExecutionError(f"Column '{text_column}' has no non-null values.")

        char_lengths = series.str.len()
        word_lengths = series.str.split().str.len()

        tokens: Counter[str] = Counter()
        documents_by_token: Counter[str] = Counter()
        for text in series:
            document_tokens: set[str] = set()
            for tok in _TOKEN_RE.findall(text.lower()):
                if tok not in _STOPWORDS and len(tok) > 1:
                    tokens[tok] += 1
                    document_tokens.add(tok)
            for tok in document_tokens:
                documents_by_token[tok] += 1

        top_tokens = dict(tokens.most_common(top_n))
        total_token_count = sum(tokens.values())
        top_token_stats = {
            token: {
                "occurrence_count": count,
                "document_count": documents_by_token[token],
                "document_prevalence": round(documents_by_token[token] / len(series), 4),
                "occurrence_frequency": round(count / total_token_count, 4) if total_token_count else 0.0,
            }
            for token, count in top_tokens.items()
        }

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
            "total_token_count": total_token_count,
            "top_token_stats": top_token_stats,
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        results: list[Finding] = []

        top_tokens = output.get("top_tokens")
        non_null = output.get("non_null_count")
        text_column = output.get("text_column")
        token_stats = output.get("top_token_stats")
        if top_tokens and non_null and isinstance(token_stats, dict):
            top_word, top_count = next(iter(top_tokens.items()))
            top_stats = token_stats.get(top_word)
            if not isinstance(top_stats, dict):
                return results
            document_count = top_stats.get("document_count")
            document_prevalence = top_stats.get("document_prevalence")
            occurrence_frequency = top_stats.get("occurrence_frequency")
            if isinstance(document_count, int) and isinstance(document_prevalence, (int, float)) and document_prevalence > 0.05:
                results.append(
                    Finding(
                        finding_id=f"{self.name}_top_token",
                        kind="text",
                        headline=(
                            f"'{top_word}' is the most frequent term in '{text_column}' "
                            f"({document_count:,} of {non_null:,} rows; {top_count:,} total occurrences)"
                        ),
                        detail=(
                            f"Vocabulary of {output.get('vocab_size', 0):,} distinct token(s); "
                            f"the term appears in {document_prevalence:.1%} of rows"
                            + (
                                f" and represents {occurrence_frequency:.1%} of retained token occurrences."
                                if isinstance(occurrence_frequency, (int, float))
                                else "."
                            )
                        ),
                        evidence={
                            "top_tokens": top_tokens,
                            "top_token_stats": {top_word: top_stats},
                            "non_null_count": non_null,
                            "total_token_count": output.get("total_token_count"),
                        },
                        measure="document_prevalence",
                        dimension=text_column,
                        level=top_word,
                        effect=round(float(document_prevalence), 4),
                        effect_kind="share",
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
