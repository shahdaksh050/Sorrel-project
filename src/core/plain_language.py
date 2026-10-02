"""
Plain-language rendering of statistical text.

Pure, deterministic helpers that turn analyst notation ("p<0.001", "d=0.8",
"95% CI [1.2, 3.4]", "R2=0.41") into sentences a non-statistician can read.
They run at render time only, over text or finding dicts: nothing here may be
applied to `Finding.headline` or anything else on the finding bus, because
`src.core.findings` builds dedup signatures from the numbers in a headline and
the prompt layer redacts/truncates those same strings.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from typing import Any

_NUM = r"-?(?:\d+(?:,\d{3})*)?\.?\d+(?:[eE]-?\d+)?"
_CAPTION_MAX = 160
_SMALL_N = 30

_SIZE_WORDS = ("negligible", "small", "moderate", "large")
_LINK_WORDS = ("negligible", "weak", "moderate", "strong", "very strong")

#: kind -> (band cut-offs, band words). Cut-offs are the conventional
#: small/medium/large thresholds for each statistic.
_SCALES: dict[str, tuple[tuple[float, ...], tuple[str, ...]]] = {
    "d": ((0.2, 0.5, 0.8), _SIZE_WORDS),
    "eta": ((0.01, 0.06, 0.14), _SIZE_WORDS),
    "v": ((0.1, 0.3, 0.5), _SIZE_WORDS),
    "r": ((0.1, 0.3, 0.5, 0.7), _LINK_WORDS),
    "silhouette": ((0.25, 0.5, 0.7), ("almost no", "weak", "reasonable", "strong")),
}

_KIND_ALIASES: dict[str, str] = {
    "cohens_d": "d", "hedges_g": "d", "d": "d", "g": "d",
    "eta": "eta", "eta_sq": "eta", "eta_squared": "eta", "epsilon_squared": "eta", "eta2": "eta",
    "cramers_v": "v", "v": "v",
    "r": "r", "rho": "r", "spearman": "r", "pearson": "r", "rank_biserial": "r",
    "r2": "r2", "r_squared": "r2",
    "lift": "lift",
    "silhouette": "silhouette",
}

_P_TIERS = (
    (0.001, "a result very unlikely to be down to chance"),
    (0.01, "a result unlikely to be down to chance"),
    (0.05, "a result probably not down to chance"),
)
_P_WEAK = "a result that could easily be down to chance"
_P_SENTENCES = {
    "a result very unlikely to be down to chance": "This is very unlikely to be down to chance.",
    "a result unlikely to be down to chance": "This is unlikely to be down to chance.",
    "a result probably not down to chance": "This is probably not down to chance.",
    _P_WEAK: "This could easily be down to chance, so treat it as a hint rather than a conclusion.",
}

_SAMPLE_KEYS = ("n", "n_level", "sample_size", "n_segment", "n_obs")
_CI_KEYS = (("ci_lower", "ci_upper"), ("ci_low", "ci_high"), ("importance_ci_lower", "importance_ci_upper"))


def _f(text: str) -> float:
    return float(text.replace(",", ""))


def _g(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"


def _band(kind: str, value: float) -> str:
    cuts, words = _SCALES[kind]
    return words[bisect_right(cuts, abs(value))]


def _pct(fraction: float) -> str:
    pct = abs(fraction) * 100
    return "under 1%" if pct < 1 else f"about {pct:.0f}%"


def format_p(p: float | None) -> str:
    """'<0.001' below 0.001, three decimals otherwise, 'n/a' when missing.
    An exact 0.0 is never printed."""
    if p is None:
        return "n/a"
    return "<0.001" if float(p) < 0.001 else f"{float(p):.3f}"


def _p_phrase(op: str, p: float) -> str:
    if op in (">", ">=", "≥"):
        return _P_WEAK
    if op == "=" or p <= 0.05:
        for cut, words in _P_TIERS:
            if p <= cut:
                return words
    return _P_WEAK


def effect_words(kind: str | None, value: float | None) -> str:
    """Magnitude of an effect in words: 'a large effect', 'a strong positive
    relationship', 'explains about 41% of the variation'. Empty when there is
    no effect to describe."""
    if value is None:
        return ""
    canon = _KIND_ALIASES.get((kind or "").lower().replace(" ", "_").replace("'", ""))
    if canon == "d":
        return f"a {_band('d', value)} effect"
    if canon == "eta":
        return f"a {_band('eta', value)} effect that accounts for {_pct(value)} of the variation"
    if canon == "v":
        return f"a {_band('v', value)} association"
    if canon == "r":
        band = _band("r", value)
        if band == "negligible":
            return "almost no relationship"
        return f"a {band} {'negative' if value < 0 else 'positive'} relationship"
    if canon == "r2":
        return f"explains {_pct(value)} of the variation"
    if canon == "lift":
        if abs(value) >= 1:
            return f"about {abs(1 + value):.1f} times as high" if value > 0 else "far lower"
        return f"about {abs(value) * 100:.0f}% {'higher' if value >= 0 else 'lower'}"
    if canon == "silhouette":
        return f"{_band('silhouette', value)} separation between the groups"
    return f"an effect of {_g(value)}"


def _ratio_words(noun: str, ratio: float) -> str:
    if 0.95 <= ratio <= 1.05:
        return f"no real difference in the {noun}"
    if ratio > 1:
        return f"{ratio:.1f} times the {noun}"
    return f"{round((1 - ratio) * 100)}% lower {noun}"


def _stars_kept(match: re.Match[str], phrase: str) -> str:
    """Trailing significance stars are dropped, but `**p<0.01**` is bold
    markdown: when the match itself sits inside `*`, put the stars back."""
    stars = match.group("stars") or ""
    opened = match.start() > 0 and match.string[match.start() - 1] == "*"
    return phrase + (stars if opened else "")


_STARS = r"(?P<stars>\*{1,3})?"
_P_RE = re.compile(
    rf"(?<![\w.])p(?:[\s_-]?(?:adj|adjusted))?(?:\((?:adj|adjusted)\))?\s*(?P<op><=|>=|[<>=≤≥])\s*(?P<val>{_NUM}){_STARS}",
    re.IGNORECASE,
)
# An adjusted p-value answers a different question from the raw one ("does it still hold after
# correcting for the other tests run?"), so it is not rewritten with the raw p-value's phrase.
_P_ADJ_RE = re.compile(
    rf"(?P<lead>\(\s*)?(?P<sep>[,;]\s*)?(?<![\w.])(?:(?:BH|FDR|Holm|Bonferroni)[\s-]*)?"
    rf"(?:(?:adjusted|corrected|adj\.?)\s+p|p[\s_-]?(?:adj|adjusted)|p\((?:adj|adjusted)\))\s*"
    rf"(?P<op><=|>=|[<>=≤≥])\s*(?P<val>{_NUM}){_STARS}(?P<tail>\s*\))?",
    re.IGNORECASE,
)
_ADJ_HOLDS = "still so after correcting for the other tests"
_ADJ_FAILS = "not after correcting for the other tests"


def _adjusted_phrase(m: re.Match[str]) -> str:
    holds = _p_phrase(m.group("op"), _f(m.group("val"))) != _P_WEAK
    phrase = _ADJ_HOLDS if holds else _ADJ_FAILS
    return f"{m.group('lead') or ''}{m.group('sep') or ''}{phrase}{m.group('tail') or ''}"


_P_VALUE_RE = re.compile(rf"(?<![\w.])p[\s_-]?values?\s+(?:of|is|was)\s+(?P<val>{_NUM}){_STARS}", re.IGNORECASE)
_STAT_RE = re.compile(
    rf"(?:(?:(?<=[(,;])|(?<=[,;] )|(?<=\( ))[tFUWz]|(?<![\w.])(?:χ²|chi2|chi-squared?|df))"
    rf"(?:\(\s*[\d.,\s]+\))?\s*=\s*{_NUM}\*{{0,3}}"
)
_CI_RE = re.compile(
    rf"(?:\b\d{{2}}(?:\.\d+)?\s*%\s*)?\b(?:CI|(?i:confidence interval))\s*(?:of\s*)?[=:]?\s*[\[(]?\s*"
    rf"(?P<lo>{_NUM})\s*(?:,|;|–|—|\bto\b|-)\s*(?P<hi>{_NUM})\s*[\])]?"
)
_N_RE = re.compile(r"(?<![\w.])[nN]\s*=\s*(\d+(?:,\d{3})*)")
_BIG_RE = re.compile(r"(?<![\w.,])(?P<sign>-?)(?P<cur>[$€£₹]?)(?P<num>\d{1,3}(?:,\d{3}){2,})(?![\d,]|\.\d)")
_SIG_RE = re.compile(
    r"\b(?P<neg>not\s+)?statistically\s+significant(?P<ly>ly)?\b(?P<noun>\s+(?:difference|effect|relationship|"
    r"correlation|association|increase|decrease|gap|trend|link|result|change)s?\b)?",
    re.IGNORECASE,
)

#: (pattern, separator, kind): a bare letter ("d", "r") only counts with "=",
#: so prose like "Tier d: 5 items" is left alone.
_EFFECT_SPECS: tuple[tuple[str, str, str], ...] = (
    (r"(?i:cohen'?s\s+d|cohens_d|hedges'?\s+g|hedges_g)", r"[=:]", "d"),
    (r"d", "=", "d"),
    (r"(?i:(?:partial\s+)?(?:eta[\s_-]?sq(?:uared)?|eta2|η²|η2|epsilon_squared))", r"[=:]", "eta"),
    (r"(?i:cram[eé]r'?s?[\s_]*v)", r"[=:]", "v"),
    (r"(?:Pearson|Spearman|Kendall)(?:'s)?\s+(?:r|ρ|τ)", r"[=:]", "r"),
    (r"(?i:rho|tau)|ρ|τ|r", "=", "r"),
)
_EFFECT_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"(?<![\w.])(?:{pattern})\s*{sep}\s*(?P<val>{_NUM}){_STARS}"), kind)
    for pattern, sep, kind in _EFFECT_SPECS
)
_R2_RE = re.compile(
    rf"(?<![\w.])(?i:(?:adjusted\s+)?(?:R²|R2|R-squared|R_squared|R\^2))\s*(?:[=:]|of|is)\s*(?P<val>{_NUM}){_STARS}"
)
_RATIO_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern + rf"\s*(?:[=:]|(?i:of|is)\s)\s*(?P<val>{_NUM})"), noun)
    for pattern, noun in (
        (r"(?<![\w.])(?:OR|(?i:odds[\s_-]ratio))", "odds"),
        (r"(?<![\w.])(?:HR|(?i:hazard[\s_-]ratio))", "risk"),
        (r"(?<![\w.])(?:RR|(?i:risk[\s_-]ratio|relative[\s_-]risk))", "risk"),
    )
)


def _humanise(match: re.Match[str]) -> str:
    value = _f(match.group("num"))
    for scale, name in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if value >= scale:
            scaled = value / scale
            text = f"{float(f'{scaled:.3g}'):g} {name}"
            approx = abs(float(f"{scaled:.3g}") - scaled) > 1e-9 * scaled
            cur, sign = match.group("cur"), match.group("sign")
            return f"{sign}{cur}{text}" if cur or not approx else f"about {sign}{text}"
    return match.group(0)


def plainify(text: str) -> str:
    """Rewrite statistics notation inside `text` as plain sentences. Text
    without notation comes back unchanged; numbers that are the point of a
    sentence (rates, counts, amounts) are left as they are."""
    if not text:
        return text
    out = text
    out = _CI_RE.sub(lambda m: f"likely between {m.group('lo')} and {m.group('hi')}", out)
    for pattern, noun in _RATIO_RES:
        def _ratio(m: re.Match[str], noun: str = noun) -> str:
            return _ratio_words(noun, _f(m.group("val")))

        out = pattern.sub(_ratio, out)
    out = _R2_RE.sub(lambda m: _stars_kept(m, f"a fit that {effect_words('r2', _f(m.group('val')))}"), out)
    for pattern, kind in _EFFECT_RES:
        def _effect(m: re.Match[str], kind: str = kind) -> str:
            return _stars_kept(m, effect_words(kind, _f(m.group("val"))))

        out = pattern.sub(_effect, out)
    out = _P_ADJ_RE.sub(_adjusted_phrase, out)
    out = _P_RE.sub(lambda m: _stars_kept(m, _p_phrase(m.group("op"), _f(m.group("val")))), out)
    out = _P_VALUE_RE.sub(lambda m: _stars_kept(m, _p_phrase("=", _f(m.group("val")))), out)
    if _STAT_RE.search(out):
        out = _STAT_RE.sub("", out)
        out = re.sub(r"\(\s*[,;]\s*", "(", out)
        out = re.sub(r"[,;]\s*(?=[,;)])", "", out)
        out = re.sub(r"\s*\(\s*\)", "", out)

    def _significant(m: re.Match[str]) -> str:
        noun = m.group("noun") or ""
        ly = m.group("ly")
        if m.group("neg"):
            phrase = f"unclear{noun}" if noun else "not clearly" if ly else "not clearly different"
        else:
            phrase = f"clear{noun}" if noun else "clearly" if ly else "unlikely to be down to chance"
        return phrase[0].upper() + phrase[1:] if m.group(0)[0].isupper() else phrase

    out = _SIG_RE.sub(_significant, out)
    out = _N_RE.sub(lambda m: m.group(1) if re.match(r"\s*[A-Za-z]", m.string[m.end():]) else f"{m.group(1)} records", out)
    out = _BIG_RE.sub(_humanise, out)
    return out


def _first_number(finding: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    evidence = finding.get("evidence")
    if not isinstance(evidence, dict):
        return None
    values = [
        float(v) for k in keys
        if isinstance(v := evidence.get(k), (int, float)) and not isinstance(v, bool)
    ]
    return min(values) if values else None


def describe_uncertainty(finding: dict[str, Any]) -> str | None:
    """How far to trust a finding, in up to three plain sentences, from its
    p-value, interval, sample size and first caveat. None when it carries
    nothing to say."""
    parts: list[str] = []
    p = finding.get("p_adjusted")
    if p is None:
        p = finding.get("p_value")
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        parts.append(_P_SENTENCES[_p_phrase("=", float(p))])
    evidence = finding.get("evidence")
    if isinstance(evidence, dict):
        for lo_key, hi_key in _CI_KEYS:
            lo, hi = evidence.get(lo_key), evidence.get(hi_key)
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                parts.append(f"The true value is likely between {_g(lo)} and {_g(hi)}.")
                break
    n = _first_number(finding, _SAMPLE_KEYS)
    if n is not None and n < _SMALL_N:
        parts.append(f"It rests on only {n:,.0f} records.")
    caveats = finding.get("caveats")
    if isinstance(caveats, list) and caveats and isinstance(caveats[0], str) and caveats[0].strip():
        parts.append(plainify(caveats[0].strip()).rstrip(".") + ".")
    return " ".join(parts[:3]) or None


def fallback_caption(finding: dict[str, Any]) -> str:
    """One plain sentence (at most 160 characters) describing a chart with no
    model-written caption, built from the finding's headline and effect."""
    headline = str(finding.get("headline") or "").strip()
    if not headline:
        measure, dimension = finding.get("measure"), finding.get("dimension")
        headline = f"{measure} by {dimension}" if measure and dimension else str(measure or dimension or "")
    sentence = re.split(r"(?<=[.!?])\s+", plainify(headline), maxsplit=1)[0].strip().rstrip(".")
    effect = finding.get("effect")
    if sentence and isinstance(effect, (int, float)) and not any(ch.isdigit() for ch in sentence):
        words = effect_words(finding.get("effect_kind"), float(effect))
        if words and not words.startswith("an effect of"):
            sentence = f"{sentence}; {words}"
    if len(sentence) >= _CAPTION_MAX:
        sentence = sentence[: _CAPTION_MAX - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"
        return sentence
    return sentence + "." if sentence else ""
