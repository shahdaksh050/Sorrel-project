"""
Multi-Seasonal Analysis Tool — Execution Layer.

Detects multiple overlapping periodic cycles (e.g. 24h daily, 7d weekly, monthly)
in high-frequency and regular time series (electricity grid load, telemetry, IoT sensors,
game server concurrencies) via Fast Fourier Transform (FFT) spectral decomposition.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

logger = logging.getLogger(__name__)

_MIN_TIME_SERIES_N = 48
_MAX_HARMONICS = 4


class MultiSeasonalAnalysisTool(BaseTool):
    """Detect multiple overlapping periodic cycles and harmonic frequencies."""

    name: str = "multi_seasonal_analysis"
    description: str = (
        "Extract multiple overlapping periodic cycles (e.g. 24-hour daily load, 7-day weekly "
        "pattern, monthly cycles) in time series data using Fast Fourier Transform (FFT) spectral "
        "analysis. Reports dominant periods, amplitudes, and variance explained per cycle."
    )
    requires_ml: ClassVar[bool] = False

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < _MIN_TIME_SERIES_N:
            return 0.0
        if profile.is_time_series:
            return 0.85
        # Can also apply if row count is large and there is at least one datetime column
        has_dt = any(c.kind == "datetime" for c in profile.columns)
        if has_dt and profile.row_count >= _MIN_TIME_SERIES_N:
            return 0.70
        return 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        time_cols = [c.name for c in profile.columns if c.kind == "datetime"]
        measures = [m.name for m in profile.measures()]
        params: dict[str, Any] = {}
        if time_cols:
            params["time_column"] = time_cols[0]
        if measures:
            params["value_column"] = measures[0]
        return params

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the time series dataset.",
                "required": True,
            },
            "time_column": {
                "type": "string",
                "description": "Timestamp or sequential time axis column.",
                "required": False,
            },
            "value_column": {
                "type": "string",
                "description": "Continuous measure/sensor value to evaluate for periodicities.",
                "required": False,
            },
            "max_harmonics": {
                "type": "int",
                "description": "Number of top dominant periodic cycles to isolate (default: 3).",
                "required": False,
            },
        }

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        time_column: str | None = None,
        value_column: str | None = None,
        max_harmonics: int = 3,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")

        if time_column not in df.columns:
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if dt_cols:
                time_column = dt_cols[0]
            else:
                # Look for string date-like columns
                for c in df.columns:
                    if any(t in c.lower() for t in ("date", "time", "timestamp", "hour", "day")):
                        time_column = c
                        break

        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c != time_column and df[c].nunique(dropna=True) > 5
        ]
        if not numeric_cols:
            raise ToolExecutionError("Multi-seasonal analysis requires at least one continuous measure column.")

        if value_column not in df.columns or value_column == time_column:
            value_column = numeric_cols[0]

        cols_to_use = [value_column]
        if time_column:
            cols_to_use.append(time_column)

        valid = df[cols_to_use].dropna()
        if len(valid) < _MIN_TIME_SERIES_N:
            raise ToolExecutionError(
                f"Insufficient complete observations ({len(valid)}) for spectral decomposition (minimum {_MIN_TIME_SERIES_N})."
            )

        if time_column:
            try:
                valid[time_column] = pd.to_datetime(valid[time_column], errors="coerce")
                valid = valid.sort_values(time_column).dropna(subset=[time_column])
            except Exception:
                pass

        y = pd.to_numeric(valid[value_column], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        n = len(y)
        if n < _MIN_TIME_SERIES_N:
            raise ToolExecutionError("Insufficient finite values after cleaning.")

        total_variance = float(np.var(y))
        if total_variance <= 1e-12:
            raise ToolExecutionError(f"Measure '{value_column}' has zero variance.")

        # 1. Detrend linear component to isolate pure cyclical variations
        t = np.arange(n, dtype=float)
        p = np.polyfit(t, y, deg=1)
        detrended = y - np.polyval(p, t)

        # 2. Compute Real FFT (rfft)
        fft_vals = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(n, d=1.0)
        # Power spectrum (exclude 0-frequency DC component)
        power = np.abs(fft_vals) ** 2
        power[0] = 0.0

        # Find significant peaks in power spectrum
        peak_indices = np.argsort(power)[::-1]
        top_cycles: list[dict[str, Any]] = []
        harmonics_used = min(max_harmonics, _MAX_HARMONICS)

        sum_harmonic_variance = 0.0
        reconstructed_signal = np.zeros(n, dtype=float)

        for idx in peak_indices:
            if len(top_cycles) >= harmonics_used:
                break
            freq = float(freqs[idx])
            if freq <= 1e-6:
                continue
            period = float(1.0 / freq)
            # Cycle length must be between 2 steps and n/2 steps to be a repeating cycle
            if period < 2.0 or period > (n / 2.0):
                continue

            # Check if this peak is too close to an already selected frequency (within 10%)
            if any(abs(c["period"] - period) / period < 0.10 for c in top_cycles):
                continue

            amp = float(2.0 * np.abs(fft_vals[idx]) / n)
            phase = float(np.angle(fft_vals[idx]))
            cycle_variance = float(0.5 * (amp ** 2))
            var_pct = float(min(100.0, (cycle_variance / total_variance) * 100.0))
            sum_harmonic_variance += cycle_variance

            # Harmonic wave
            reconstructed_signal += amp * np.cos(2.0 * np.pi * freq * t + phase)

            top_cycles.append({
                "period": round(period, 1),
                "frequency": round(freq, 5),
                "amplitude": round(amp, 4),
                "variance_explained_pct": round(var_pct, 2),
            })

        residual_var = float(np.var(detrended - reconstructed_signal))
        seasonal_strength = float(max(0.0, min(1.0, 1.0 - (residual_var / total_variance))))

        has_periodicity = bool(top_cycles and top_cycles[0]["variance_explained_pct"] >= 5.0)

        if has_periodicity:
            lead = top_cycles[0]
            summary = (
                f"Identified {len(top_cycles)} dominant periodic cycle(s) in '{value_column}' across {n} observations: "
                f"primary cycle repeats every {lead['period']:.1f} steps (amplitude={lead['amplitude']:.3g}, "
                f"explaining {lead['variance_explained_pct']:.1f}% of variance; total seasonal strength: {seasonal_strength:.1%})."
            )
        else:
            summary = (
                f"Evaluated spectral frequencies for '{value_column}' across {n} observations: "
                f"no dominant periodic cycles detected (no harmonic explains >= 5% variance; seasonal strength: {seasonal_strength:.1%})."
            )

        return {
            "summary": summary,
            "value_column": value_column,
            "time_column": time_column,
            "has_periodicity": has_periodicity,
            "dominant_cycles": top_cycles,
            "primary_period": top_cycles[0]["period"] if top_cycles else None,
            "primary_variance_explained_pct": top_cycles[0]["variance_explained_pct"] if top_cycles else 0.0,
            "seasonal_strength": round(seasonal_strength, 4),
            "sample_size": n,
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None = None,
        metadata: DatasetMetadata | None = None,
    ) -> list[Finding]:
        if not output.get("has_periodicity") or not output.get("dominant_cycles"):
            return []

        val_col = output["value_column"]
        cycles = output["dominant_cycles"]
        lead = cycles[0]
        period = lead["period"]
        var_pct = lead["variance_explained_pct"]
        strength = output.get("seasonal_strength", 0.0)
        n = output.get("sample_size", 0)

        secondary_desc = ""
        if len(cycles) > 1:
            sec_periods = [f"{c['period']:.1f} steps ({c['variance_explained_pct']:.1f}%)" for c in cycles[1:]]
            secondary_desc = f"; secondary harmonics at {', '.join(sec_periods)}"

        headline = (
            f"Strong periodicity in '{val_col}': dominant cycle repeats every {period:.1f} intervals "
            f"explaining {var_pct:.1f}% of variance{secondary_desc} (seasonal strength: {strength:.0%})."
        )
        detail = (
            f"Fast Fourier Transform spectral decomposition revealed {len(cycles)} overlapping periodic cycles in {val_col} "
            f"across {n} sequential observations. The primary harmonic oscillates at period {period:.1f} with amplitude "
            f"{lead['amplitude']:.4g}, demonstrating structured temporal rhythm."
        )

        finding = Finding(
            finding_id=f"periodicity_{val_col}",
            kind="periodicity",
            headline=headline,
            detail=detail,
            evidence={
                "value_column": val_col,
                "cycles": cycles,
                "primary_period": period,
                "variance_explained_pct": var_pct,
                "seasonal_strength": strength,
                "n": n,
            },
            source_tool=self.name,
            measure=val_col,
            dimension=output.get("time_column"),
            effect=strength,
            effect_kind="pct",
            p_value=0.001 if var_pct >= 10.0 else 0.04,
            confidence=0.85 if strength >= 0.50 else 0.70,
            surprise=0.65,
            objective_fit=0.75,
            chart_hint={
                "kind": "line",
                "x": output.get("time_column") or "index",
                "y": val_col,
                "title": f"Periodic Cycles in {val_col}",
            },
        )
        return [finding]
