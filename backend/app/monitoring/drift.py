"""Distribution-shift metrics used by the daily monitoring pass.

Two complementary tools:

* **PSI** (Population Stability Index) — aggregate measure of how much a
  single continuous feature's distribution has shifted. Industry-standard
  thresholds: ``< 0.1`` stable, ``0.1..0.25`` warning, ``> 0.25`` significant
  drift.
* **Two-sample KS** — distribution-free test that returns a statistic and
  p-value. Complements PSI: PSI can miss bimodal shifts with the same mean,
  KS picks those up; KS can miss tail-only shifts on large samples, PSI picks
  those up.

Both operate on 1D numeric arrays. The caller is responsible for splitting a
feature frame into "reference" (training window) and "current" (live window).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class DriftStat:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    n_ref: int
    n_cur: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 6),
            "ks_statistic": round(self.ks_statistic, 6),
            "ks_pvalue": round(self.ks_pvalue, 6),
            "n_ref": self.n_ref,
            "n_cur": self.n_cur,
        }


def psi(
    reference: np.ndarray | list[float],
    current: np.ndarray | list[float],
    *,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index between two 1D samples.

    The reference sample is binned by quantiles; the current sample is binned
    with the same edges. Empty bins are smoothed with ``eps`` to keep the log
    finite (standard trick). Returns a non-negative float; zero means the two
    empirical distributions are identical under the given binning.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size < 2 or cur.size < 2:
        return 0.0

    # Quantile bin edges from the reference. Deduplicate so collapsed bins
    # don't produce zero-width intervals.
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        return 0.0
    # Open the outer edges so points outside the reference range still land in
    # the extreme bins rather than being dropped.
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_frac = ref_counts / ref.size
    cur_frac = cur_counts / cur.size
    ref_frac = np.where(ref_frac == 0, eps, ref_frac)
    cur_frac = np.where(cur_frac == 0, eps, cur_frac)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def ks_two_sample(
    reference: np.ndarray | list[float],
    current: np.ndarray | list[float],
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p-value).

    Falls back to ``(0.0, 1.0)`` for samples too small to evaluate (SciPy
    raises on empty inputs).
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size < 2 or cur.size < 2:
        return 0.0, 1.0
    result = ks_2samp(ref, cur, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def compute_feature_drift(
    reference: dict[str, list[float]],
    current: dict[str, list[float]],
    *,
    bins: int = 10,
) -> list[DriftStat]:
    """Compute PSI + KS per feature. Keys missing from either side are skipped.

    Ignores non-numeric or all-NaN columns; upstream callers should have
    already projected to numeric-only features.
    """
    out: list[DriftStat] = []
    keys = sorted(set(reference) & set(current))
    for key in keys:
        ref_vals = np.asarray(reference[key], dtype=float)
        cur_vals = np.asarray(current[key], dtype=float)
        ref_vals = ref_vals[~np.isnan(ref_vals)]
        cur_vals = cur_vals[~np.isnan(cur_vals)]
        psi_value = psi(ref_vals, cur_vals, bins=bins)
        ks_stat, ks_p = ks_two_sample(ref_vals, cur_vals)
        out.append(
            DriftStat(
                feature=key,
                psi=psi_value,
                ks_statistic=ks_stat,
                ks_pvalue=ks_p,
                n_ref=int(ref_vals.size),
                n_cur=int(cur_vals.size),
            )
        )
    return out
