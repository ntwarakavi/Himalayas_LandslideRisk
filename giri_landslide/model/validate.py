"""Independent validation of a susceptibility map against a held-out inventory.

Calibration tells you how well a model fits the landslides it was trained on.
That is not evidence the map works anywhere else. This module answers the
question that matters for deployment:

    Given landslides the model has never seen - ideally in a different region -
    do they actually fall in the classes the map calls dangerous?

Two complementary measures are reported.

**Frequency ratio per class** is the primary one. For each susceptibility class,

    FR = (share of landslides in the class) / (share of map area in the class)

FR > 1 means landslides are over-represented there. A usable map has FR rising
monotonically from class 1 to class 5: the higher the class, the denser the
landslides. A map can have a respectable AUC and still fail this, which is why
it is checked explicitly.

**AUC** is the probability that a random landslide sits in a higher class than
a random background point. It compresses the whole map into one number and is
useful for comparing maps, but it hides non-monotonicity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from .susceptibility import SUSC_NODATA


@dataclass
class ValidationResult:
    n_landslides: int
    n_classified: int
    class_area_pct: Dict[str, float]        # share of map area, per class
    class_landslide_pct: Dict[str, float]   # share of landslides, per class
    frequency_ratio: Dict[str, float]       # landslide share / area share
    monotonic: bool                         # does FR rise with class?
    auc: float
    pct_landslides_in_top2: float
    pct_area_in_top2: float
    efficiency: float                       # top-2 landslide share / area share
    verdict: str
    warnings: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def is_continuous(susc_path: str) -> bool:
    """True if the raster holds the continuous index rather than 1-5 classes."""
    import rasterio

    with rasterio.open(susc_path) as src:
        return src.dtypes[0].startswith("float")


def quantile_bin_edges(susc_path: str, n_bins: int = 5,
                       block: int = 1024) -> List[float]:
    """Equal-area edges of the continuous index, for a frequency-ratio table.

    Binning is only for reporting: the AUC is computed on the raw index, so no
    information is lost to the bin widths.
    """
    import rasterio

    from ..utility.grid import iter_blocks

    sample = []
    with rasterio.open(susc_path) as src:
        for win in iter_blocks(src.width, src.height, block):
            a = src.read(1, window=win).astype("float64").ravel()
            a = a[(a != src.nodata) & np.isfinite(a) & (a > 0)]
            if a.size:
                sample.append(a[::7])
    vals = np.concatenate(sample) if sample else np.array([0.0, 1.0])
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = sorted({float(v) for v in np.quantile(vals, qs)})
    if len(edges) < n_bins - 1:
        # Ties collapsed the quantiles; spread edges over the distinct values
        # instead so the reporting table keeps its bins.
        distinct = np.unique(vals)
        if distinct.size >= n_bins:
            pick = np.linspace(0, distinct.size - 1, n_bins + 1)[1:-1]
            edges = sorted({float(distinct[int(i)]) for i in pick})
    return edges


def _binned_areas(susc_path: str, edges: List[float],
                  block: int = 1024) -> Dict[int, int]:
    """Pixel count per index bin (bin 1 = lowest)."""
    import rasterio

    from ..utility.grid import iter_blocks

    counts = {k: 0 for k in range(1, len(edges) + 2)}
    with rasterio.open(susc_path) as src:
        for win in iter_blocks(src.width, src.height, block):
            a = src.read(1, window=win).astype("float64")
            a = a[(a != src.nodata) & np.isfinite(a)]
            if not a.size:
                continue
            idx = np.digitize(a, edges) + 1
            for k in counts:
                counts[k] += int((idx == k).sum())
    return counts


def _class_areas(susc_path: str, block: int = 1024) -> Dict[int, int]:
    """Pixel count per susceptibility class over the whole map."""
    import rasterio

    from ..utility.grid import iter_blocks

    counts = {k: 0 for k in range(1, 6)}
    with rasterio.open(susc_path) as src:
        for win in iter_blocks(src.width, src.height, block):
            a = src.read(1, window=win)
            for k in range(1, 6):
                counts[k] += int(np.count_nonzero(a == k))
    return counts


def validate_susceptibility(susc_path: str, inventory_points: np.ndarray,
                            background_points: Optional[np.ndarray] = None,
                            block: int = 1024) -> ValidationResult:
    """Score a susceptibility map against landslides it was not fitted on."""
    from ..input.inventory import sample_factors_at_points

    if len(inventory_points) == 0:
        raise ValueError("no inventory points fall inside the map extent")

    raw = sample_factors_at_points(inventory_points, [susc_path])[:, 0]
    continuous = is_continuous(susc_path)
    if continuous:
        raw = raw[np.isfinite(raw) & (raw != -9999.0)]
        edges = quantile_bin_edges(susc_path, block=block)
        areas = _binned_areas(susc_path, edges, block=block)
        ls = np.digitize(raw, edges) + 1.0    # bin index, 1 = lowest
    else:
        raw = raw[np.isfinite(raw) & (raw != SUSC_NODATA)]
        areas = _class_areas(susc_path, block=block)
        ls = raw
    if len(ls) == 0:
        raise ValueError("inventory points do not overlap valid map pixels")
    total_area = sum(areas.values()) or 1
    total_ls = len(ls)

    area_pct, ls_pct, fr = {}, {}, {}
    for k in range(1, 6):
        a = 100.0 * areas.get(k, 0) / total_area
        n = 100.0 * float((ls == k).sum()) / total_ls
        area_pct[str(k)] = round(a, 3)
        ls_pct[str(k)] = round(n, 3)
        fr[str(k)] = round(n / a, 3) if a > 0 else float("nan")

    # Monotonicity over the classes that actually occupy area.
    present = [k for k in range(1, 6)
               if areas.get(k, 0) > 0 and np.isfinite(fr[str(k)])]
    seq = [fr[str(k)] for k in present]
    monotonic = all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))

    # AUC on the raw values where the map is continuous - binning is only for
    # the reporting table, so no resolution is given up here.
    if background_points is not None and len(background_points):
        bg = sample_factors_at_points(background_points, [susc_path])[:, 0]
        nod = -9999.0 if continuous else SUSC_NODATA
        bg = bg[np.isfinite(bg) & (bg != nod)]
        if continuous:
            auc = _auc_from_classes(raw, bg)
    else:                       # fall back to the map's own class distribution
        bg = np.concatenate([np.full(areas.get(k, 0), k) for k in range(1, 6)
                             if areas.get(k, 0)])
        if len(bg) > 200000:
            bg = bg[:: max(1, len(bg) // 200000)]
    if not continuous or background_points is None or not len(background_points):
        auc = _auc_from_classes(ls, bg)

    top2_ls = ls_pct["4"] + ls_pct["5"]
    top2_area = area_pct["4"] + area_pct["5"]
    efficiency = round(top2_ls / top2_area, 2) if top2_area > 0 else float("nan")

    warnings: List[str] = []
    if total_ls < 100:
        warnings.append(f"only {total_ls} landslides inside the map - the "
                        "class statistics are noisy")
    if not monotonic:
        warnings.append("frequency ratio does not rise monotonically with "
                        "class: the ordering of the classes is not supported "
                        "by this inventory")
    if top2_area > 50:
        warnings.append(f"classes 4-5 cover {top2_area:.0f}% of the map - a "
                        "map that calls half the region dangerous is not "
                        "selective enough to be useful")

    if not np.isfinite(auc):
        verdict = "inconclusive"
    elif monotonic and auc >= 0.75 and efficiency >= 1.5:
        verdict = "good - classes are ordered correctly and selective"
    elif monotonic and auc >= 0.65:
        verdict = "fair - ordering holds but discrimination is modest"
    elif monotonic:
        verdict = "weak - ordering holds but little better than chance"
    else:
        verdict = "fails - class ordering not supported by the inventory"

    return ValidationResult(
        n_landslides=int(len(inventory_points)), n_classified=int(total_ls),
        class_area_pct=area_pct, class_landslide_pct=ls_pct,
        frequency_ratio=fr, monotonic=bool(monotonic), auc=float(auc),
        pct_landslides_in_top2=round(top2_ls, 2),
        pct_area_in_top2=round(top2_area, 2), efficiency=float(efficiency),
        verdict=verdict, warnings=warnings)


def _auc_from_classes(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC on ordinal classes, with ties handled."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype="float64")
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):                       # average ranks within ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def format_report(r: ValidationResult) -> str:
    """Human-readable validation table."""
    lines = ["  class   map area   landslides   freq. ratio",
             "  ----------------------------------------------"]
    for k in range(1, 6):
        fr = r.frequency_ratio[str(k)]
        bar = "#" * min(int(round(fr * 4)), 30) if np.isfinite(fr) else ""
        lines.append(f"    {k}     {r.class_area_pct[str(k)]:6.2f}%    "
                     f"{r.class_landslide_pct[str(k)]:6.2f}%      "
                     f"{fr:5.2f}  {bar}")
    lines.append("")
    lines.append(f"  landslides in classes 4-5 : {r.pct_landslides_in_top2:.1f}%"
                 f"  (those classes cover {r.pct_area_in_top2:.1f}% of the map)")
    lines.append(f"  efficiency                : {r.efficiency:.2f}x "
                 "(>1 means the map concentrates landslides)")
    lines.append(f"  AUC                       : {r.auc:.3f}")
    lines.append(f"  monotonic class ordering  : {'yes' if r.monotonic else 'NO'}")
    lines.append(f"  landslides tested         : {r.n_classified}")
    lines.append(f"\n  VERDICT: {r.verdict}")
    for w in r.warnings:
        lines.append(f"    ! {w}")
    return "\n".join(lines)
