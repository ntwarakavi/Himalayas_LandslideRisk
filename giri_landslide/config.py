"""Configuration and calibration tables for the GIRI landslide model.

Every reclassification table, weight and hazard matrix used by the pipeline is
defined here so the whole model can be inspected and re-calibrated in one place.
Values are transcribed from the GIRI/NGI manuscript (Palau et al. 2023). Where
the manuscript relies on figures whose numeric values are not machine-readable
(the rainfall hazard matrix, Fig. 3) or on internal expert calibration (the
susceptibility factor weights and class breaks, the GLiM lithology mapping), the
defaults below are clearly flagged as *illustrative / calibratable* and can be
overridden from a JSON config file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 3.1  Susceptibility factor reclassification tables
# ---------------------------------------------------------------------------

# Table 2 - slope susceptibility factor Sr.
# Breaks in DEGREES (paper expresses them in 1/100 degree). Each entry is
# (upper_bound_exclusive, factor). The last bound is math.inf.
# Note the non-monotonic tail: very steep slopes (>36 deg) shed sediment and
# become progressively more likely to be bare/hard rock, so the factor drops.
SLOPE_BREAKS_DEG: List[Tuple[float, int]] = [
    (6.0, 0),    # 0-6 deg      flat / near-flat -> negligible
    (12.0, 1),   # 6-12         very low
    (18.0, 2),   # 12-18        low
    (24.0, 3),   # 18-24        moderate
    (30.0, 4),   # 24-30        medium
    (36.0, 5),   # 30-36        high / very high
    (40.0, 4),   # 36-40        probably stiff soil
    (44.0, 3),   # 40-44        probably rock
    (50.0, 2),   # 44-50        probably hard rock
    (float("inf"), 1),  # >50    stable hard rock
]

# Table 5 - land-cover / vegetation susceptibility factor Sv.
# Keyed by ESA WorldCover class codes (the default open land-cover product).
# Values follow the "non-resistance to landslides" logic of the paper's Table 5.
WORLDCOVER_SV: Dict[int, int] = {
    10: 2,   # Tree cover            -> close forest
    20: 3,   # Shrubland            -> shrubs
    30: 5,   # Grassland            -> grassland
    40: 5,   # Cropland             -> agriculture
    50: 1,   # Built-up             -> urban
    60: 5,   # Bare / sparse veg.   -> bare areas
    70: 1,   # Snow and ice         -> permanent ice
    80: 0,   # Permanent water      -> water bodies
    90: 3,   # Herbaceous wetland
    95: 2,   # Mangroves
    100: 4,  # Moss and lichen      -> sparse vegetation
}
WORLDCOVER_SV_NODATA = 0

# Table 5 mapping expressed against Copernicus/ICDR LCCS flag values, kept for
# users who supply the C3S land-cover product used in the manuscript.
LCCS_SV: Dict[int, int] = {
    0: 0,     # no data
    10: 5, 11: 5, 12: 5,        # cropland, rainfed
    20: 5,                       # cropland, irrigated
    30: 4, 40: 3,               # mosaic cropland / natural veg.
    50: 2, 60: 2, 61: 2, 62: 3, # tree cover (broadleaved)
    70: 2, 71: 2, 72: 2,        # tree cover (needleleaved evergreen)
    80: 2, 81: 2, 82: 2,        # tree cover (needleleaved deciduous)
    90: 2,                       # tree cover, mixed
    100: 3, 110: 3,             # mosaic tree/shrub, herbaceous
    120: 3, 121: 4, 122: 3,     # shrubland
    130: 5,                      # grassland
    140: 4, 150: 4, 151: 4, 152: 4, 153: 4,  # lichens / sparse veg.
    160: 2, 170: 2, 180: 3,     # flooded tree cover / shrub or herbaceous
    190: 1,                      # urban
    200: 5, 201: 5, 202: 5,     # bare areas
    210: 0,                      # water bodies
    220: 1,                      # permanent snow and ice
}

# Lithology susceptibility factor Sl (1=low, 2=moderate, 3=high relevance),
# keyed by GLiM level-1 two-letter class code (Hartmann & Moosdorf 2012).
# Expert mapping - editable. 0 means "excluded" (water/ice/no-data).
GLIM_SL: Dict[str, int] = {
    "su": 3,  # unconsolidated sediments
    "ss": 3,  # siliciclastic sedimentary rocks
    "sm": 3,  # mixed sedimentary rocks
    "py": 3,  # pyroclastics
    "va": 3,  # acid volcanic rocks
    "vi": 3,  # intermediate volcanic rocks
    "sc": 2,  # carbonate sedimentary rocks
    "vb": 2,  # basic volcanic rocks
    "mt": 2,  # metamorphics
    "ev": 2,  # evaporites
    "pa": 1,  # acid plutonic rocks
    "pi": 1,  # intermediate plutonic rocks
    "pb": 1,  # basic plutonic rocks
    "ig": 0,  # ice and glaciers
    "wb": 0,  # water bodies
    "nd": 0,  # no data
}
GLIM_SL_DEFAULT = 2  # used for any code not in the table

# Table 3 - soil-moisture factor Sp for RAINFALL-induced landslides, from the
# mean-year-maximum-monthly-rainfall (MYMMR, mm). (upper_bound_inclusive, Sp).
MYMMR_BREAKS_MM: List[Tuple[float, int]] = [
    (125.0, 1),
    (250.0, 2),
    (500.0, 3),
    (1000.0, 4),
    (float("inf"), 5),
]

# Table 4 - soil-moisture factor Sp for EARTHQUAKE-induced landslides, from the
# volumetric water content (VWC, m3/m3). (upper_bound_inclusive, Sp).
VWC_BREAKS: List[Tuple[float, int]] = [
    (0.16, 1),
    (0.36, 2),
    (float("inf"), 3),
]


# ---------------------------------------------------------------------------
# 3.2  Susceptibility combination:  S = product_i ( w_i * f(S_i) )
# ---------------------------------------------------------------------------

@dataclass
class Weights:
    """Multiplicative weights for each susceptibility factor (calibratable).

    The manuscript notes that soil moisture carries a *smaller* weight in the
    earthquake-induced model, so slope dominates there - reflected in the
    earthquake preset below.
    """

    slope: float = 1.0
    lithology: float = 1.0
    vegetation: float = 1.0
    soil_moisture: float = 1.0


# Class breaks applied to the weighted product S to produce the five
# susceptibility categories 1..5 (Very Low .. Very High). Illustrative defaults
# calibratable against a landslide inventory. (upper_bound_inclusive, class).
SUSCEPTIBILITY_BREAKS: List[Tuple[float, int]] = [
    (5.0, 1),    # Very Low
    (15.0, 2),   # Low
    (35.0, 3),   # Moderate
    (75.0, 4),   # High
    (float("inf"), 5),  # Very High
]


# ---------------------------------------------------------------------------
# 4  Triggering conditions
# ---------------------------------------------------------------------------

# 4.1 Rainfall - return periods (years) separating the 5 rainfall hazard
# classes. The 24 h rainfall is normalised as z = (I24 - mu)/sigma and the
# return period is recovered assuming a Gumbel distribution of annual maxima.
RAINFALL_RETURN_PERIODS_YR: List[float] = [5.0, 25.0, 200.0, 1000.0]

# 4.2 Earthquake - PGA thresholds (g) separating the 5 seismic hazard classes.
# PGA < 0.05 g -> negligible triggering (class 0).
PGA_THRESHOLDS_G: List[float] = [0.05, 0.15, 0.25, 0.35, 0.45]


# ---------------------------------------------------------------------------
# 5  Hazard matrices (probability of a significant landslide impacting a 1 km
#     infrastructure stretch, given trigger class x susceptibility class)
# ---------------------------------------------------------------------------

# Figure 4 - EARTHQUAKE hazard matrix. Rows = PGA class 1..5, cols = susc 1..5.
# Values transcribed directly from the manuscript (as fractions, not %).
EARTHQUAKE_MATRIX: List[List[float]] = [
    #  susc1   susc2   susc3   susc4   susc5
    [0.000, 0.000, 0.000, 0.001, 0.005],  # 0.05-0.15 g
    [0.000, 0.000, 0.001, 0.005, 0.010],  # 0.15-0.25 g
    [0.000, 0.001, 0.005, 0.010, 0.050],  # 0.25-0.35 g
    [0.001, 0.005, 0.010, 0.050, 0.100],  # 0.35-0.45 g
    [0.005, 0.010, 0.050, 0.100, 0.400],  # >= 0.45 g
]

# Figure 3 - RAINFALL hazard matrix. The numeric values are published only as a
# figure and are NOT machine-readable in the PDF, so the matrix below is an
# illustrative diagonal calibration with the same structure as the earthquake
# matrix. Re-calibrate against an inventory (paper target: ~400k significant
# rainfall-induced landslides/yr globally). Rows = rainfall class 1..5.
RAINFALL_MATRIX: List[List[float]] = [
    #  susc1    susc2    susc3    susc4    susc5
    [0.0000, 0.0000, 0.0000, 0.0005, 0.0010],  # RP  < 5 yr
    [0.0000, 0.0000, 0.0005, 0.0010, 0.0050],  # 5-25 yr
    [0.0000, 0.0005, 0.0010, 0.0050, 0.0200],  # 25-200 yr
    [0.0005, 0.0010, 0.0050, 0.0200, 0.1000],  # 200-1000 yr
    [0.0010, 0.0050, 0.0200, 0.1000, 0.3000],  # >= 1000 yr
]


# ---------------------------------------------------------------------------
# Region of interest - the Hindu Kush Himalaya (HKH)
# ---------------------------------------------------------------------------

# The study region is the Hindu Kush Himalaya as defined by ICIMOD: an arc of
# mountain ranges spanning eight countries from the Hindu Kush (Afghanistan)
# across the Karakoram, Himalaya and Hengduan Shan. Approximate extent
# (west, south, east, north) in EPSG:4326 degrees.
HKH_BBOX: Tuple[float, float, float, float] = (60.0, 16.0, 105.0, 39.0)

# The eight HKH member countries (as they appear in the NASA COOLR/GLC
# "country_name" field). The bounding box plus this list restricts a global
# inventory to the region; neighbouring lowland countries (e.g. Vietnam, Laos)
# are excluded even though they fall inside the box.
HKH_COUNTRIES = (
    "Afghanistan", "Pakistan", "India", "Nepal", "Bhutan", "Bangladesh",
    "China", "Myanmar",
)

# Backwards-compatible aliases (the project began as a Himalaya-only study).
HIMALAYA_BBOX = HKH_BBOX
HIMALAYA_COUNTRIES = HKH_COUNTRIES


# ---------------------------------------------------------------------------
# Top-level configuration object
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """End-to-end run configuration.

    Attributes
    ----------
    name : str
        Label used for output file naming.
    bbox : (west, south, east, north)
        Area of interest in EPSG:4326 degrees. Keep small for local runs.
    resolution_deg : float
        Target grid resolution in degrees (3 arc-seconds ~= 0.0008333 as in the
        paper). Coarsen (e.g. 0.0025) for quicker local experiments.
    trigger : str
        "rainfall" or "earthquake".
    block_size : int
        Tile/block edge in pixels for memory-bounded windowed processing.
    data_dir / work_dir / out_dir : str
        Locations for raw downloads, intermediate factor rasters and outputs.
    dem_source : str
        "copernicus90" | "copernicus30" | "local".
    landcover_source : str
        "worldcover" | "lccs" | "local".
    """

    name: str = "himalaya"
    bbox: Tuple[float, float, float, float] = (83.0, 27.5, 85.0, 29.0)  # C. Nepal
    resolution_deg: float = 0.0008333333
    trigger: str = "rainfall"

    # Region restriction: the model is scoped to the Hindu Kush Himalaya.
    # AOIs are clipped to `region_bbox` and inventories filtered to it.
    region_bbox: Tuple[float, float, float, float] = HKH_BBOX

    # Susceptibility combination:
    #   "multiplicative"  S = prod_i (w_i * f_i)        (weights only rescale)
    #   "exponent"        S = prod_i (f_i + 1) ** w_i   (weights change ranking;
    #                     required for meaningful weight calibration)
    weight_mode: str = "multiplicative"
    # Class breaks for S -> 1..5:  "fixed" (config table) or "quantile"
    # (equal-area quintiles of S over the AOI; used after calibration).
    classification: str = "fixed"

    # Calibration inputs -----------------------------------------------------
    inventory_path: Optional[str] = None   # landslide inventory CSV/GeoJSON
    calibrated_config: Optional[str] = None

    block_size: int = 1024
    data_dir: str = "data/raw"
    work_dir: str = "data/work"
    out_dir: str = "outputs"

    # Source selection / local overrides -----------------------------------
    # Defaults are chosen for ROBUSTNESS, not for the smallest download:
    #   * copernicus90 - 3 arc-second DEM. Deliberately NOT the 30 m product:
    #     the slope reclassification table (Table 2) was calibrated against
    #     ~90 m slope distributions, and slope statistics are strongly
    #     resolution-dependent in steep terrain, so a finer DEM would silently
    #     bias every slope class. Set "copernicus30" only if you also
    #     re-calibrate SLOPE_BREAKS_DEG.
    #   * glim_full - use the 1.2M-polygon GLiM geodatabase rather than the
    #     0.5-degree grid, which is effectively uniform at hillslope scale.
    #   * worldclim_res "30s" (~1 km) - the finest open monthly climatology;
    #     the coarser products smear orographic rainfall gradients that drive
    #     the soil-moisture factor across whole mountain ranges.
    dem_source: str = "copernicus90"
    dem_path: Optional[str] = None
    landcover_source: str = "worldcover"
    landcover_path: Optional[str] = None
    glim_full: bool = True                    # fetch/use full-resolution GLiM
    worldclim_res: str = "30s"                # 30s | 2.5m | 5m | 10m

    # Climate scenario -------------------------------------------------------
    # "current" uses the WorldClim 1970-2000 baseline. An SSP name switches the
    # soil-moisture factor to downscaled CMIP6 projections, giving a
    # future-climate susceptibility (and hence hazard) map, as in section 3.1 of
    # the manuscript. The triggering rainfall return period is always defined
    # against *today's* climate: the terrain takes centuries to adapt to a new
    # regime, so a "100-year storm" keeps its present-day meaning.
    climate: str = "current"                  # current|ssp126|ssp245|ssp370|ssp585
    climate_period: str = "2061-2080"         # 2021-2040|2041-2060|2061-2080|2081-2100
    climate_model: str = "IPSL-CM6A-LR"       # the model used in the manuscript
    climate_res: str = "2.5m"                 # CMIP6 grid (30s is ~22 GB/file)
    glim_path: Optional[str] = None           # GLiM vector (.shp) or raster
    precip_monthly_dir: Optional[str] = None  # 12 monthly precip rasters (mm)
    vwc_path: Optional[str] = None            # ERA5 volumetric water content
    trigger_path: Optional[str] = None        # 24h rainfall (mm) or PGA (g) grid

    # Trigger-scenario scalars (used when no trigger raster supplied) --------
    scenario_pga_g: float = 0.30              # uniform PGA scenario (g)
    scenario_return_period_yr: float = 100.0  # uniform rainfall RP scenario

    # Calibration tables (defaults from module-level constants) --------------
    weights: Weights = field(default_factory=Weights)
    susceptibility_breaks: List[Tuple[float, int]] = field(
        default_factory=lambda: list(SUSCEPTIBILITY_BREAKS)
    )

    # ---- (de)serialisation ------------------------------------------------
    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        raw = dict(raw)
        for key in ("bbox", "region_bbox"):
            if key in raw and raw[key] is not None:
                raw[key] = tuple(raw[key])
        if "weights" in raw and isinstance(raw["weights"], dict):
            raw["weights"] = Weights(**raw["weights"])
        if "susceptibility_breaks" in raw and raw["susceptibility_breaks"]:
            raw["susceptibility_breaks"] = [
                (float(b), int(c)) for b, c in raw["susceptibility_breaks"]
            ]
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, default=list)

    # Region clipping -------------------------------------------------------
    def clipped_bbox(self) -> Tuple[float, float, float, float]:
        """Intersect the requested AOI with the Himalayan region bounds.

        Raises if the AOI lies entirely outside the region.
        """
        w, s, e, n = self.bbox
        rw, rs, re, rn = self.region_bbox
        cw, cs, ce, cn = max(w, rw), max(s, rs), min(e, re), min(n, rn)
        if cw >= ce or cs >= cn:
            raise ValueError(
                f"AOI {self.bbox} is outside the Hindu Kush Himalaya region "
                f"{self.region_bbox}. This model is restricted to that region.")
        return (cw, cs, ce, cn)

    # Convenience for earthquake weight preset ------------------------------
    def with_earthquake_defaults(self) -> "Config":
        """Return a copy tuned for earthquake-induced landslides.

        Slope dominates; soil-moisture weight is reduced, per the manuscript.
        """
        import copy

        c = copy.deepcopy(self)
        c.trigger = "earthquake"
        c.weights.soil_moisture = 0.5
        return c


DEFAULT_CONFIG = Config()
