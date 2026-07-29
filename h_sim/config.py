"""H-SIM run configuration, and the model's few free parameters.

The physical model has far fewer knobs than a heuristic index does, because the
form of the response is fixed by mechanics rather than chosen. What remains is
collected here so the whole model can be inspected in one place:

* the ranges the soil parameters are searched over (in ``model.physical``,
  since they belong to the search rather than to a run);
* the two trigger parameters, the rainfall coefficient of variation and the
  pseudo-static fraction of PGA (in ``model.hazard``);
* everything below, which describes a particular run rather than the model.

Fitted parameters are written to a JSON file by the fit step and read back by
later steps, so a run is reproducible from the config plus that file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


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
# Calibration regions
# ---------------------------------------------------------------------------
#
# SINMAP's own answer to spatially varying soils is the calibration region: a
# zoning within which the soil parameters are taken as uniform. Two open
# rasters can supply that zoning, and they carry different physics.

#: GLiM level-1 lithology codes, mapped to small integers so they can be
#: rasterised. Lithology sets the friction angle and the soil cohesion of the
#: weathering products.
GLIM_CODES: Dict[str, int] = {
    "su": 1,   # unconsolidated sediments
    "ss": 2,   # siliciclastic sedimentary rocks
    "sm": 3,   # mixed sedimentary rocks
    "sc": 4,   # carbonate sedimentary rocks
    "py": 5,   # pyroclastics
    "va": 6,   # acid volcanic rocks
    "vi": 7,   # intermediate volcanic rocks
    "vb": 8,   # basic volcanic rocks
    "mt": 9,   # metamorphics
    "ev": 10,  # evaporites
    "pa": 11,  # acid plutonic rocks
    "pi": 12,  # intermediate plutonic rocks
    "pb": 13,  # basic plutonic rocks
    "ig": 14,  # ice and glaciers
    "wb": 15,  # water bodies
    "nd": 0,   # no data
}

#: ESA WorldCover classes are used as-is (10, 20, ... 100). Land cover sets
#: root cohesion, which for a shallow failure surface can be the largest single
#: term: forest roots add strength that grassland does not.
WORLDCOVER_NODATA = 0


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
        Area of interest in EPSG:4326 degrees. Flow routing is not tiled, so
        this bounds memory directly - see the note on ``resolution_deg``.
    resolution_deg : float
        Target grid resolution in degrees. 0.00083333 is 3 arc-seconds, about
        90 m; 0.00027778 is 1 arc-second, about 30 m. Specific catchment area
        is resolution-sensitive by construction, so a change here changes the
        wetness field and the fitted parameters should be refreshed.
    trigger : str
        "rainfall" or "earthquake". Selects which scenario term is applied.
    block_size : int
        Tile edge in pixels for the memory-bounded warping stages.
    """

    name: str = "hkh"
    bbox: Tuple[float, float, float, float] = (84.5, 27.6, 85.3, 28.2)  # Gorkha
    resolution_deg: float = 0.0008333333
    trigger: str = "rainfall"

    # Region restriction: the model is scoped to the Hindu Kush Himalaya.
    # AOIs are clipped to `region_bbox` and inventories filtered to it.
    region_bbox: Tuple[float, float, float, float] = HKH_BBOX

    # Fitting inputs ---------------------------------------------------------
    inventory_path: Optional[str] = None   # landslide inventory CSV/GeoJSON/shp
    #: JSON written by the fit step, holding the soil parameters, the recharge
    #: reference and any per-region parameters. Read back by later steps.
    fitted_params: Optional[str] = None

    #: Which raster zones the calibration regions, if any.
    #: None fits one parameter set for the whole area; "lithology" uses GLiM;
    #: "landcover" uses ESA WorldCover. Regions with too few landslides to
    #: constrain three parameters fall back to the whole-area fit.
    calibration_regions: Optional[str] = None
    #: Minimum mapped landslides required before a region gets its own fit.
    min_region_presence: int = 100

    #: Number of Monte Carlo parameter draws per pixel evaluation. The output
    #: is a probability with resolution 1/n_samples, so 200 resolves 0.5%.
    n_samples: int = 200
    #: Draws used inside the parameter search and cross-validation, where the
    #: cost is paid once per candidate. Lower, deliberately.
    n_samples_fit: int = 60

    # Cross-validation -------------------------------------------------------
    #: Block size for spatial-block cross-validation, in degrees. Must exceed
    #: the range over which terrain and landslide density are correlated, or
    #: the scheme collapses towards a random split. 0.25 deg is about 25 km.
    cv_block_deg: float = 0.25
    cv_folds: int = 5

    # Recharge ---------------------------------------------------------------
    #: Wettest-month precipitation (mm) that the fitted R/T corresponds to.
    #: Recharge is scaled by P / reference per pixel, so this fixes what "1x"
    #: means. Set by the fit step when absent, from the median over the AOI.
    recharge_reference_mm: Optional[float] = None
    #: Turn off to hold recharge uniform across the map, which isolates the
    #: effect of terrain alone.
    spatial_recharge: bool = True

    block_size: int = 1024
    data_dir: str = "data/raw"
    work_dir: str = "data/work"
    out_dir: str = "outputs"

    # Source selection / local overrides -----------------------------------
    #   * copernicus30 - 1 arc-second DEM, the default. The physical model
    #     carries no table calibrated at a coarser resolution, so unlike a
    #     heuristic index it has nothing to invalidate; and flow convergence,
    #     which the wetness term depends on, is exactly what a coarse DEM
    #     smooths away. Use copernicus90 for a quicker first pass.
    #   * glim_full - the 1.2M-polygon GLiM geodatabase rather than the
    #     0.5-degree grid, which is effectively uniform at hillslope scale.
    #   * worldclim_res "30s" (~1 km) - the finest open monthly climatology;
    #     the coarser products smear the orographic rainfall gradients that
    #     drive spatial variation in recharge across whole mountain ranges.
    dem_source: str = "copernicus30"
    dem_path: Optional[str] = None
    landcover_source: str = "worldcover"
    landcover_path: Optional[str] = None
    glim_full: bool = True                    # fetch/use full-resolution GLiM
    worldclim_res: str = "30s"                # 30s | 2.5m | 5m | 10m

    # Climate ----------------------------------------------------------------
    # Climate enters only through the recharge field. "current" is the
    # WorldClim 1970-2000 baseline; an SSP specification switches recharge to
    # downscaled CMIP6 projections, so a wetter future raises R, raises wetness
    # and lowers the factor of safety. See model/climate.py.
    #
    #   climate           the scenario a single stability run is evaluated
    #                     under: "current", or "ssp585:2041-2060"
    #   climate_suite     the scenarios step7 sweeps. The baseline is always
    #                     included, since every future is measured against it.
    #                     The window is climate.DEFAULT_PERIOD - a twenty to
    #                     thirty year planning horizon rather than end of
    #                     century, which is the horizon decisions are taken
    #                     over. Name a later window explicitly to see how bad
    #                     it eventually gets.
    climate: str = "current"
    climate_suite: List[str] = field(
        default_factory=lambda: ["current", "ssp245:2041-2060",
                                 "ssp585:2041-2060"])
    climate_model: str = "IPSL-CM6A-LR"       # a mid-sensitivity CMIP6 model
    climate_res: str = "2.5m"                 # CMIP6 grid (30s is ~22 GB/file)

    glim_path: Optional[str] = None           # GLiM vector (.shp/.gdb) or raster
    precip_monthly_dir: Optional[str] = None  # 12 monthly precip rasters (mm)
    pga_path: Optional[str] = None            # PGA raster (g) for seismic runs

    # Trigger-scenario scalars (used when no trigger raster is supplied) ------
    scenario_pga_g: float = 0.30              # uniform PGA scenario (g)
    scenario_return_period_yr: float = 100.0  # uniform rainfall RP scenario
    #: Coefficient of variation of annual maximum 24 h rainfall. The one
    #: trigger parameter not derived from data here; see model/hazard.py.
    rainfall_cv: float = 0.30
    #: Fraction of PGA used as the pseudo-static seismic coefficient.
    pga_fraction: float = 0.5

    #: "probability" (continuous 0-1), "classes" (SINMAP 1-6), or "both".
    output: str = "both"

    #: Extra outputs that cost almost nothing once the fit exists.
    write_critical_acceleration: bool = True

    # Exposure and reach (step10) -------------------------------------------
    #: Angle of reach in degrees. Debris from a source can reach a target if
    #: the line between them is steeper than this. Reported values cluster at
    #: 11-25 deg for channelised debris flows; 18 is towards the conservative
    #: (longer reach) end, which suits a screening product.
    travel_angle_deg: float = 18.0
    #: How far upslope to search for sources, in metres.
    reach_radius_m: float = 2000.0
    #: Length of the pieces roads are cut into before scoring.
    road_segment_m: float = 500.0
    #: OSM highway classes to fetch. Adding residential and track multiplies
    #: the segment count by an order of magnitude.
    road_classes: List[str] = field(
        default_factory=lambda: ["motorway", "trunk", "primary", "secondary",
                                 "tertiary", "unclassified"])
    #: Climate scenarios each settlement and road segment is scored under.
    #: The present day plus both CMIP6 windows inside the planning horizon -
    #: 2021-2040 and climate.DEFAULT_PERIOD - under an intermediate and a very
    #: high pathway. Two windows rather than one because the trajectory across
    #: the horizon is the question, and two pathways because the spread between
    #: them at a fixed date is the honest measure of how much of the change is
    #: a modelling choice. Four futures is four downloads and four stability
    #: runs; cut it with --risk-climate if that is too much.
    risk_climate: List[str] = field(
        default_factory=lambda: ["current",
                                 "ssp245:2021-2040", "ssp585:2021-2040",
                                 "ssp245:2041-2060", "ssp585:2041-2060"])

    # Regional sweep (step9) -------------------------------------------------
    #: Polygon layer of states and provinces. None downloads Natural Earth
    #: admin-1; supply a national dataset instead if you have one.
    admin_path: Optional[str] = None
    #: Countries to sweep. None means every HKH member country.
    admin_countries: Optional[List[str]] = None
    #: How far outside a unit to route flow before clipping the map back to it.
    #: A provincial border cuts catchments, so a cell just inside one is handed
    #: too little upslope area if the DEM stops at the border.
    #:
    #: Measured rather than assumed (analysis/07_boundary_buffer.py): with no
    #: buffer the damage is confined to the outer ring - 1% of cells lose more
    #: than half their catchment area, and 0.4% shift failure probability by
    #: more than 0.05. A buffer of 0.028 deg (3 km) removed it entirely. The
    #: effect is small because hillslope contributing areas are hundreds of
    #: metres, and the cells with genuinely long flow paths are valley floors
    #: already saturated at w = 1, where more water changes nothing.
    #:
    #: 0.05 deg (5.5 km) is roughly twice what was needed on steep crystalline
    #: terrain, as margin for flatter ground with longer flow paths. Raising it
    #: is cheap insurance only up to a point: buffering a 1x1 deg province by
    #: 0.25 deg would grow it to 2.25x the cells for no measured gain.
    admin_buffer_deg: float = 0.05
    #: Units larger than this are reported and skipped rather than attempted.
    #: Flow routing holds the area in memory, so the alternative is an
    #: out-of-memory kill part-way through a multi-day sweep.
    admin_max_cells: int = 40_000_000

    #: Scenarios step6 evaluates: rainfall return periods (years) and peak
    #: ground accelerations (g). Each produces its own map.
    return_periods_yr: List[float] = field(
        default_factory=lambda: [10.0, 100.0, 1000.0])
    pga_scenarios_g: List[float] = field(default_factory=lambda: [0.15, 0.35])

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
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, default=list)

    # Region clipping -------------------------------------------------------
    def clipped_bbox(self) -> Tuple[float, float, float, float]:
        """Intersect the requested AOI with the Hindu Kush Himalaya bounds.

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

    # Cost guard ------------------------------------------------------------
    def cell_count(self) -> int:
        """Pixels in the AOI at the configured resolution."""
        w, s, e, n = self.clipped_bbox()
        return int(round((e - w) / self.resolution_deg)) * \
            int(round((n - s) / self.resolution_deg))


DEFAULT_CONFIG = Config()
