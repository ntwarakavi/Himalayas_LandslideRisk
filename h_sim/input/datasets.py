"""Registry of every dataset the model uses, with caching and availability checks.

One place that answers three questions for each dataset:

  * **Do I already have it?**  Nothing is ever re-downloaded: every fetch checks
    the local cache first and returns immediately if the file is present.
  * **Can I get it?**         ``check()`` probes the endpoint without downloading.
  * **How do I get it?**      ``fetch()`` downloads only what is missing.

Datasets are grouped by role in the model:

  TERRAIN    elevation, and optionally lithology and land cover -> slope and
                                              catchment area; calibration regions
  CLIMATE    present and future rainfall   -> the recharge field
  INVENTORY  historical landslides         -> fits the soil parameters
  TRIGGER    earthquake shaking            -> the seismic coefficient
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

TERRAIN, CLIMATE, INVENTORY, TRIGGER = "terrain", "climate", "inventory", "trigger"


@dataclass
class Dataset:
    """One data source: where it comes from, where it lands, how big it is."""

    key: str
    name: str
    group: str
    licence: str
    approx_mb: float
    required: bool
    # Relative to data_dir. A directory means "extracted product".
    rel_path: str
    probe_url: Optional[str] = None      # HEAD-able URL for availability check
    note: str = ""
    manual_url: str = ""                 # set when it cannot be auto-fetched

    def local_path(self, data_dir: str) -> str:
        return os.path.join(data_dir, self.rel_path)

    def cached(self, data_dir: str) -> bool:
        p = self.local_path(data_dir)
        if os.path.isdir(p):
            return any(os.scandir(p))
        return os.path.exists(p) and os.path.getsize(p) > 0


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

REGISTRY: List[Dataset] = [
    # ---- TERRAIN ---------------------------------------------------------
    Dataset(
        key="dem", name="Copernicus GLO-90 DEM", group=TERRAIN,
        licence="free, attribution", approx_mb=5.0, required=True,
        rel_path="dem",
        probe_url=("https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com/"
                   "Copernicus_DSM_COG_30_N28_00_E084_00_DEM/"
                   "Copernicus_DSM_COG_30_N28_00_E084_00_DEM.tif"),
        note="per 1-degree tile; drives the slope factor",
    ),
    Dataset(
        key="landcover", name="ESA WorldCover 2021 v200", group=TERRAIN,
        licence="CC BY 4.0", approx_mb=100.0, required=True,
        rel_path="landcover",
        probe_url=("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/"
                   "2021/map/ESA_WorldCover_10m_2021_v200_N27E084_Map.tif"),
        note="per 3-degree tile; drives the vegetation factor",
    ),
    Dataset(
        key="glim", name="GLiM global lithology (full geodatabase)",
        group=TERRAIN, licence="CC BY 3.0", approx_mb=1140.0, required=False,
        rel_path="glim/LiMW_GIS 2015.gdb",
        probe_url="https://www.dropbox.com/s/9vuowtebp9f1iud/LiMW_GIS%202015.gdb.zip?dl=1",
        note="1.2M polygons; without it the coarse 0.5-degree grid is used",
    ),

    # ---- CLIMATE ---------------------------------------------------------
    Dataset(
        key="worldclim", name="WorldClim v2.1 monthly precipitation (30s)",
        group=CLIMATE, licence="CC BY 4.0", approx_mb=1025.0, required=True,
        rel_path="worldclim/30s",
        probe_url="https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_prec.zip",
        note="present-day wetness factor (~1 km)",
    ),
    Dataset(
        key="cmip6", name="WorldClim CMIP6 future precipitation (IPSL-CM6A-LR)",
        group=CLIMATE, licence="CC BY 4.0", approx_mb=131.0, required=False,
        rel_path="worldclim_future",
        probe_url=("https://geodata.ucdavis.edu/cmip6/2.5m/IPSL-CM6A-LR/ssp585/"
                   "wc2.1_2.5m_prec_IPSL-CM6A-LR_ssp585_2041-2060.tif"),
        note="only needed for --climate ssp126|ssp245|ssp370|ssp585",
    ),

    # ---- INVENTORY -------------------------------------------------------
    Dataset(
        key="coolr", name="NASA Global Landslide Catalog / COOLR",
        group=INVENTORY, licence="open", approx_mb=6.0, required=False,
        rel_path="inventory/coolr_points.geojson",
        probe_url=("https://gis.earthdata.nasa.gov/gis05/rest/services/"
                   "Landslides/COOLR_Events_Points/FeatureServer/0?f=json"),
        note="global, but media-report derived: biased towards roads/towns",
    ),
    Dataset(
        key="glc", name="NASA Global Landslide Catalog (authoritative CSV)",
        group=INVENTORY, licence="open", approx_mb=8.5, required=False,
        rel_path="inventory/glc_export.csv",
        probe_url=("https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog"
                   "_Export/Global_Landslide_Catalog_Export_rows.csv"),
        note="11,033 records with location_accuracy; only ~1/3 are placed to "
             "1 km or better - screen before use",
    ),
    Dataset(
        key="gorkha", name="Roback 2018 Gorkha earthquake inventory (Nepal)",
        group=INVENTORY, licence="public domain (USGS)", approx_mb=129.0,
        required=False,
        rel_path="inventory/roback/Roback_Nepal_final_files",
        probe_url="https://www.sciencebase.gov/catalog/item/582c74fbe4b04d580bd377e8?format=json",
        note="24,795 satellite-mapped source areas; best earthquake calibration",
    ),
    Dataset(
        key="farwest", name="Far-Western Nepal multi-temporal inventory",
        group=INVENTORY, licence="CC BY 4.0", approx_mb=24.0, required=False,
        rel_path="inventory/farwest/LandslideInventory_FarWesternNepal",
        probe_url="https://zenodo.org/api/records/4290100",
        note="26,350 polygons, monsoon-triggered; best rainfall calibration",
    ),
    Dataset(
        key="sikkim", name="Southern Sikkim multi-temporal inventory (India)",
        group=INVENTORY, licence="CC BY 4.0", approx_mb=1.0, required=False,
        rel_path="inventory/sikkim",
        probe_url="https://zenodo.org/api/records/8169506",
        note="255 polygons + 185 points, eastern Indian Himalaya",
    ),
    # ---- TRIGGER ---------------------------------------------------------
    Dataset(
        key="pga", name="GEM Global Seismic Hazard Map (PGA, 475-yr)",
        group=TRIGGER, licence="CC BY-SA 4.0", approx_mb=50.0, required=False,
        rel_path="pga/gem_pga_475.tif",
        manual_url="https://www.globalquakemodel.org/product/global-seismic-hazard-map",
        note="MANUAL download; without it use a uniform --pga scenario",
    ),
]

BY_KEY: Dict[str, Dataset] = {d.key: d for d in REGISTRY}


# ---------------------------------------------------------------------------
# Availability checking
# ---------------------------------------------------------------------------

def check_one(ds: Dataset, data_dir: str, probe: bool = True,
              timeout: int = 45) -> Dict[str, object]:
    """Report cache state and (optionally) endpoint reachability."""
    result = {
        "key": ds.key, "name": ds.name, "group": ds.group,
        "required": ds.required, "approx_mb": ds.approx_mb,
        "cached": ds.cached(data_dir), "path": ds.local_path(data_dir),
        "reachable": None, "manual": bool(ds.manual_url), "note": ds.note,
    }
    if result["cached"] or not probe:
        return result           # never probe what we already have
    if ds.manual_url:
        result["reachable"] = False
        return result
    if not ds.probe_url:
        return result
    try:
        import requests
        r = requests.head(ds.probe_url, timeout=timeout, allow_redirects=True)
        if r.status_code >= 400:            # some hosts reject HEAD
            r = requests.get(ds.probe_url, timeout=timeout, stream=True)
            r.close()
        result["reachable"] = r.status_code < 400
        result["status"] = r.status_code
    except Exception as exc:                # noqa: BLE001
        result["reachable"] = False
        result["error"] = f"{type(exc).__name__}"
    return result


def check_all(data_dir: str, probe: bool = True,
              keys: Optional[List[str]] = None) -> List[Dict[str, object]]:
    """Check every dataset (or a subset) and return one report row each."""
    sel = REGISTRY if not keys else [BY_KEY[k] for k in keys if k in BY_KEY]
    return [check_one(d, data_dir, probe=probe) for d in sel]


def format_report(rows: List[Dict[str, object]]) -> str:
    """Human-readable table of dataset availability."""
    lines = []
    order = [TERRAIN, CLIMATE, INVENTORY, TRIGGER]
    titles = {TERRAIN: "TERRAIN  (susceptibility factors)",
              CLIMATE: "CLIMATE  (wetness factor + rainfall trigger)",
              INVENTORY: "INVENTORY (calibration)",
              TRIGGER: "TRIGGER  (earthquake)"}
    for grp in order:
        grp_rows = [r for r in rows if r["group"] == grp]
        if not grp_rows:
            continue
        lines.append(f"\n{titles[grp]}")
        for r in grp_rows:
            if r["cached"]:
                state = "CACHED "
            elif r["manual"]:
                state = "MANUAL "
            elif r["reachable"] is True:
                state = "READY  "
            elif r["reachable"] is False:
                state = "BLOCKED"
            else:
                state = "?      "
            req = "required" if r["required"] else "optional"
            lines.append(f"  [{state}] {r['name'][:52]:52s} "
                         f"{r['approx_mb']:7.0f} MB  {req}")
            if r["note"]:
                lines.append(f"            {r['note']}")
    missing_req = [r for r in rows if r["required"] and not r["cached"]
                   and r["reachable"] is False]
    if missing_req:
        lines.append("\n! required datasets unavailable: "
                     + ", ".join(r["key"] for r in missing_req))
    return "\n".join(lines)
