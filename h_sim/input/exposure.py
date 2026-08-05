"""What is exposed: settlements and roads.

Two vector layers, fetched for the area of interest and cached like everything
else. They answer "what is there to be harmed", which is the half of risk the
stability model says nothing about.

Sources, in the order they are tried:

**Settlements**
    OpenStreetMap ``place=city|town|village|hamlet`` nodes, via Overpass. Much
    the best village-level coverage in the Himalaya, and carries a
    ``population`` tag where somebody has filled it in. Overpass is a shared
    public service that sheds load under pressure, so failures are expected
    and retried; if it stays down, GeoNames country files are complete for
    populated places and never rate-limit.

**Roads**
    OpenStreetMap ``highway=*`` ways, via Overpass, with geometry. There is no
    good open alternative at segment resolution - Natural Earth's road layer
    exists and is always available, but it carries only trunk routes and is
    generalised to about 1:10 M, so it is a last resort that should be labelled
    as such in anything it produces.

Both are ODbL (OSM) or CC-BY (GeoNames); attribute them in published maps.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

#: Required, not decorative - see :func:`_overpass`.
OVERPASS_HEADERS = {
    "User-Agent": ("h-sim/0.1 (Himalayan Slope Instability Model; "
                   "landslide exposure screening; "
                   "https://github.com/ntwarakavi/Himalayas_LandslideRisk)"),
    "Content-Type": "text/plain; charset=utf-8",
}
GEONAMES_URL = "https://download.geonames.org/export/dump/{iso2}.zip"
NE_ROADS_URL = ("https://naciscdn.org/naturalearth/10m/cultural/"
                "ne_10m_roads.zip")

#: OSM place classes kept, coarsest first. Anything smaller than a hamlet is
#: usually a named field or a farmstead and adds noise without adding exposure.
PLACE_CLASSES = ("city", "town", "village", "hamlet")

#: OSM highway classes kept by default: the classified motorable network,
#: comparable to NH + state highway + district road statistics. Residential,
#: service and track were always excluded; "unclassified" (rural link roads)
#: is excluded by default too - it roughly doubles a hill state's total and
#: makes the ROADS panel read implausibly large against the figures road
#: departments publish. Set config.road_classes to widen or narrow; the
#: cache refetches automatically when a requested class was never fetched.
ROAD_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary")

#: The classes older caches were fetched with, before the default narrowed.
_LEGACY_CLASSES = ROAD_CLASSES + ("unclassified",)

#: Rough population where OSM has no ``population`` tag. Only used to rank
#: settlements when nothing better exists, never reported as a count.
PLACE_DEFAULT_POPULATION = {"city": 100000, "town": 10000,
                            "village": 1000, "hamlet": 100}

EXPOSURE_SOURCE_INFO = (
    "Settlements and roads come from OpenStreetMap via the Overpass API "
    "(ODbL). If Overpass is unavailable, settlements fall back to GeoNames "
    "(CC-BY) and roads to Natural Earth 1:10m, which carries trunk routes "
    "only. Attribute whichever was used."
)


@dataclass
class Settlement:
    name: str
    lon: float
    lat: float
    place: str = "village"
    population: Optional[int] = None
    source: str = "osm"

    def as_dict(self) -> dict:
        return {"name": self.name, "lon": self.lon, "lat": self.lat,
                "place": self.place, "population": self.population,
                "source": self.source}


@dataclass
class Road:
    name: str
    highway: str
    coords: List[Tuple[float, float]] = field(default_factory=list)
    source: str = "osm"


# ---------------------------------------------------------------------------
# Overpass
# ---------------------------------------------------------------------------

def _overpass(query: str, retries: int = 4, timeout: int = 180
              ) -> Optional[dict]:
    """POST an Overpass QL query, trying each mirror with backoff.

    Overpass is a free shared service and refuses work when busy, usually with
    an HTML error page rather than an HTTP error code. Both failure modes are
    treated the same: wait and try again, then move to the next mirror.

    The User-Agent is not optional. Both public mirrors reject anonymous
    clients outright - overpass-api.de with a 406 and kumi.systems with a 429
    whose body asks for a meaningful agent string - and the 429 in particular
    reads as rate limiting, which sends you looking for the wrong problem.
    """
    import requests

    delay = 5.0
    for attempt in range(1, retries + 1):
        for url in OVERPASS_ENDPOINTS:
            try:
                r = requests.post(url, data=query.encode("utf-8"),
                                  headers=OVERPASS_HEADERS, timeout=timeout)
                if r.status_code == 200 and r.text.lstrip().startswith("{"):
                    return r.json()
                reason = ("busy" if "too busy" in r.text or "timeout" in r.text
                          else f"http {r.status_code}")
            except Exception as exc:                     # noqa: BLE001
                reason = f"{type(exc).__name__}"
            print(f"  overpass {url.split('/')[2]}: {reason}")
        if attempt < retries:
            print(f"  retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{retries})")
            time.sleep(delay)
            delay *= 2
    return None


def _bbox_clause(bbox: Sequence[float]) -> str:
    w, s, e, n = bbox
    return f"({s},{w},{n},{e})"


def fetch_settlements_osm(bbox: Sequence[float]) -> Optional[List[Settlement]]:
    classes = "|".join(PLACE_CLASSES)
    q = (f'[out:json][timeout:180];\n'
         f'(node["place"~"^({classes})$"]{_bbox_clause(bbox)};);\n'
         f'out body;')
    data = _overpass(q)
    if data is None:
        return None
    out = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        pop = tags.get("population")
        try:
            pop = int(str(pop).replace(",", "")) if pop else None
        except ValueError:
            pop = None
        out.append(Settlement(
            name=tags.get("name") or tags.get("name:en") or "(unnamed)",
            lon=float(el["lon"]), lat=float(el["lat"]),
            place=tags.get("place", "village"), population=pop, source="osm"))
    return out


def fetch_roads_osm(bbox: Sequence[float],
                    classes: Sequence[str] = ROAD_CLASSES
                    ) -> Optional[List[Road]]:
    cls = "|".join(classes)
    q = (f'[out:json][timeout:300];\n'
         f'(way["highway"~"^({cls})$"]{_bbox_clause(bbox)};);\n'
         f'out geom;')
    data = _overpass(q, timeout=300)
    if data is None:
        return None
    out = []
    for el in data.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        tags = el.get("tags", {})
        out.append(Road(name=tags.get("name") or tags.get("ref") or "",
                        highway=tags.get("highway", "road"),
                        coords=[(float(p["lon"]), float(p["lat"]))
                                for p in geom],
                        source="osm"))
    return out


# ---------------------------------------------------------------------------
# fallbacks
# ---------------------------------------------------------------------------

#: GeoNames feature codes for populated places, coarsest first.
_GEONAMES_PLACE = {"PPLC": "city", "PPLA": "city", "PPLA2": "town",
                   "PPLA3": "town", "PPLA4": "town", "PPL": "village",
                   "PPLX": "village", "PPLL": "hamlet"}

#: ISO2 codes for the HKH member countries, for the GeoNames fallback.
HKH_ISO2 = {"Afghanistan": "AF", "Pakistan": "PK", "India": "IN",
            "Nepal": "NP", "Bhutan": "BT", "Bangladesh": "BD",
            "China": "CN", "Myanmar": "MM"}


def fetch_settlements_geonames(bbox: Sequence[float], data_dir: str,
                               iso2: Sequence[str] = ("NP", "IN", "BT")
                               ) -> List[Settlement]:
    """Populated places from GeoNames country dumps, clipped to the bbox."""
    from .sources import download_file

    w, s, e, n = bbox
    out: List[Settlement] = []
    for code in iso2:
        dest = os.path.join(data_dir, "exposure", f"geonames_{code}.zip")
        got = download_file(GEONAMES_URL.format(iso2=code), dest, timeout=300)
        if not got:
            continue
        with zipfile.ZipFile(got) as zf:
            name = f"{code}.txt"
            if name not in zf.namelist():
                continue
            with zf.open(name) as fh:
                for raw in fh:
                    f = raw.decode("utf-8", "replace").split("\t")
                    if len(f) < 15 or f[6] != "P":
                        continue
                    try:
                        lat, lon = float(f[4]), float(f[5])
                    except ValueError:
                        continue
                    if not (w <= lon <= e and s <= lat <= n):
                        continue
                    try:
                        pop = int(f[14]) or None
                    except ValueError:
                        pop = None
                    out.append(Settlement(
                        name=f[1], lon=lon, lat=lat,
                        place=_GEONAMES_PLACE.get(f[7], "village"),
                        population=pop, source="geonames"))
    return out


def fetch_roads_naturalearth(bbox: Sequence[float],
                             data_dir: str) -> List[Road]:
    """Trunk routes from Natural Earth. Coarse; a last resort."""
    import fiona
    from .sources import download_file

    out_dir = os.path.join(data_dir, "exposure")
    shp = os.path.join(out_dir, "ne_10m_roads.shp")
    if not os.path.exists(shp):
        got = download_file(NE_ROADS_URL,
                            os.path.join(out_dir, "ne_10m_roads.zip"),
                            timeout=300)
        if not got:
            return []
        with zipfile.ZipFile(got) as zf:
            zf.extractall(out_dir)
    if not os.path.exists(shp):
        return []

    w, s, e, n = bbox
    out = []
    with fiona.open(shp) as src:
        for feat in src.filter(bbox=(w, s, e, n)):
            geom = feat["geometry"]
            if not geom or geom["type"] != "LineString":
                continue
            props = feat["properties"]
            out.append(Road(name=props.get("name") or "",
                            highway=(props.get("type") or "road").lower(),
                            coords=[(float(x), float(y))
                                    for x, y in geom["coordinates"]],
                            source="naturalearth"))
    return out


# ---------------------------------------------------------------------------
# cached entry points
# ---------------------------------------------------------------------------

def load_settlements(bbox: Sequence[float], data_dir: str,
                     cache_key: str = "aoi",
                     allow_fallback: bool = True) -> List[Settlement]:
    """Settlements for the AOI, cached to disk after the first fetch."""
    path = os.path.join(data_dir, "exposure", f"settlements_{cache_key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return [Settlement(**d) for d in json.load(fh)]

    got = fetch_settlements_osm(bbox)
    if got is None and allow_fallback:
        print("  Overpass unavailable; falling back to GeoNames")
        got = fetch_settlements_geonames(bbox, data_dir)
    got = got or []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([s.as_dict() for s in got], fh)
    return got


def load_roads(bbox: Sequence[float], data_dir: str, cache_key: str = "aoi",
               classes: Sequence[str] = ROAD_CLASSES,
               allow_fallback: bool = True) -> List[Road]:
    """Roads for the AOI, cached to disk after the first fetch.

    The cache remembers which classes it was fetched with. Asking for a
    subset filters the cached records; asking for a class the cache never
    fetched triggers a refetch - so changing ``road_classes`` in the config
    does what it says without anyone deleting cache files by hand.
    """
    path = os.path.join(data_dir, "exposure", f"roads_{cache_key}.json")
    want = set(classes)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, list):            # pre-metadata cache format
            cached_classes, records = set(_LEGACY_CLASSES), raw
        else:
            cached_classes, records = set(raw["classes"]), raw["roads"]
        if want <= cached_classes:
            return [Road(**d) for d in records
                    if d.get("highway") in want]

    got = fetch_roads_osm(bbox, classes)
    if got is None and allow_fallback:
        print("  Overpass unavailable; falling back to Natural Earth "
              "(trunk routes only, ~1:10M)")
        got = fetch_roads_naturalearth(bbox, data_dir)
    got = got or []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"classes": sorted(want),
                   "roads": [{"name": r.name, "highway": r.highway,
                              "coords": r.coords, "source": r.source}
                             for r in got]}, fh)
    return got
