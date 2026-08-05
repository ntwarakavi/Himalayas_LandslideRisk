"""Build a standalone Leaflet page from a run's outputs.

Everything the model produces is a GeoTIFF or a JSON file, which is the right
format for analysis and the wrong one for looking at. This turns a finished run
into a single HTML file plus a folder of assets that opens in any browser.

What goes on the map
--------------------

* the susceptibility raster, rendered to a PNG and stretched over its bounds;
* the landslide inventory the parameters were fitted to, and the background
  points they were fitted against - so the training data is visible next to
  what it produced, rather than being taken on trust;
* settlements, coloured by the susceptibility that can reach them;
* road segments, likewise.

Climate scenarios
-----------------

Every layer that depends on the stability model - the raster, the settlements,
the road segments - is carried for each climate the run was scored under, and a
selector switches between them. Each feature keeps its whole set of scores, so
switching is a restyle rather than a reload, and a popup can show a settlement's
present-day exposure next to its projected exposure without another request.

The present day is always the reference. Change columns are differences from it,
which is the only comparison that means anything: a future field is normalised by
the present-day recharge reference, so its absolute value is only interpretable
relative to today.

Leaflet is loaded from a CDN and the basemap tiles come from OpenStreetMap, so
the page needs a network connection to draw. The model outputs themselves are
written alongside it and load from disk.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional, Sequence

import numpy as np

#: Colour ramp for failure probability, low to high. Perceptually ordered and
#: legible for the common forms of colour blindness: it varies in lightness
#: throughout, so it survives being printed in grey.
PROB_COLOURS = [
    (0.00, (247, 251, 255)), (0.20, (198, 219, 239)),
    (0.40, (107, 174, 214)), (0.60, (253, 174, 97)),
    (0.80, (240, 89, 40)), (1.00, (153, 0, 13)),
]

#: Matching colours for the discrete asset bands in model.risk.
BAND_COLOURS = {"very low": "#2c7bb6", "low": "#abd9e9",
                "moderate": "#fdae61", "high": "#e66101",
                "very high": "#a50026"}


def _ramp(values: np.ndarray) -> np.ndarray:
    """Map 0-1 values to RGB using PROB_COLOURS."""
    stops = np.array([c[0] for c in PROB_COLOURS])
    cols = np.array([c[1] for c in PROB_COLOURS], dtype="float64")
    v = np.clip(values, 0.0, 1.0)
    idx = np.clip(np.searchsorted(stops, v, side="right") - 1, 0,
                  len(stops) - 2)
    lo, hi = stops[idx], stops[idx + 1]
    t = np.where(hi > lo, (v - lo) / np.maximum(hi - lo, 1e-9), 0.0)
    return (cols[idx] + (cols[idx + 1] - cols[idx]) * t[..., None])


def raster_to_png(tif_path: str, png_path: str,
                  max_px: int = 2000) -> Optional[Dict[str, float]]:
    """Render a probability GeoTIFF to a transparent PNG. Returns its bounds.

    Downsampled so the browser is not asked to hold a 6-megapixel overlay;
    ``max_px`` caps the longer side. Nodata becomes fully transparent, so the
    basemap shows through where the model said nothing.
    """
    import rasterio
    from rasterio.enums import Resampling

    try:
        from PIL import Image
    except ImportError:
        Image = None

    with rasterio.open(tif_path) as src:
        scale = min(1.0, max_px / max(src.width, src.height))
        w = max(int(src.width * scale), 1)
        h = max(int(src.height * scale), 1)
        arr = src.read(1, out_shape=(h, w), resampling=Resampling.average)
        nod = src.nodata
        b = src.bounds

    a = arr.astype("float64")
    valid = np.isfinite(a) if nod is None else (np.isfinite(a) & (a != nod))
    rgb = _ramp(np.where(valid, a, 0.0))
    rgba = np.dstack([rgb, np.where(valid, 205, 0)[..., None]]).astype("uint8")

    if Image is None:
        _write_png_numpy(rgba, png_path)
    else:
        Image.fromarray(rgba, mode="RGBA").save(png_path)
    return {"west": b.left, "south": b.bottom, "east": b.right, "north": b.top}


def _write_png_numpy(rgba: np.ndarray, path: str) -> None:
    """Minimal PNG writer, so Pillow stays optional."""
    import struct
    import zlib

    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)


# ---------------------------------------------------------------------------
# GeoJSON assembly
# ---------------------------------------------------------------------------

#: Coordinate precision written to the page. Five decimal places is about a
#: metre, which is finer than the 30 m grid the scores came from; full float
#: repr triples the file for no visible difference.
COORD_DP = 5

#: Fields kept for scenarios other than the present day. The rest - the source
#: cell count, its relief and its distance - are properties of the reach
#: geometry, which does not change with climate, so carrying them per scenario
#: multiplies the page size to restate the same numbers.
SCENARIO_FIELDS = ("score", "on_site", "reaching", "reaching_max",
                   "delivering_m2")

#: Written once at the top level and read from there; anything else is either
#: a per-scenario score or something the page derives. The mechanism flags and
#: footprint geometry are invariant across scenarios, so they live here too.
IDENTITY_FIELDS = ("name", "place", "population", "highway", "segment",
                   "length_m", "source",
                   "cut_slope", "cut_slope_deg", "washout", "washout_sca_m",
                   "footprint_m", "n_cells")


def _thin(rec: dict, baseline: str = "current") -> dict:
    """Keep what the page reads and nothing else.

    A road segment carries five scenario records; at 18,000 segments the
    difference between writing everything and writing what is read is tens of
    megabytes, which is the difference between a page that opens and one that
    does not. Two things go: the top-level copy of the present-day scores,
    which duplicates ``scenarios[baseline]``, and the band label, which the
    page recomputes from the score.
    """
    out = {k: rec[k] for k in IDENTITY_FIELDS if k in rec}
    scen = rec.get("scenarios")
    if scen:
        out["scenarios"] = {
            sk: {f: sv[f] for f in
                 (SCENARIO_FIELDS if sk != baseline
                  else [f for f in sv if f != "band"])
                 if f in sv}
            for sk, sv in scen.items()}
    else:
        out.update({k: v for k, v in rec.items()
                    if k not in ("lon", "lat", "coords", "band")})
    return out


def _round(xy) -> List[float]:
    return [round(float(xy[0]), COORD_DP), round(float(xy[1]), COORD_DP)]


#: Douglas-Peucker tolerance for road geometry on the page, in metres. Half a
#: 30 m cell: finer than the grid the scores were computed on, so nothing
#: visible is lost, while OSM's native node density - about 23 vertices per
#: 500 m segment - is roughly halved.
SIMPLIFY_TOLERANCE_M = 15.0


def simplify(coords: Sequence[Sequence[float]],
             tolerance_m: float = SIMPLIFY_TOLERANCE_M
             ) -> List[Sequence[float]]:
    """Douglas-Peucker, iterative, in approximate local metres.

    Degrees are converted to metres with a fixed scale for the line's mean
    latitude. Over a 500 m segment that is exact enough for a display
    tolerance, and it avoids a projection dependency for a cosmetic step.
    """
    if len(coords) < 3 or tolerance_m <= 0:
        return list(coords)

    import math
    lat0 = sum(c[1] for c in coords) / len(coords)
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    pts = [(c[0] * kx, c[1] * ky) for c in coords]

    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = pts[i]
        bx, by = pts[j]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        best, at = -1.0, -1
        for k in range(i + 1, j):
            px, py = pts[k]
            d = (abs(dy * px - dx * py + bx * ay - by * ax) / norm if norm > 0
                 else math.hypot(px - ax, py - ay))
            if d > best:
                best, at = d, k
        if best > tolerance_m:
            keep[at] = True
            stack.append((i, at))
            stack.append((at, j))
    return [c for c, k in zip(coords, keep) if k]


def points_geojson(rows: Sequence[dict], lon_key: str = "lon",
                   lat_key: str = "lat",
                   props: Optional[Sequence[str]] = None,
                   baseline: str = "current") -> dict:
    feats = []
    for r in rows:
        p = ({k: r.get(k) for k in props} if props else _thin(r, baseline))
        feats.append({"type": "Feature",
                      "properties": p,
                      "geometry": {"type": "Point",
                                   "coordinates": _round((r[lon_key],
                                                          r[lat_key]))}})
    return {"type": "FeatureCollection", "features": feats}


def lines_geojson(rows: Sequence[dict], baseline: str = "current",
                  tolerance_m: float = SIMPLIFY_TOLERANCE_M) -> dict:
    """Scored road segments as a FeatureCollection.

    The geometry is simplified for display only. Scores were computed on the
    full vertex list before this ran, so nothing here changes a number - it
    changes how many points the browser has to draw.
    """
    feats = []
    for r in rows:
        feats.append({"type": "Feature",
                      "properties": _thin(r, baseline),
                      "geometry": {"type": "LineString",
                                   "coordinates": [
                                       _round(c) for c in
                                       simplify(r["coords"], tolerance_m)]}})
    return {"type": "FeatureCollection", "features": feats}


def inventory_geojson(points: np.ndarray, label: str) -> dict:
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"kind": label},
         "geometry": {"type": "Point", "coordinates": [float(x), float(y)]}}
        for x, y in points]}


def write_data(out_dir: str, name: str, obj) -> str:
    """Write a layer as a JavaScript assignment rather than a .json file.

    ``fetch`` of a sibling file is blocked by the same-origin policy when a
    page is opened from ``file://``, which is how anybody actually opens a
    local map: double-click it. A ``<script src>`` is not blocked, so the data
    ships as an assignment into a namespace the page reads. The cost is a few
    bytes of wrapper; the benefit is that the map works without a web server.

    The payload is a **string** handed to ``JSON.parse``, not an object
    literal. A JavaScript engine parses an object literal through its full
    expression grammar; ``JSON.parse`` runs a dedicated parser over a flat
    string. Measured on the Gorkha road layer - 18,109 segments - that is
    460 ms against 255 ms, for about 10 % more bytes from the escaping.
    """
    path = os.path.join(out_dir, f"{name}.js")
    # Encoding twice yields a JSON string literal, which is also a valid
    # JavaScript one: every escape JSON produces is an escape JS understands.
    payload = json.dumps(json.dumps(obj))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("window.HSIM_DATA=window.HSIM_DATA||{};\n"
                 f"window.HSIM_DATA[{json.dumps(name)}]="
                 f"JSON.parse({payload});\n")
    return f"{name}.js"


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

#: Leaflet, fetched once and kept beside the page. Pinned rather than floating,
#: so a run packaged today draws the same way in a year.
LEAFLET_VERSION = "1.9.4"
LEAFLET_BASE = f"https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist"
#: Referenced by leaflet.css, so the layers control and default marker draw.
LEAFLET_IMAGES = ("layers.png", "layers-2x.png", "marker-icon.png",
                  "marker-icon-2x.png", "marker-shadow.png")


def vendor_leaflet(out_dir: str, cache_dir: Optional[str] = None) -> bool:
    """Copy Leaflet next to the page. False if it could not be obtained.

    Loading the library from a CDN makes the whole page - sidebar included -
    depend on a network the user may not have. Institutional networks in the
    region this model is for block plenty; the map degrades to missing basemap
    tiles when offline, which is tolerable, but a blank page is not.
    """
    import shutil as _sh

    dest = os.path.join(out_dir, "leaflet")
    os.makedirs(os.path.join(dest, "images"), exist_ok=True)
    cache = os.path.join(cache_dir or out_dir, "_leaflet_cache")

    def fetch(rel: str, target: str) -> bool:
        cached = os.path.join(cache, rel.replace("/", "_"))
        if os.path.exists(cached) and os.path.getsize(cached) > 0:
            _sh.copyfile(cached, target)
            return True
        try:
            import requests
            r = requests.get(f"{LEAFLET_BASE}/{rel}", timeout=60)
            if r.status_code != 200 or not r.content:
                return False
        except Exception:                                # noqa: BLE001
            return False
        os.makedirs(cache, exist_ok=True)
        with open(cached, "wb") as fh:
            fh.write(r.content)
        _sh.copyfile(cached, target)
        return True

    if not fetch("leaflet.js", os.path.join(dest, "leaflet.js")):
        return False
    if not fetch("leaflet.css", os.path.join(dest, "leaflet.css")):
        return False
    for img in LEAFLET_IMAGES:
        fetch(f"images/{img}", os.path.join(dest, "images", img))
    return True


def basemaps(maptiler_key: Optional[str] = None) -> List[dict]:
    """The base layers the page offers, first one shown by default.

    "DataViz Light" is the default: muted, light cartography keeps the
    probability raster and exposure colours legible, which the busier street
    and terrain maps fight. It is MapTiler's dataviz-light style when a key is
    supplied; without one it falls back to CARTO's keyless light tiles - the
    same style of cartography - with the attribution naming whichever provider
    actually served it.
    """
    if maptiler_key:
        light = {
            "name": "DataViz Light",
            "url": ("https://api.maptiler.com/maps/dataviz-light/"
                    "{z}/{x}/{y}.png?key=" + maptiler_key),
            "options": {"maxZoom": 20,
                        "attribution": "&copy; MapTiler, "
                                       "&copy; OpenStreetMap contributors"}}
    else:
        light = {
            "name": "DataViz Light",
            "url": "https://{s}.basemaps.cartocdn.com/light_all/"
                   "{z}/{x}/{y}{r}.png",
            "options": {"maxZoom": 19, "subdomains": "abcd",
                        "attribution": "&copy; OpenStreetMap contributors, "
                                       "&copy; CARTO"}}
    return [
        light,
        {"name": "OpenStreetMap",
         "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
         "options": {"maxZoom": 18,
                     "attribution": "&copy; OpenStreetMap contributors"}},
        {"name": "Terrain",
         "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
         "options": {"maxZoom": 17,
                     "attribution": "&copy; OpenTopoMap, "
                                    "&copy; OpenStreetMap contributors"}},
    ]


def build(out_dir: str, title: str, bounds: Dict[str, float],
          layers: Dict[str, str], summary: Optional[dict] = None,
          meta: Optional[dict] = None,
          cache_dir: Optional[str] = None,
          data_files: Sequence[str] = (),
          maptiler_key: Optional[str] = None) -> str:
    """Write index.html next to the assets in ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    local = vendor_leaflet(out_dir, cache_dir)
    src = "leaflet" if local else LEAFLET_BASE
    html = _PAGE.replace("__TITLE__", title)
    html = html.replace("__LEAFLET__", src)
    html = html.replace("__DATA__", "\n".join(
        f'<script src="{f}"></script>' for f in data_files))
    html = html.replace("__BOUNDS__", json.dumps(bounds))
    html = html.replace("__BASEMAPS__", json.dumps(basemaps(maptiler_key)))
    html = html.replace("__LAYERS__", json.dumps(layers))
    html = html.replace("__SUMMARY__", json.dumps(summary or {}))
    html = html.replace("__META__", json.dumps(meta or {}))
    html = html.replace("__BANDS__", json.dumps(BAND_COLOURS))
    from .model.risk import RISK_BANDS
    html = html.replace("__EDGES__", json.dumps([list(b) for b in RISK_BANDS]))
    html = html.replace("__RAMP__", json.dumps(
        [[s, f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"]
         for s, c in PROB_COLOURS]))
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="__LEAFLET__/leaflet.css"/>
<script src="__LEAFLET__/leaflet.js"></script>
<style>
  :root {
    --bg: #ffffff; --fg: #16191d; --muted: #5b6470;
    --line: #dfe3e8; --panel: #f7f8fa; --accent: #a50026;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14171a; --fg:#e8eaed; --muted:#9aa4b2;
            --line:#2b3038; --panel:#1c2025; --accent:#ff6b57; }
  }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; font: 14px/1.5 -apple-system,
    BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--fg); }
  #app { position: relative; height: 100%; }
  #map { position: absolute; inset: 0; }
  #side { position: absolute; top: 12px; right: 12px; bottom: 12px;
          width: 340px; z-index: 1000; overflow-y: auto;
          background: var(--panel); border: 1px solid var(--line);
          border-radius: 10px; box-shadow: 0 8px 28px rgba(0,0,0,.25);
          opacity: 0.97; }
  @media (max-width: 900px) {
    #side { top: auto; left: 12px; right: 12px; bottom: 12px;
            width: auto; max-height: 45%; }
  }
  .pad { padding: 14px 16px; }
  h1 { font-size: 16px; margin: 0 0 2px; letter-spacing: -0.01em; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.07em;
       color: var(--muted); margin: 20px 0 8px; font-weight: 600; }
  .sub { color: var(--muted); font-size: 12px; margin: 0; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 600; font-size: 11px;
       text-transform: uppercase; letter-spacing: 0.05em; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .chip { display:inline-block; width:10px; height:10px; border-radius:2px;
          margin-right:6px; vertical-align: -1px; }
  .ramp { height: 12px; border-radius: 3px; margin: 6px 0 4px;
          border: 1px solid var(--line); }
  .ends { display:flex; justify-content:space-between; color:var(--muted);
          font-size:11px; }
  .note { font-size:12px; color: var(--muted); border-left: 3px solid var(--line);
          padding: 8px 10px; margin-top: 10px; background: var(--bg); }
  .leaflet-popup-content { font: 13px/1.45 inherit; }
  .leaflet-popup-content b { font-size: 13px; }
  .kv { color: var(--muted); }
  .scroll { max-height: 260px; overflow-y: auto; }
  .wrap { overflow-x: auto; }
  select { width:100%; padding:7px 8px; font:inherit; font-size:13px;
           color:var(--fg); background:var(--bg); border:1px solid var(--line);
           border-radius:4px; }
  .up { color: var(--accent); }
  .down { color: #2c7bb6; }
  .pill { display:inline-block; padding:1px 6px; border-radius:9px;
          font-size:11px; background:var(--line); color:var(--muted);
          margin-left:6px; }
  details.gloss { margin-top:16px; font-size:12px; }
  details.gloss summary { cursor:pointer; font-size:12px;
      text-transform:uppercase; letter-spacing:.07em; color:var(--muted);
      font-weight:600; }
  details.gloss dl { margin:10px 0 0; }
  details.gloss dt { font-weight:600; margin-top:8px; }
  details.gloss dd { margin:1px 0 0 0; color:var(--muted); }
  .mechicon { background:none; border:none; }
  .glyph { vertical-align:-2px; margin-right:6px; }
</style>
</head>
<body>
<div id="app">
  <div id="map"></div>
  <aside id="side">
    <div class="pad">
      <h1>__TITLE__</h1>
      <p class="sub" id="meta"></p>

      <div id="picker"></div>

      <h2>Failure probability</h2>
      <div class="ramp" id="ramp"></div>
      <div class="ends"><span>0 &middot; stable</span><span>1 &middot; unstable</span></div>

      <h2>Exposure bands</h2>
      <div id="bands"></div>
      <label class="sub" style="display:block;margin-top:6px;cursor:pointer">
        <input type="checkbox" id="bandtoggle" checked>
        colour settlements and roads by exposure
      </label>

      <div id="mechlegend"></div>

      <div id="stats"></div>
      <div id="compare"></div>
      <div id="clusters"></div>
      <div id="worst"></div>

      <details class="gloss"><summary>Glossary &mdash; every term, and
      every % of what</summary><dl>
      <dt>Failure probability (the raster)</dt>
      <dd>The share of Monte Carlo soil-parameter draws that push a cell's
      factor of safety below 1. A <i>relative</i> ranking under parameter
      uncertainty &mdash; never an annual chance. "Unstable" anywhere on this
      page means probability &ge; 0.5.</dd>
      <dt>Unstable area %</dt>
      <dd>% of the grid cells <i>inside this province's boundary</i> with
      failure probability &ge; 0.5.</dd>
      <dt>Mean P &middot; P90</dt>
      <dd>Mean and 90th-percentile per-cell failure probability over the same
      cells.</dd>
      <dt>Exposure (the score on every asset)</dt>
      <dd>The greater of <b>on site</b> and <b>reaching</b>, 0&ndash;1.</dd>
      <dt>on site</dt>
      <dd>Failure probability of the cell the asset stands on.</dd>
      <dt>reaching</dt>
      <dd>Of the upslope ground within 2&nbsp;km that can reach the asset
      along an 18&deg; travel line, the fraction the model calls unstable,
      weighted toward nearer sources (1/distance). A % <i>of that reachable
      ground</i>, not of the province.</dd>
      <dt>exposed</dt>
      <dd>Score at or above <span class="glothr">0.08</span>. The threshold
      is a screening convention, not a physical constant.</dd>
      <dt>bands</dt>
      <dd>The continuous score cut at 0.02 / 0.08 / 0.20 / 0.40 for the
      legend. "Moderate" is the quantitative statement that roughly a fifth
      of the reachable ground is called unstable.</dd>
      <dt>worst source</dt>
      <dd>The single most unstable reachable cell: its height above the asset
      and its horizontal distance.</dd>
      <dt>unstable supply</dt>
      <dd>Expected unstable area positioned to reach the asset:
      &Sigma; probability &times; cell area, in hectares. Separates thirty
      threatening cells from three thousand at the same mean. Relative, like
      everything here.</dd>
      <dt>worst sector</dt>
      <dd>The compass octant whose reachable ground carries the highest
      weighted unstable fraction &mdash; which slope to walk first.</dd>
      <dt>assessed over N cells</dt>
      <dd>Settlements are scored over a footprint disc scaled by place type
      (hamlet 100&nbsp;m &hellip; city 1&nbsp;km); the headline is the
      90th-percentile cell &mdash; the exposed edge of town, not its safest
      point and not its single worst cell.</dd>
      <dt>roads: total length</dt>
      <dd>The classified motorable network from OpenStreetMap - motorway,
      trunk, primary, secondary, tertiary. Rural link roads
      ("unclassified"), residential streets, service roads and tracks are
      not counted; config.road_classes widens the definition.</dd>
      <dt>share exposed (roads)</dt>
      <dd>Exposed kilometres as a % of all road kilometres assessed in this
      province.</dd>
      <dt>risk clusters (green &middot; yellow outlines)</dt>
      <dd>Tight outlines around distinct concentrations of at-risk assets:
      exposed settlements (green border) or exposed road segments (yellow
      border) grouped when within 2.5&nbsp;km of one another. Each shape is
      the convex outline of its members, padded 250&nbsp;m, so shapes hug
      their clusters and do not overlap. The sidebar lists the largest by
      member count; singleton clusters are counted but not drawn.</dd>
      <dt>dashed line: cut-slope</dt>
      <dd>Ground immediately above the segment steeper than the configured
      angle (default 35&deg;; the popup shows the measured value). A terrain
      flag marking where the model's road-cut blind spot is &mdash; not a
      model score.</dd>
      <dt>dotted line: washout</dt>
      <dd>The segment touches a drainage cell (specific catchment area above
      the configured threshold, default 5,000&nbsp;m). Crossings there are
      taken by flows arriving <i>along the channel</i>.</dd>
      <dt>Climate scenarios</dt>
      <dd>Only recharge changes: CMIP6 wettest-month precipitation for the
      chosen pathway and window, normalised by the present-day reference.
      Terrain, soils and thresholds are held fixed.</dd>
      <dt>&Delta; / change</dt>
      <dd>Difference from the present-day scenario, same units as the figure
      it follows.</dd>
      </dl></details>

      <div class="note" id="caveat"></div>
    </div>
  </aside>
</div>
__DATA__
<script>
const BOUNDS  = __BOUNDS__;
const BASEMAPS = __BASEMAPS__;   // [{name, url, options}], first is default
const LAYERS  = __LAYERS__;
const SUMMARY = __SUMMARY__;
const META    = __META__;
const BANDS   = __BANDS__;
const EDGES   = __EDGES__;   // [upper bound, label], ascending
const RAMP    = __RAMP__;

// Climate scenarios the run was scored under. The first is always the present
// day, and every change figure on the page is measured against it.
const SCENARIOS = (LAYERS.scenarios && LAYERS.scenarios.length)
  ? LAYERS.scenarios
  : [{key: 'current', label: 'present day', raster: LAYERS.raster}];
const BASE = SUMMARY.baseline || SCENARIOS[0].key;
// The page opens on the present day, explicitly: futures are something the
// reader selects, never something the map asserts by default.
const START = SCENARIOS.find(s => !s.ssp) || SCENARIOS[0];
const STATE = {key: START.key, bands: true};

// Neutral styling for when band colouring is switched off: the assets stay
// on the map as geography - where the settlements and roads are - without
// asserting anything about their exposure. Scores stay in the popups.
const NEUTRAL = {point: '#5b6470', road: '#5b6470'};

// The tables are worth reading even when the map library did not load - a
// blocked network should cost the basemap, not the whole page.
const HAVE_MAP = (typeof L !== 'undefined');
let map = null, bnds = null;
const baseLayers = {};
const overlays = {};
let rasterOverlay = null;

if (HAVE_MAP) {
  map = L.map('map', {preferCanvas: true});
  BASEMAPS.forEach((b, i) => {
    const t = L.tileLayer(b.url, b.options);
    baseLayers[b.name] = t;
    if (i === 0) t.addTo(map);
  });

  bnds = [[BOUNDS.south, BOUNDS.west], [BOUNDS.north, BOUNDS.east]];
  map.fitBounds(bnds);

  if (START.raster) {
    rasterOverlay = L.imageOverlay(START.raster, bnds, {opacity: 0.75});
    overlays['Susceptibility'] = rasterOverlay.addTo(map);
  }
} else {
  document.getElementById('map').innerHTML =
    '<div style="padding:24px;max-width:44ch;color:var(--muted)">' +
    '<b style="color:var(--fg)">The map could not be drawn.</b><br>' +
    'Leaflet did not load. The figures in the panel come from the run itself ' +
    'and are unaffected.</div>';
}

// The band is derived, not stored: repeating "very high" five times per road
// segment across 18,000 segments costs a megabyte to say what the score says.
function bandOf(score) {
  for (const [edge, label] of EDGES) if (score < edge) return label;
  return EDGES[EDGES.length - 1][1];
}
function bandColour(b) { return BANDS[b] || '#888'; }
function bandFor(rec) { return rec.band || bandOf(rec.score); }
function row(k, v) { return v === null || v === undefined || v === ''
  ? '' : `<div><span class="kv">${k}:</span> ${v}</div>`; }
function num(v, d) { return (v ?? 0).toLocaleString(undefined,
  {minimumFractionDigits: d ?? 0, maximumFractionDigits: d ?? 0}); }
function signed(v, d) {
  const t = (v > 0 ? '+' : '') + num(v, d);
  return v === 0 ? `<span class="kv">0</span>`
    : `<span class="${v > 0 ? 'up' : 'down'}">${t}</span>`;
}

// A feature carries one record per scenario. Reading through this everywhere
// means the selector only has to change STATE.key and ask for a restyle.
function at(p, key) {
  return (p.scenarios && p.scenarios[key]) || p;
}
function cur(p) { return at(p, STATE.key); }

/* Layers arrive as <script src> assignments, not fetches - see write_data. */
function addData(key, name, fn, show) {
  const gj = (window.HSIM_DATA || {})[key];
  if (!gj || !HAVE_MAP) return null;
  try {
    const layer = fn(gj);
    overlays[name] = layer;
    if (show) layer.addTo(map);
    return layer;
  } catch (e) { console.warn('layer failed', name, e); return null; }
}

/* Per-scenario block in a popup, so the future is visible without switching. */
function scenarioRows(p) {
  if (!p.scenarios || SCENARIOS.length < 2) return '';
  const b = at(p, BASE).score;
  return `<table style="margin-top:6px">
    <tr><th>climate</th><th class="num">score</th><th class="num">vs now</th></tr>` +
    SCENARIOS.map(sc => {
      const r = p.scenarios[sc.key];
      if (!r) return '';
      const d = r.score - b;
      return `<tr${sc.key === STATE.key ? ' style="font-weight:600"' : ''}>
        <td><span class="chip" style="background:${bandColour(bandFor(r))}"></span>${sc.short || sc.key}</td>
        <td class="num">${r.score.toFixed(3)}</td>
        <td class="num">${sc.key === BASE ? '<span class="kv">—</span>'
                                          : signed(d, 3)}</td></tr>`;
    }).join('') + '</table>';
}

function assetPopup(p, title, extra) {
  const c = cur(p);
  // Reach geometry is a property of the terrain, not of the climate, so it is
  // stored once against the present day rather than repeated per scenario.
  const g = at(p, BASE);
  return `<b>${title}</b><br>${extra}
    <hr style="border:none;border-top:1px solid #ccc;margin:6px 0">
    ${row('exposure', `<b>${c.score}</b> (${bandFor(c)})`)}
    ${row('reaching', c.reaching)}
    ${row('on site', c.on_site)}
    ${row('unstable supply', c.delivering_m2
         ? `${num(c.delivering_m2 / 10000, 1)} ha positioned to reach` : null)}
    ${g.sector ? sectorFigure(g.sector, g.sector_reaching) : ''}
    ${row('worst source', g.n_sources
         ? `${g.source_relief_m} m above, ${g.source_distance_m} m away`
         : 'none')}
    ${p.footprint_m
      ? row('assessed over', `${p.n_cells} cells, ${p.footprint_m} m footprint (p90)`)
      : ''}
    ${scenarioRows(p)}`;
}

function markerRadius(p) {
  return p.population > 20000 ? 8 : p.population > 2000 ? 6 : 4.5;
}

let settlements = null, roads = null;

function fillFor(p) {
  return STATE.bands ? bandColour(bandFor(cur(p))) : NEUTRAL.point;
}

function settlementLayer(gj) {
  return L.geoJSON(gj, {
    pointToLayer: (f, latlng) => L.circleMarker(latlng, {
      radius: markerRadius(f.properties),
      fillColor: fillFor(f.properties), color: '#00000055',
      weight: 1, fillOpacity: 0.92}),
    onEachFeature: (f, l) => {
      const p = f.properties;
      l.bindPopup(() => assetPopup(p, p.name || '(unnamed)',
        row('type', p.place) +
        row('population', p.population ? p.population.toLocaleString() : null)));
    }});
}

// Exposure is the colour; the failure mechanism is the stroke pattern.
// Solid = burial from above only; dashed = cut-slope; dotted = washout;
// dash-dot = both. One vocabulary everywhere: the legend, the popups and
// the glossary all use these exact words.
const MECH = {cut: 'cut-slope above the road',
              wash: 'washout, crosses a channel'};
const MECH_MIN_ZOOM = 11;

function assessed(p) {
  const c = cur(p);
  return !!(c && c.score !== null && c.score !== undefined);
}

function mechDash(p) {
  // Zoomed out only the exposure colour reads; the pattern appears once
  // there is room for it. Unassessed segments never claim a mechanism.
  if (!assessed(p) || map.getZoom() < MECH_MIN_ZOOM) return null;
  if (p.cut_slope && p.washout) return '12 6 1 6';
  if (p.cut_slope) return '10 6';
  if (p.washout) return '1 8';
  return null;
}

function roadStyle(f) {
  const p = f.properties;
  const dash = mechDash(p);
  const base = {dashArray: dash,
                lineCap: (p.washout && !p.cut_slope) ? 'round' : 'butt'};
  if (!STATE.bands)
    return {...base, color: NEUTRAL.road, weight: 2.5, opacity: 0.75};
  const c = cur(p);
  return {...base, color: bandColour(bandFor(c)),
          weight: c.score >= (SUMMARY.exposed_threshold ?? 0.08) ? 4 : 2.5,
          opacity: 0.9};
}

// The direction risk arrives from, drawn rather than said: a compass ring
// with an arrow entering from the worst sector and pointing at the
// settlement in the centre. North is up.
const SECTOR_AZ = {N: 0, NE: 45, E: 90, SE: 135, S: 180, SW: 225,
                   W: 270, NW: 315};

function sectorFigure(sector, w) {
  const az = SECTOR_AZ[sector] ?? 0;
  return `<div style="display:flex;align-items:center;gap:10px;margin:6px 0">
    <svg width="46" height="46" viewBox="0 0 46 46">
      <circle cx="23" cy="23" r="20" fill="none" stroke="#bbb"/>
      <text x="23" y="9" text-anchor="middle" font-size="8"
            fill="#888">N</text>
      <circle cx="23" cy="23" r="3" fill="#555"/>
      <g transform="rotate(${az} 23 23)">
        <line x1="23" y1="6" x2="23" y2="14" stroke="#c0392b"
              stroke-width="2.5"/>
        <path d="M23 19 L18.5 11.5 L27.5 11.5 Z" fill="#c0392b"/>
      </g>
    </svg>
    <span class="kv">risk arrives from the <b>${sector}</b><br>
    (weighted ${(w ?? 0).toFixed(3)})</span></div>`;
}

function mechRows(p) {
  if (!assessed(p) || (!p.cut_slope && !p.washout)) return '';
  return row('mechanisms',
    [p.cut_slope ? `${MECH.cut} (${p.cut_slope_deg}&deg;)` : null,
     p.washout ? `${MECH.wash} (SCA ${num(p.washout_sca_m)} m)` : null]
    .filter(Boolean).join('<br>'));
}

function roadPopup(p) {
  return assetPopup(p, p.name || '(unnamed road)',
    row('class', p.highway) +
    row('segment', `${p.segment} &middot; ${p.length_m} m`) +
    mechRows(p));
}

function roadLayer(gj) {
  return L.geoJSON(gj, {
    style: roadStyle,
    onEachFeature: (f, l) => l.bindPopup(() => roadPopup(f.properties))});
}

// A dark casing beneath flagged segments - the border that makes a
// mechanism-carrying road stand out from a plain one at any zoom. Its own
// overlay so the emphasis can be switched off; the stroke pattern on the
// road line itself always tells the mechanism.
let mechCount = 0;
let casing = null;

function casingOpacity() {
  return map.getZoom() >= MECH_MIN_ZOOM ? 0.5 : 0;
}

function casingLayer(gj) {
  mechCount = 0;
  return L.geoJSON(gj, {
    interactive: false,
    filter: f => {
      const hit = assessed(f.properties)
        && !!(f.properties.cut_slope || f.properties.washout);
      if (hit) mechCount += 1;
      return hit;
    },
    style: () => ({color: '#111', weight: 6.5, opacity: casingOpacity()})});
}

function combinedRoadLayer(gj) {
  // Risk and failure mechanism are one thing on one road: a single overlay
  // draws the emphasis casing beneath the exposure-coloured, pattern-coded
  // line. Zoomed out the casing and pattern fade, leaving colour alone.
  casing = casingLayer(gj);
  roads = roadLayer(gj);
  return L.layerGroup([casing, roads]);
}

function mechSample(dash, cap) {
  return '<svg class="glyph" width="58" height="12">'
    + '<line x1="3" y1="6" x2="55" y2="6" stroke="#111" stroke-width="9"'
    + ' opacity="0.5"/>'
    + '<line x1="3" y1="6" x2="55" y2="6" stroke="#e8590c" stroke-width="4"'
    + (cap ? ' stroke-linecap="round"' : '')
    + (dash ? ' stroke-dasharray="' + dash + '"' : '') + '/></svg>';
}

function inventoryLayer(colour, r) {
  return gj => L.geoJSON(gj, {
    pointToLayer: (f, latlng) => L.circleMarker(latlng, {
      radius: r, fillColor: colour, color: colour,
      weight: 0, fillOpacity: 0.55})});
}

function restyle() {
  const sc = SCENARIOS.find(s => s.key === STATE.key) || SCENARIOS[0];
  if (rasterOverlay && sc.raster) rasterOverlay.setUrl(sc.raster);
  if (settlements) settlements.eachLayer(l => l.setStyle(
    {fillColor: fillFor(l.feature.properties)}));
  if (roads) roads.setStyle(roadStyle);
  // The legend describes colours that are no longer on the map when the
  // toggle is off, so it dims rather than lies.
  document.getElementById('bands').style.opacity = STATE.bands ? '' : '0.35';
  drawStats();

// Distinct clusters of at-risk assets, counted at page build: exposed
// settlements (and road segments) linked when within the clustering
// distance of one another. A ranked list of the largest, so "where are
// the concentrations" has an answer without panning.
if (SUMMARY.clusters) {
  const c = SUMMARY.clusters;
  const li = (t) => (t || []).map(x =>
    `<tr><td>${x.name}</td><td class="num">${x.n}</td></tr>`).join('');
  const block = (kind, d) => !d || !d.n ? '' :
    `<h2>${kind} risk clusters: ${d.n}</h2>` +
    (d.top && d.top.length
      ? `<table>${li(d.top)}</table>` : '');
  document.getElementById('clusters').innerHTML =
    block('Settlement', c.settlements) + block('Road', c.roads);
}
}

if (HAVE_MAP) {
  // The administrative boundary the assets were clipped to. Drawn first so
  // everything scored sits on top of it; non-interactive so it never steals
  // a click from a settlement on the border.
  addData('boundary', 'Unit boundary', gj => L.geoJSON(gj, {
    interactive: false,
    style: {color: '#111', weight: 3, opacity: 0.95,
            fill: false, dashArray: '8 5'}}),
    true);
  addData('roads', 'Roads: exposure & failure mechanism',
          combinedRoadLayer, true);

  // Risk clusters as transparent shapes: an enclosing circle per cluster of
  // two or more exposed assets - green borders for settlement clusters,
  // yellow for road clusters - each in the layer picker.
  const CLUSTER_STYLE = {
    settlements: {colour: '#2e7d32', label: 'Settlement risk clusters',
                  what: 'exposed settlements'},
    roads: {colour: '#d4a017', label: 'Road risk clusters',
            what: 'exposed road segments'}};
  const CLUST = (window.HSIM_DATA || {})['clusters'] || {};
  Object.entries(CLUSTER_STYLE).forEach(([kind, cs]) => {
    const list = CLUST[kind] || [];
    if (!list.length) return;
    overlays[cs.label] = L.layerGroup(list.map(c =>
      L.polygon((c.polygon || []).map(([lon, lat]) => [lat, lon]), {
        color: cs.colour, weight: 2,
        fillColor: cs.colour, fillOpacity: 0.07})
      .bindPopup(`<b>${c.name}</b><br>${c.n} ${cs.what} in this cluster`)
    )).addTo(map);
  });
  map.on('zoomend', () => {
    if (roads) roads.setStyle(roadStyle);
    if (casing) casing.setStyle({opacity: casingOpacity()});
  });
  settlements = addData('settlements', 'Settlements by exposure',
                        settlementLayer, true);
  addData('inventory', 'Training landslides',
          inventoryLayer('#111111', 2.2), false);
  addData('background', 'Training background',
          inventoryLayer('#2c7bb6', 1.8), false);
  // Top-left, because the floating info panel owns the top-right corner -
  // this is what made the layer picker vanish when the panel began to
  // float. Expanded, so the layers are a visible menu, not an icon hunt.
  L.control.layers(baseLayers, overlays,
                   {collapsed: false, position: 'topleft'}).addTo(map);
  restyle();
}

// ---- side panel ----------------------------------------------------------
document.getElementById('ramp').style.background =
  'linear-gradient(90deg,' + RAMP.map(s => `${s[1]} ${s[0] * 100}%`).join(',') + ')';

document.getElementById('bands').innerHTML = Object.entries(BANDS)
  .map(([k, v]) => `<div><span class="chip" style="background:${v}"></span>${k}</div>`)
  .join('');

document.querySelectorAll('.glothr').forEach(el => {
  el.textContent = String(SUMMARY.exposed_threshold ?? 0.08);
});

document.getElementById('bandtoggle').addEventListener('change', ev => {
  STATE.bands = ev.target.checked;
  restyle();
});

if (mechCount) {
  document.getElementById('mechlegend').innerHTML =
    '<h2>Road failure mechanisms</h2>'
    + '<div>' + mechSample(null) + 'solid: burial from above only</div>'
    + '<div>' + mechSample('12 7') + 'dashed: ' + MECH.cut + '</div>'
    + '<div>' + mechSample('1 9', true) + 'dotted: ' + MECH.wash + '</div>'
    + '<div>' + mechSample('14 7 2 7') + 'dash-dot: both</div>'
    + '<p class="sub" style="margin-top:4px">Colour is exposure; the stroke '
    + 'pattern is the mechanism; the dark border marks flagged segments '
    + '(visible when zoomed in). Shown only where exposure was assessed.</p>';
}

document.getElementById('meta').textContent = [
  META.area, META.resolution].filter(Boolean).join(' · ');

if (SCENARIOS.length > 1) {
  document.getElementById('picker').innerHTML =
    `<h2>Climate</h2><select id="scen">` + SCENARIOS.map(
      s => `<option value="${s.key}"${s.key === STATE.key ? ' selected' : ''}>`
           + `${s.label}</option>`).join('') +
    `</select><p class="sub" style="margin-top:6px" id="scenlabel"></p>`;
  document.getElementById('scen').addEventListener('change', ev => {
    STATE.key = ev.target.value;
    restyle();
  });
}

function table(title, rows) {
  if (!rows.length) return '';
  return `<h2>${title}</h2><table>${rows.map(
    r => `<tr><td>${r[0]}</td><td class="num">${r[1]}</td></tr>`).join('')}</table>`;
}

function statsFor(key) {
  return (SUMMARY.scenarios && SUMMARY.scenarios[key]) || SUMMARY;
}

function drawStats() {
  const s = statsFor(STATE.key);
  const b = statsFor(BASE);
  const same = STATE.key === BASE;
  const d = (k) => same ? '' : ` <span class="pill">${
    signed(s[k] - b[k], k === 'road_km_exposed' ? 1 : 0)}</span>`;
  document.getElementById('stats').innerHTML = [
    table('Settlements', [
      ['assessed', num(s.n_settlements)],
      ['exposed', num(s.n_settlements_exposed) + d('n_settlements_exposed')],
      ['people in those', num(s.population_exposed)],
      ['mean score', (s.mean_settlement_score ?? 0).toFixed(3)],
    ]),
    table('Roads', [
      ['segments', num(s.n_road_segments)],
      ['total length', `${num(s.road_km_total, 1)} km`],
      ['exposed', `${num(s.road_km_exposed, 1)} km` + d('road_km_exposed')],
      ['share exposed', `${s.road_pct_exposed ?? 0}%`],
      // Mechanism flags are terrain geometry, so unlike everything above
      // they do not move with the scenario selector.
      ...(SUMMARY.mechanisms ? [
        ['cut-slope flagged', `${num(SUMMARY.mechanisms.road_km_cut_slope, 1)} km`],
        ['channel crossings', `${num(SUMMARY.mechanisms.road_km_washout, 1)} km`],
      ] : []),
    ]),
  ].join('');

  const lab = document.getElementById('scenlabel');
  const sc = SCENARIOS.find(x => x.key === STATE.key);
  if (lab && sc) lab.textContent = sc.detail || '';

  drawWorst();
}

/* Every scenario side by side, which is the comparison the map cannot show. */
if (SCENARIOS.length > 1 && SUMMARY.scenarios) {
  document.getElementById('compare').innerHTML =
    `<h2>Across scenarios</h2><div class="wrap"><table>
      <tr><th>climate</th><th class="num">settlements</th>
          <th class="num">road km</th><th class="num">mean</th></tr>` +
    SCENARIOS.map(sc => {
      const st = SUMMARY.scenarios[sc.key];
      if (!st) return '';
      return `<tr><td>${sc.short || sc.key}</td>
        <td class="num">${num(st.n_settlements_exposed)}</td>
        <td class="num">${num(st.road_km_exposed, 1)}</td>
        <td class="num">${(st.mean_settlement_score ?? 0).toFixed(3)}</td></tr>`;
    }).join('') + `</table></div>
    <p class="sub" style="margin-top:6px">Counts and length at or above a
    score of ${SUMMARY.exposed_threshold ?? 0.08}.</p>`;
}

function drawWorst() {
  const el = document.getElementById('worst');
  const rows = LAYERS.worst || [];
  if (!rows.length) { el.innerHTML = ''; return; }
  const key = STATE.key;
  const ranked = rows.slice().map(w => {
    const r = (w.scenarios && w.scenarios[key]) || w;
    const b = (w.scenarios && w.scenarios[BASE]) || w;
    return {name: w.name, score: r.score, band: bandFor(r), delta: r.score - b.score};
  }).sort((a, b) => b.score - a.score);
  el.innerHTML = `<h2>Most exposed settlements</h2><div class="scroll"><table>
      <tr><th>Place</th><th class="num">Score</th><th class="num">vs now</th></tr>` +
    ranked.map(w => `<tr><td><span class="chip" style="background:${
      bandColour(w.band)}"></span>${w.name}</td>
      <td class="num">${w.score.toFixed(3)}</td>
      <td class="num">${key === BASE ? '<span class="kv">—</span>'
                                     : signed(w.delta, 3)}</td></tr>`).join('') +
    '</table></div>';
}

drawStats();


</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# the regional index
# ---------------------------------------------------------------------------

def build_region_index(out_dir: str, title: str, rows: Sequence[dict],
                       meta: Optional[dict] = None) -> str:
    """One page that is the whole regional product: pick a province, see it.

    A sweep leaves one folder per province, which is an archive rather than a
    result. This writes the single page a user opens instead: a country
    dropdown, a province dropdown under it, and the selected province's map
    embedded below - plus the ranked, sortable table over every province,
    whose rows select into the same view. The per-province pages stay on disk
    as the frames this page loads, so nothing is duplicated and a province is
    still shareable on its own or via the page's #slug hash.
    """
    os.makedirs(out_dir, exist_ok=True)
    html = _INDEX.replace("__TITLE__", title)
    html = html.replace("__ROWS__", json.dumps(list(rows)))
    html = html.replace("__META__", json.dumps(meta or {}))
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


_INDEX = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#fff; --fg:#16191d; --muted:#5b6470; --line:#dfe3e8;
          --panel:#f7f8fa; --accent:#a50026; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14171a; --fg:#e8eaed; --muted:#9aa4b2; --line:#2b3038;
            --panel:#1c2025; --accent:#ff6b57; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:28px 24px 60px; background:var(--bg); color:var(--fg);
         font:14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
              Helvetica, Arial, sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; }
  h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }
  p.sub { color:var(--muted); margin:0 0 18px; }
  header.app { border-bottom:2px solid var(--accent); padding-bottom:14px;
               margin-bottom:20px; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.07em;
       color:var(--muted); margin:26px 0 8px; font-weight:600; }
  .cards { display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:6px;
          padding:12px 14px; }
  .card .n { font-size:22px; font-variant-numeric:tabular-nums; }
  .card .k { color:var(--muted); font-size:12px; }
  .tablewrap { overflow-x:auto; border:1px solid var(--line); border-radius:6px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line);
           white-space:nowrap; }
  th { background:var(--panel); color:var(--muted); font-size:11px; font-weight:600;
       text-transform:uppercase; letter-spacing:.05em; cursor:pointer;
       position:sticky; top:0; }
  th:hover { color:var(--fg); }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  tr:last-child td { border-bottom:none; }
  a { color:inherit; }
  .bar { height:7px; border-radius:4px; background:var(--accent); display:inline-block;
         vertical-align:middle; margin-right:7px; min-width:2px; }
  .note { font-size:12px; color:var(--muted); border-left:3px solid var(--line);
          padding:9px 12px; margin-top:20px; background:var(--panel); }
  input { padding:7px 10px; font:inherit; font-size:13px; width:260px;
          color:var(--fg); background:var(--bg); border:1px solid var(--line);
          border-radius:4px; margin-bottom:10px; }
  .toolbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
             background:var(--panel); border:1px solid var(--line);
             border-bottom:none; border-radius:8px 8px 0 0; padding:10px 12px; }
  .toolbar label { font-size:11px; text-transform:uppercase; color:var(--muted);
                   letter-spacing:.05em; margin-right:-4px; }
  .toolbar select { padding:8px 10px; font:inherit; font-size:13px;
                    color:var(--fg); background:var(--bg);
                    border:1px solid var(--line); border-radius:6px;
                    min-width:200px; }
  .toolbar a { font-size:12px; color:var(--muted); margin-left:auto; }
  .framewrap { border:1px solid var(--line); border-radius:0 0 8px 8px;
               overflow:hidden; height:70vh; min-height:420px;
               background:var(--panel); position:relative; }
  .framewrap iframe { width:100%; height:100%; border:0; display:block; }
  .fallback { position:absolute; inset:0; display:none; align-items:center;
              justify-content:center; text-align:center; padding:30px;
              color:var(--muted); background:var(--panel); font-size:13px; }
  .fallback a { color:var(--accent); }
  .caps { font-size:12px; color:var(--muted); margin:10px 14px 0; }
  .fullbleed { width:100vw; position:relative; left:50%; margin-left:-50vw; }
  .fullbleed .toolbar { border-radius:0; border-left:none; border-right:none;
                        padding-left:16px; padding-right:16px; }
  .fullbleed .framewrap { border-radius:0; border-left:none;
                          border-right:none; height:78vh; }
  .caps b { color:var(--fg); font-weight:600; }
  tr:hover td { background:var(--panel); }
  .tabs { display:flex; gap:6px; margin:0 0 18px; border-bottom:1px solid var(--line); }
  .tab { font:inherit; font-size:13px; padding:9px 16px; cursor:pointer;
         background:none; border:none; color:var(--muted);
         border-bottom:2px solid transparent; margin-bottom:-1px; }
  .tab.active { color:var(--fg); border-bottom-color:var(--accent);
                font-weight:600; }
  .info { display:inline-block; cursor:pointer; color:var(--muted);
          border:1px solid var(--line); border-radius:50%; width:15px;
          height:15px; line-height:14px; text-align:center; font-size:10px;
          font-style:normal; margin-left:6px; vertical-align:1px;
          user-select:none; }
  .info:hover { color:var(--fg); border-color:var(--muted); }
  .card[data-info] { cursor:pointer; }
  .card[data-info]:hover { border-color:var(--muted); }
  .modalwrap { position:fixed; inset:0; background:rgba(0,0,0,.45);
               display:none; align-items:center; justify-content:center;
               z-index:50; padding:20px; }
  .modalwrap.open { display:flex; }
  .modal { background:var(--bg); color:var(--fg); border:1px solid var(--line);
           border-radius:10px; max-width:520px; width:100%; padding:18px 20px;
           box-shadow:0 12px 40px rgba(0,0,0,.35); font-size:13px;
           line-height:1.55; }
  .modal h3 { margin:0 0 8px; font-size:15px; }
  .modal .close { float:right; cursor:pointer; border:none; background:none;
                  color:var(--muted); font-size:16px; padding:0 2px; }
  .modal .close:hover { color:var(--fg); }
  .rowlink { cursor:pointer; color:var(--accent); text-decoration:underline; }
  details.gloss { margin-top:20px; font-size:12px; }
  details.gloss summary { cursor:pointer; font-size:12px;
      text-transform:uppercase; letter-spacing:.07em; color:var(--muted);
      font-weight:600; }
  details.gloss dl, div.gloss dl { margin:10px 0 0; max-width:820px; }
  details.gloss dt, div.gloss dt { font-weight:600; margin-top:8px; }
  details.gloss dd, div.gloss dd { margin:1px 0 0 0; color:var(--muted);
                                   font-size:12px; }

</style>
</head>
<body>
<div class="wrap">
  <header class="app">
    <h1>__TITLE__</h1>
  </header>

  <div class="tabs" role="tablist">
    <button class="tab active" data-tab="map">Map</button>
    <button class="tab" data-tab="table">Provinces table</button>
    <button class="tab" data-tab="gloss">Glossary</button>
  </div>

  <section id="tab-map">
  <div id="viewer" class="fullbleed" style="display:none">
    <div class="toolbar">
      <label for="pickCountry">Country</label>
      <select id="pickCountry"></select>
      <label for="pickUnit">Province</label>
      <select id="pickUnit"></select>
      <a id="pickExt" target="_blank" rel="noopener">open in its own tab &#8599;</a>
      <span class="info" data-info="viewer">i</span>
    </div>
    <div class="framewrap">
      <iframe id="frame" title="province map"></iframe>
      <div class="fallback" id="fallback">
        <div>The embedded view did not load &mdash; some browsers block
        local frames.<br>
        <a id="fallbackLink" target="_blank" rel="noopener">Open this
        province's map directly &#8599;</a></div>
      </div>
    </div>
    <p class="caps">Inside each province view: <b>climate scenarios</b>
    (present day and both near-term windows per pathway) &middot;
    <b>exposure bands</b> with an on/off colour toggle &middot;
    <b>failure mechanisms</b> as road stroke patterns (dashed cut-slope,
    dotted washout) &middot; <b>worst settlements</b> list &middot;
    the unit <b>boundary</b> &middot; three basemaps. Every layer and table
    follows the scenario selector.</p>
  </div>

  <h2>Region at a glance <span class="info" data-info="glance">i</span></h2>
  <div class="cards" id="cards"></div>
  </section>

  <section id="tab-table" hidden>
  <h2>Provinces, most unstable first</h2>
  <input id="q" placeholder="filter by province or country">
  <div class="tablewrap"><table id="t">
    <thead><tr>
      <th data-k="country">Country</th>
      <th data-k="name">Province</th>
      <th class="num" data-k="unstable_pct">Unstable %<span class="info"
          data-info="unstable_pct">i</span></th>
      <th class="num" data-k="mean_probability">Mean P<span class="info"
          data-info="mean_p">i</span></th>
      <th class="num" data-k="p90_probability">P90<span class="info"
          data-info="p90">i</span></th>
      <th class="num" data-k="settlements_exposed">Settlements exposed<span
          class="info" data-info="sett_exposed">i</span></th>
      <th class="num" data-k="road_km_exposed">Road km exposed<span
          class="info" data-info="road_exposed">i</span></th>
      <th class="num" data-k="sett_clusters">Settlement clusters<span
          class="info" data-info="clusters">i</span></th>
      <th class="num" data-k="road_clusters">Road clusters<span
          class="info" data-info="clusters">i</span></th>
      <th>Map<span class="info" data-info="map_col">i</span></th>
    </tr></thead>
    <tbody></tbody>
  </table></div>

  </section>

  <section id="tab-gloss" hidden>
  <h2>Glossary &mdash; every term, and every % of what</h2>
  <div class="gloss"><dl>
  <dt>Unstable %</dt>
  <dd>% of the grid cells <i>inside that province's boundary</i> whose
  failure probability is &ge; 0.5. The probability itself is the share of
  Monte Carlo soil-parameter draws that push a cell's factor of safety below
  1 &mdash; a relative ranking, never an annual chance.</dd>
  <dt>Mean P &middot; P90</dt>
  <dd>Mean and 90th-percentile per-cell failure probability over the same
  cells, 0&ndash;1.</dd>
  <dt>Settlements exposed</dt>
  <dd>Settlements whose exposure score is at or above 0.08 under the present
  day. The score is the greater of the probability where the settlement
  stands and the unstable fraction of upslope ground positioned to reach it
  (18&deg; travel line, 2&nbsp;km, distance-weighted).</dd>
  <dt>Road km exposed</dt>
  <dd>Kilometres of assessed road segments at or above the same threshold,
  each 500&nbsp;m segment scored at its most exposed point.</dd>
  <dt>Most unstable (card)</dt>
  <dd>The largest Unstable % among the mapped provinces.</dd>
  <dt>Map</dt>
  <dd>Opens that province in the viewer on the Map tab; every per-province
  figure, mechanism pattern and climate scenario lives there, with its own
  fuller glossary in the sidebar.</dd>
  <dt>Risk clusters (settlements &middot; roads)</dt>
  <dd>Distinct concentrations of at-risk assets: exposed settlements (or
  exposed road segments) grouped when within 2.5&nbsp;km of one another,
  counted per province. Each province page lists its largest clusters by
  member count.</dd>
  </dl></div>

  <div class="note" id="caveat"></div>
  </section>

  <div class="modalwrap" id="modalwrap">
    <div class="modal" role="dialog" aria-modal="true">
      <button class="close" id="modalclose" aria-label="close">&#10005;</button>
      <h3 id="modaltitle"></h3>
      <div id="modalbody"></div>
    </div>
  </div>
</div>
<script>
const ROWS = __ROWS__;
const META = __META__;
let sortKey = 'unstable_pct', desc = true;

const num = (v, d) => (v === null || v === undefined)
  ? '<span style="color:var(--muted)">—</span>'
  : v.toLocaleString(undefined, {minimumFractionDigits: d ?? 0,
                                 maximumFractionDigits: d ?? 0});


const worst = Math.max(...ROWS.map(r => r.unstable_pct || 0), 0);
// Bars need a non-zero denominator; the headline figure must not borrow it.
const barScale = worst || 1;
const sum = (k) => ROWS.reduce((a, r) => a + (r[k] || 0), 0);

document.getElementById('cards').innerHTML = [
  ['Provinces mapped', num(ROWS.length), 'provinces'],
  ['Most unstable', ROWS.length ? `${num(worst, 1)}%` : '—', 'most_unstable'],
  ['Settlements exposed', num(sum('settlements_exposed')), 'sett_exposed'],
  ['Road km exposed', num(sum('road_km_exposed')), 'road_exposed'],
].map(([k, v, i]) => `<div class="card" data-info="${i}"><div class="n">${v}</div>
   <div class="k">${k}<span class="info">i</span></div></div>`).join('');
document.querySelectorAll('.card[data-info]').forEach(c =>
  c.addEventListener('click', () => showInfo(c.dataset.info)));

// ---- one app over every province ------------------------------------------
// Each province's map is its own self-contained page (its rasters and asset
// layers are per-province and lazy by construction), so the app is a picker
// over those pages: country, then province, loaded into the frame below.
// The selection lives in the URL hash, so a province is linkable.
function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                  .replace(/"/g, '&quot;');
}
// Click-to-explain: every card, column and control opens a small window
// saying what the thing is - and for every percentage, % of what.
const INFO = {
  glance: ["Region at a glance", "Totals over every province currently in \
the product. They grow as the run completes provinces; a dash in the table \
means that province's stage has not finished yet."],
  provinces: ["Provinces mapped", "How many provinces have results on this \
page. The selection is set in the config (admin_countries); provinces from \
deselected countries stay on disk but out of the product."],
  most_unstable: ["Most unstable", "The largest 'Unstable %' among the \
mapped provinces: the % of grid cells inside that one province's boundary \
whose failure probability is at or above 0.5."],
  sett_exposed: ["Settlements exposed", "Count of settlements with an \
exposure score at or above 0.08 under the present day. The score is the \
greater of the failure probability where the settlement stands and the \
unstable fraction of upslope ground positioned to reach it (18\u00b0 travel \
line, 2 km, distance-weighted). A screening threshold, not a physical \
constant."],
  road_exposed: ["Road km exposed", "Kilometres of 500 m road segments at \
or above the same 0.08 screen, each segment scored at its most exposed \
point. Counted over the classified motorable network (motorway to \
tertiary); rural link roads, residential streets and tracks are not \
included."],
  unstable_pct: ["Unstable %", "% of the grid cells inside that province's \
boundary whose failure probability is at or above 0.5. The probability is \
the share of Monte Carlo soil-parameter draws pushing a cell's factor of \
safety below 1 - a relative ranking, never an annual chance."],
  mean_p: ["Mean P", "Mean per-cell failure probability over the cells \
inside the boundary, 0-1. Relative: compare provinces, do not read as a \
yearly chance."],
  p90: ["P90", "The 90th-percentile per-cell failure probability inside the \
boundary - the level the worst tenth of the province's ground exceeds."],
  clusters: ["Risk clusters", "Distinct concentrations of at-risk assets: \
exposed settlements (or exposed 500 m road segments) grouped when within \
2.5 km of one another, counted per province. Open the province to see its \
largest clusters listed by member count."],
  map_col: ["Map", "'view' opens that province in the Map tab. A dash means \
the province's page is not built yet; 'failed' carries the build error on \
hover."],
  viewer: ["The province viewer", "Pick a country, then a province; the \
full interactive map loads below. Inside it: five climate scenarios \
(present day opens by default), exposure bands with a colour toggle, \
failure mechanisms as road stroke patterns, the worst settlements, the \
clipped boundary, three basemaps, and a glossary in its sidebar. The selection \
lives in this page's URL hash, so a province can be linked directly."],
};

const modalwrap = document.getElementById('modalwrap');
function showInfo(key) {
  const e = INFO[key];
  if (!e) return;
  document.getElementById('modaltitle').textContent = e[0];
  document.getElementById('modalbody').textContent = e[1];
  modalwrap.classList.add('open');
}
document.getElementById('modalclose').onclick =
  () => modalwrap.classList.remove('open');
modalwrap.addEventListener('click', ev => {
  if (ev.target === modalwrap) modalwrap.classList.remove('open');
});
document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape') modalwrap.classList.remove('open');
});
document.querySelectorAll('.info').forEach(el =>
  el.addEventListener('click', ev => {
    ev.stopPropagation();                 // an i in a sortable header
    showInfo(el.dataset.info);            // explains; it does not sort
  }));

// ---- tabs: the map is the product, the table is the ranking ---------------
function showTab(name) {
  document.querySelectorAll('.tab').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name));
  ['map', 'table', 'gloss'].forEach(t => {
    document.getElementById('tab-' + t).hidden = t !== name;
  });
}
document.querySelectorAll('.tab').forEach(b =>
  b.addEventListener('click', () => showTab(b.dataset.tab)));

const MAPPED = ROWS.filter(r => r.map && r.slug);
const elCountry = document.getElementById('pickCountry');
const elUnit = document.getElementById('pickUnit');
const elFrame = document.getElementById('frame');
const elExt = document.getElementById('pickExt');

function unitsIn(c) {
  return MAPPED.filter(r => r.country === c)
    .slice().sort((a, b) => (b.unstable_pct || 0) - (a.unstable_pct || 0));
}

function fillUnits(c) {
  elUnit.innerHTML = unitsIn(c).map(r =>
    `<option value="${r.slug}">${r.name}</option>`).join('');
}

let frameLoaded = false;
elFrame.addEventListener('load', () => {
  frameLoaded = true;
  document.getElementById('fallback').style.display = 'none';
});

function select(slug, push) {
  const row = MAPPED.find(r => r.slug === slug) || MAPPED[0];
  if (!row) return;
  elCountry.value = row.country;
  fillUnits(row.country);
  elUnit.value = row.slug;
  if (elFrame.getAttribute('src') !== row.map) {
    frameLoaded = false;
    elFrame.src = row.map;
    // If nothing has rendered after a beat, offer the direct link instead of
    // an unexplained blank rectangle.
    setTimeout(() => {
      if (!frameLoaded)
        document.getElementById('fallback').style.display = 'flex';
    }, 2500);
  }
  elExt.href = row.map;
  document.getElementById('fallbackLink').href = row.map;
  if (push) history.replaceState(null, '', '#' + row.slug);
}

if (!MAPPED.length) {
  // A table with no maps should say why, not leave the reader wondering
  // where the product went.
  const v = document.getElementById('viewer');
  v.style.display = '';
  v.innerHTML = '<div class="note">No province map pages were found next to '
    + 'this index, so there is nothing to embed yet. Build them with '
    + '<b>step9-webapp</b> against the region config; if this index was '
    + 'generated by an older version of the code, delete the '
    + '<b>*_webmap.json</b> markers in outputs/ first so every province '
    + 'page is rebuilt with the current viewer.</div>';
}
if (MAPPED.length) {
  document.getElementById('viewer').style.display = '';
  elCountry.innerHTML = [...new Set(MAPPED.map(r => r.country))].sort()
    .map(c => `<option>${c}</option>`).join('');
  elCountry.addEventListener('change', () => {
    fillUnits(elCountry.value);
    select(elUnit.value, true);
  });
  elUnit.addEventListener('change', () => select(elUnit.value, true));
  window.addEventListener('hashchange',
    () => select(location.hash.slice(1), false));
  // Start on the linked province, or the most unstable one mapped.
  select(location.hash.slice(1) || MAPPED[0].slug, false);
}

function draw() {
  const q = document.getElementById('q').value.trim().toLowerCase();
  const rows = ROWS
    .filter(r => !q || (r.name + ' ' + r.country).toLowerCase().includes(q))
    .slice()
    .sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      if (typeof x === 'string' || typeof y === 'string')
        return (desc ? -1 : 1) * String(x).localeCompare(String(y));
      // Missing values sink to the bottom whichever way the column is sorted.
      const xv = x ?? -Infinity, yv = y ?? -Infinity;
      return desc ? yv - xv : xv - yv;
    });
  document.querySelector('#t tbody').innerHTML = rows.map(r => `
    <tr>
      <td>${r.country}</td>
      <td>${r.name}</td>
      <td class="num"><span class="bar" style="width:${
        Math.round(46 * (r.unstable_pct || 0) / barScale)}px"></span>${
        num(r.unstable_pct, 1)}</td>
      <td class="num">${num(r.mean_probability, 3)}</td>
      <td class="num">${num(r.p90_probability, 3)}</td>
      <td class="num">${num(r.settlements_exposed)}</td>
      <td class="num">${num(r.road_km_exposed, 0)}</td>
      <td class="num">${num(r.sett_clusters)}</td>
      <td class="num">${num(r.road_clusters)}</td>
      <td>${r.map ? `<span class="rowlink" data-slug="${r.slug}">view</span>`
                  : r.map_error
                  ? `<span style="color:var(--accent)" title="${esc(r.map_error)}">failed &#9432;</span>`
                  : '<span style="color:var(--muted)">—</span>'}</td>
    </tr>`).join('');
}

document.querySelector('#t tbody').addEventListener('click', ev => {
  const slug = ev.target && ev.target.dataset && ev.target.dataset.slug;
  if (!slug) return;
  select(slug, true);
  showTab('map');
  window.scrollTo({top: 0, behavior: 'smooth'});
});

document.querySelectorAll('#t th').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.k;
  if (!k) return;
  if (k === sortKey) desc = !desc; else { sortKey = k; desc = true; }
  draw();
}));
document.getElementById('q').addEventListener('input', draw);
draw();

const skipped = META.skipped_too_large || [], failed = META.failed || [];
document.getElementById('caveat').innerHTML =
  '<b>What these numbers are.</b> Unstable % is the share of a province above ' +
  'failure probability 0.5 — a relative figure, not an annual probability. ' +
  'Settlement and road counts are exposure screening by angle of reach: no ' +
  'runout model, no vulnerability, no damage function. Provinces are ' +
  'comparable only because they share one fitted parameter set, which was ' +
  'fitted on soil-mantled crystalline terrain and is extrapolated wherever no ' +
  'inventory exists.' +
  (skipped.length ? `<br><br><b>Skipped as too large:</b> ${skipped.join(', ')}.
     Run these at a coarser resolution.` : '') +
  (failed.length ? `<br><br><b>Failed:</b> ${failed.join(', ')}.` : '');
</script>
</body>
</html>
"""
