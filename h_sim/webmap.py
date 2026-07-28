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

def points_geojson(rows: Sequence[dict], lon_key: str = "lon",
                   lat_key: str = "lat",
                   props: Optional[Sequence[str]] = None) -> dict:
    feats = []
    for r in rows:
        p = ({k: r.get(k) for k in props} if props
             else {k: v for k, v in r.items() if k not in (lon_key, lat_key)})
        feats.append({"type": "Feature",
                      "properties": p,
                      "geometry": {"type": "Point",
                                   "coordinates": [r[lon_key], r[lat_key]]}})
    return {"type": "FeatureCollection", "features": feats}


def lines_geojson(rows: Sequence[dict]) -> dict:
    feats = []
    for r in rows:
        p = {k: v for k, v in r.items() if k != "coords"}
        feats.append({"type": "Feature",
                      "properties": p,
                      "geometry": {"type": "LineString",
                                   "coordinates": [list(c)
                                                   for c in r["coords"]]}})
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
    """
    path = os.path.join(out_dir, f"{name}.js")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("window.HSIM_DATA=window.HSIM_DATA||{};\n"
                 f"window.HSIM_DATA[{json.dumps(name)}]=")
        json.dump(obj, fh)
        fh.write(";\n")
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


def build(out_dir: str, title: str, bounds: Dict[str, float],
          layers: Dict[str, str], summary: Optional[dict] = None,
          meta: Optional[dict] = None,
          cache_dir: Optional[str] = None,
          data_files: Sequence[str] = ()) -> str:
    """Write index.html next to the assets in ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    local = vendor_leaflet(out_dir, cache_dir)
    src = "leaflet" if local else LEAFLET_BASE
    html = _PAGE.replace("__TITLE__", title)
    html = html.replace("__LEAFLET__", src)
    html = html.replace("__DATA__", "\n".join(
        f'<script src="{f}"></script>' for f in data_files))
    html = html.replace("__BOUNDS__", json.dumps(bounds))
    html = html.replace("__LAYERS__", json.dumps(layers))
    html = html.replace("__SUMMARY__", json.dumps(summary or {}))
    html = html.replace("__META__", json.dumps(meta or {}))
    html = html.replace("__BANDS__", json.dumps(BAND_COLOURS))
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
  #app { display: flex; height: 100%; }
  #map { flex: 1 1 auto; min-width: 0; }
  #side { width: 360px; flex: 0 0 360px; border-left: 1px solid var(--line);
          overflow-y: auto; background: var(--panel); }
  @media (max-width: 900px) {
    #app { flex-direction: column; }
    #side { width: 100%; flex: 0 0 45%; border-left: none;
            border-top: 1px solid var(--line); }
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

      <div id="stats"></div>
      <div id="compare"></div>
      <div id="worst"></div>

      <div class="note" id="caveat"></div>
    </div>
  </aside>
</div>
__DATA__
<script>
const BOUNDS  = __BOUNDS__;
const LAYERS  = __LAYERS__;
const SUMMARY = __SUMMARY__;
const META    = __META__;
const BANDS   = __BANDS__;
const RAMP    = __RAMP__;

// Climate scenarios the run was scored under. The first is always the present
// day, and every change figure on the page is measured against it.
const SCENARIOS = (LAYERS.scenarios && LAYERS.scenarios.length)
  ? LAYERS.scenarios
  : [{key: 'current', label: 'present day', raster: LAYERS.raster}];
const BASE = SUMMARY.baseline || SCENARIOS[0].key;
const STATE = {key: SCENARIOS[0].key};

// The tables are worth reading even when the map library did not load - a
// blocked network should cost the basemap, not the whole page.
const HAVE_MAP = (typeof L !== 'undefined');
let map = null, base = null, terrain = null, bnds = null;
const overlays = {};
let rasterOverlay = null;

if (HAVE_MAP) {
  map = L.map('map', {preferCanvas: true});
  base = L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    {maxZoom: 18, attribution: '&copy; OpenStreetMap contributors'}).addTo(map);
  terrain = L.tileLayer(
    'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    {maxZoom: 17, attribution: '&copy; OpenTopoMap, &copy; OpenStreetMap contributors'});

  bnds = [[BOUNDS.south, BOUNDS.west], [BOUNDS.north, BOUNDS.east]];
  map.fitBounds(bnds);

  if (SCENARIOS[0].raster) {
    rasterOverlay = L.imageOverlay(SCENARIOS[0].raster, bnds, {opacity: 0.75});
    overlays['Susceptibility'] = rasterOverlay.addTo(map);
  }
} else {
  document.getElementById('map').innerHTML =
    '<div style="padding:24px;max-width:44ch;color:var(--muted)">' +
    '<b style="color:var(--fg)">The map could not be drawn.</b><br>' +
    'Leaflet did not load. The figures in the panel come from the run itself ' +
    'and are unaffected.</div>';
}

function bandColour(b) { return BANDS[b] || '#888'; }
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
        <td><span class="chip" style="background:${bandColour(r.band)}"></span>${sc.short || sc.key}</td>
        <td class="num">${r.score.toFixed(3)}</td>
        <td class="num">${sc.key === BASE ? '<span class="kv">—</span>'
                                          : signed(d, 3)}</td></tr>`;
    }).join('') + '</table>';
}

function assetPopup(p, title, extra) {
  const c = cur(p);
  return `<b>${title}</b><br>${extra}
    <hr style="border:none;border-top:1px solid #ccc;margin:6px 0">
    ${row('exposure', `<b>${c.score}</b> (${c.band})`)}
    ${row('reaching', c.reaching)}
    ${row('on site', c.on_site)}
    ${row('worst source', c.n_sources
         ? `${c.source_relief_m} m above, ${c.source_distance_m} m away` : 'none')}
    ${scenarioRows(p)}`;
}

function markerRadius(p) {
  return p.population > 20000 ? 8 : p.population > 2000 ? 6 : 4.5;
}

let settlements = null, roads = null;

function settlementLayer(gj) {
  return L.geoJSON(gj, {
    pointToLayer: (f, latlng) => L.circleMarker(latlng, {
      radius: markerRadius(f.properties),
      fillColor: bandColour(cur(f.properties).band), color: '#00000055',
      weight: 1, fillOpacity: 0.92}),
    onEachFeature: (f, l) => {
      const p = f.properties;
      l.bindPopup(() => assetPopup(p, p.name || '(unnamed)',
        row('type', p.place) +
        row('population', p.population ? p.population.toLocaleString() : null)));
    }});
}

function roadStyle(f) {
  const c = cur(f.properties);
  return {color: bandColour(c.band),
          weight: c.score >= (SUMMARY.exposed_threshold ?? 0.08) ? 4 : 2.5,
          opacity: 0.9};
}

function roadLayer(gj) {
  return L.geoJSON(gj, {
    style: roadStyle,
    onEachFeature: (f, l) => {
      const p = f.properties;
      l.bindPopup(() => assetPopup(p, p.name || '(unnamed road)',
        row('class', p.highway) +
        row('segment', `${p.segment} &middot; ${p.length_m} m`)));
    }});
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
    {fillColor: bandColour(cur(l.feature.properties).band)}));
  if (roads) roads.setStyle(roadStyle);
  drawStats();
}

if (HAVE_MAP) {
  roads = addData('roads', 'Roads by exposure', roadLayer, true);
  settlements = addData('settlements', 'Settlements by exposure',
                        settlementLayer, true);
  addData('inventory', 'Training landslides',
          inventoryLayer('#111111', 2.2), false);
  addData('background', 'Training background',
          inventoryLayer('#2c7bb6', 1.8), false);
  L.control.layers({'OpenStreetMap': base, 'Terrain': terrain}, overlays,
                   {collapsed: false}).addTo(map);
  restyle();
}

// ---- side panel ----------------------------------------------------------
document.getElementById('ramp').style.background =
  'linear-gradient(90deg,' + RAMP.map(s => `${s[1]} ${s[0] * 100}%`).join(',') + ')';

document.getElementById('bands').innerHTML = Object.entries(BANDS)
  .map(([k, v]) => `<div><span class="chip" style="background:${v}"></span>${k}</div>`)
  .join('');

document.getElementById('meta').textContent = [
  META.area, META.resolution].filter(Boolean).join(' · ');

if (SCENARIOS.length > 1) {
  document.getElementById('picker').innerHTML =
    `<h2>Climate</h2><select id="scen">` + SCENARIOS.map(
      s => `<option value="${s.key}">${s.label}</option>`).join('') +
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
    return {name: w.name, score: r.score, band: r.band, delta: r.score - b.score};
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

document.getElementById('caveat').innerHTML =
  '<b>Read this as screening, not risk.</b> Assets are scored by the ' +
  'proximity-weighted fraction of upslope ground that could reach them under ' +
  'an angle-of-reach criterion and that the model calls unstable — not by a ' +
  'runout model, and with no vulnerability or damage function. Susceptibility ' +
  'itself is relative: differences between places are meaningful, the value is ' +
  'not an annual probability of failure. Future scenarios change one thing, ' +
  'the recharge field; the terrain, the soil parameters and the meaning of a ' +
  'return period are held at their present-day values.';
</script>
</body>
</html>
"""
