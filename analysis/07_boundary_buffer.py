"""How wide a buffer does a province-by-province sweep actually need?

A regional run is cut into states and provinces because that is the unit maps
get requested in. But a provincial border crosses catchments without regard for
drainage, and the wetness term depends on specific catchment area - the upslope
area draining through a cell. Route flow over a box clipped at the border and
every catchment starts *at* the border, so cells just inside are handed almost
no upslope area and come out spuriously stable.

The fix implemented in ``pipeline.run_admin_unit`` is to route over the unit's
box grown by ``admin_buffer_deg`` and clip the output back afterwards. This
measures whether that works, and how wide the buffer has to be.

The experiment uses one cached DEM as ground truth. A window inside it stands
in for a province; the same window is then routed on its own and with buffers
of increasing width, and both are compared against the same cells taken from
the full-extent routing.

    python analysis/07_boundary_buffer.py
"""

from __future__ import annotations

import numpy as np
import rasterio

import common as K
from h_sim.model import hydrology as H, physical as P

RES = 0.00027778                      # 30 m
BUFFERS_DEG = [0.0, 0.028, 0.083, 0.167, 0.222]

#: The stand-in province: a window well inside the cached extent, leaving room
#: for the widest buffer on every side.
WINDOW = (900, 1300, 1200, 1700)      # row0, row1, col0, col1


def route(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    sca, _ = H.specific_catchment_area(dem, dx, dy)
    return sca


def main() -> None:
    cfg = K.make_config("gorkha", RES)
    layers = K.terrain_layers(cfg, want_precip=False)
    with rasterio.open(layers["dem"]) as src:
        dem = src.read(1).astype("float64")
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
    dx, dy = K.pipeline.metres_per_cell(cfg.clipped_bbox(), RES)
    print(f"reference DEM {dem.shape[0]}x{dem.shape[1]} at {dx:.0f}x{dy:.0f} m")

    r0, r1, c0, c1 = WINDOW
    print(f"stand-in province: rows {r0}-{r1}, cols {c0}-{c1} "
          f"({(r1 - r0) * (c1 - c0):,} cells)")

    print("\nrouting the full extent as ground truth...")
    sca_ref = route(dem, dx, dy)[r0:r1, c0:c1]
    slope_ref = H.dinf_flow_direction(H.fill_depressions(dem), dx, dy)[1]
    slope_win = slope_ref[r0:r1, c0:c1]

    params = P.SoilParameters((0.0, 0.25), (25.0, 35.0), (1e-5, 5e-4))
    p_ref = P.failure_probability(slope_win, sca_ref, params, n_samples=200)

    rows = []
    for buf_deg in BUFFERS_DEG:
        b = int(round(buf_deg / RES))
        rr0, rr1 = max(r0 - b, 0), min(r1 + b, dem.shape[0])
        cc0, cc1 = max(c0 - b, 0), min(c1 + b, dem.shape[1])
        sub = dem[rr0:rr1, cc0:cc1]
        sca = route(sub, dx, dy)[r0 - rr0:r1 - rr0, c0 - cc0:c1 - cc0]

        ok = np.isfinite(sca) & np.isfinite(sca_ref) & (sca_ref > 0)
        ratio = sca[ok] / sca_ref[ok]
        p = P.failure_probability(slope_win, sca, params, n_samples=200)
        dp = p - p_ref
        good = np.isfinite(dp)

        # How far into the province does the damage reach? Compare the error in
        # the outermost ring of cells against the interior.
        edge = np.zeros(sca.shape, bool)
        w = 30
        edge[:w, :] = edge[-w:, :] = edge[:, :w] = edge[:, -w:] = True
        edge &= ok

        rows.append({
            "buffer_deg": buf_deg,
            "buffer_km": round(buf_deg * 111.0, 1),
            "buffer_cells": b,
            "median_sca_ratio": round(float(np.median(ratio)), 4),
            "pct_under_half": round(float((ratio < 0.5).mean() * 100), 2),
            "pct_under_half_edge": round(float(
                (sca[edge] / sca_ref[edge] < 0.5).mean() * 100), 2),
            "mean_abs_dP": round(float(np.abs(dp[good]).mean()), 5),
            "max_abs_dP": round(float(np.abs(dp[good]).max()), 4),
            "pct_dP_over_0.05": round(float(
                (np.abs(dp[good]) > 0.05).mean() * 100), 3),
        })
        print(f"  buffer {buf_deg:5.3f} deg ({b:4d} cells): "
              f"median SCA ratio {rows[-1]['median_sca_ratio']:.3f}, "
              f"{rows[-1]['pct_under_half']:5.2f}% of cells under half, "
              f"mean |dP| {rows[-1]['mean_abs_dP']:.5f}")

    print("\n\nBOUNDARY BUFFER  (province routed alone vs within its "
          "surroundings)\n")
    print(K.table(rows, [
        ("buffer_deg", "buffer", ".3f"),
        ("buffer_km", "km", ".1f"),
        ("median_sca_ratio", "SCA ratio", ".4f"),
        ("pct_under_half", "% <half", ".2f"),
        ("pct_under_half_edge", "% <half edge", ".2f"),
        ("mean_abs_dP", "mean |dP|", ".5f"),
        ("pct_dP_over_0.05", "% dP>.05", ".3f"),
    ]))
    print("\n  SCA ratio is the routed value over the ground truth; 1.000 is "
          "perfect.\n  dP is the change in failure probability the error "
          "causes.")

    K.save("07_boundary_buffer", {"rows": rows, "resolution_deg": RES,
                                  "window": list(WINDOW)})


if __name__ == "__main__":
    main()
