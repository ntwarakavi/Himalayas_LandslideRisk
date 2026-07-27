"""GIRI-style global/regional landslide hazard model (local, modular, tile-based).

An open-source, end-to-end implementation of the landslide susceptibility and
scenario-based hazard methodology described in:

    Palau, R. M., Nadim, F., Paulsen, E., Storrosten, E. (2023).
    "A new model for global landslide susceptibility assessment and
    scenario-based hazard assessment." Norwegian Geotechnical Institute /
    Global Infrastructure Resilience Index (GIRI), CDRI.
    https://giri.unepgrid.ch

The package is organised into small, independently runnable modules so the
whole pipeline (download -> factors -> susceptibility -> triggers -> hazard)
can be executed piece by piece on a modest local computer, processing the
area of interest in memory-bounded tiles/blocks.
"""

__version__ = "0.1.0"

from .config import Config, DEFAULT_CONFIG  # noqa: F401
