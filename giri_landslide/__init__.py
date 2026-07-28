"""Physically based landslide hazard model for the Hindu Kush Himalaya.

An open-source, end-to-end implementation of infinite-slope stability under
steady-state wetness, fitted to mapped landslide inventories:

    Pack, R. T., Tarboton, D. G., Goodwin, C. N. (1998).
    "The SINMAP approach to terrain stability mapping."
    8th Congress of the International Association of Engineering Geology.

    Tarboton, D. G. (1997). "A new method for the determination of flow
    directions and upslope areas in grid digital elevation models."
    Water Resources Research 33(2), 309-319.

    Barnes, R., Lehman, C., Mulla, D. (2014). "Priority-flood: an optimal
    depression-filling and watershed-labeling algorithm for digital elevation
    models." Computers & Geosciences 62, 117-127.

The package is organised into small, independently runnable modules so the
whole workflow (download -> fit -> stability -> hazard) can be executed piece
by piece on a modest local computer.
"""

__version__ = "0.2.0"

from .config import Config, DEFAULT_CONFIG  # noqa: F401
