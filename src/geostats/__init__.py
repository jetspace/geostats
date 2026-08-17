"""geostats: lecture kernels for geometric inference.

Mathematics lives in ``notebooks/program.ipynb``. Numerics live here.
Primary sources [1] through [5] are listed in REFERENCES.md.
"""

from geostats.convex import minkowski_sum_h
from geostats.statistical import fisher

__version__ = "0.1.0"

__all__ = ["__version__", "minkowski_sum_h", "fisher"]
