"""Package imports and numerical kernels."""

from geostats import __version__
import geostats.convex  # noqa: F401
import geostats.diffusion  # noqa: F401
import geostats.inference  # noqa: F401
import geostats.integral  # noqa: F401
import geostats.statistical  # noqa: F401


def test_version():
    assert __version__ == "0.1.0"
