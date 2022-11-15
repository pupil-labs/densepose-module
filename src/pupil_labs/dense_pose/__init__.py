"""Top-level entry-point for the <pl-densepose> package"""

import sys

if sys.version_info < (3, 8):
    from importlib_metadata import PackageNotFoundError, version
else:
    from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil-labs-dense-pose")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["__version__"]

from . import __main__
