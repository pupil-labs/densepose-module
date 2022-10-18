"""Top-level entry-point for the <pl-densepose> package"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs_dense_pose")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["__version__"]

from . import main

if __name__ == "__main__":
    main.main()
