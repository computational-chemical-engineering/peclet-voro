"""peclet.voro — dynamic 3D Voronoi tessellation of moving particles.

A device-native (Kokkos) moving-cell Voronoi engine: periodic boxes, incremental cell repair, and
compressible Euler / Navier–Stokes / multiphase dynamics on the moving cells. Also serves as
an unstructured-mesh generator that can feed an Eulerian solve in :mod:`peclet.flow`. The compiled
backend (Serial / OpenMP / CUDA / HIP) is chosen at build time — ``peclet.voro.execution_space`` reports
which one this build has.

The public surface (suite/docs/QUALITY_PLAN.md D2 — everything else is on each object's
``diagnostics``):

* :class:`Tessellation` — cold build + incremental repair of a moving point set, volumes, neighbour
  and wall counts, the energy layer (``energy_forces``).
* :class:`FlowSolver` — the static collocated / covolume Navier–Stokes solvers on the face mesh.
* :class:`Simulation` — the moving-cell compressible-Euler / Navier–Stokes fluid.
* :func:`optimize_volume_mesh`, :func:`minimize_interface` — mesh optimisers on a periodic box.
* :mod:`peclet.voro.pore_mesh` (imported on first use) — the SDF-walled pore-space family:
  ``optimize_pore_mesh``, ``redistribute_pore_mesh``, ``sdf_voronoi_cells``, ``sdf_voronoi_section``.
* :mod:`peclet.voro.scenes` (imported on first use) — scene helpers: ``sphere_union_scene``,
  ``sphere_union_sdf``.
* :class:`VoronoiHalo`, :class:`DistributedTessellation` — the MPI path (builds with
  ``PECLET_VORO_MPI=ON`` only).
* ``defaults`` — the named defaults the engine is driven with; ``execution_space``; ``finalize``.

``peclet`` is an implicit (PEP 420) namespace shared with the other ``peclet-*`` packages, so it has no
top-level ``__init__.py``.
"""

from . import _voro
from ._voro import (  # noqa: F401
    FlowSolver,
    InterfaceResult,
    OptimizeResult,
    Simulation,
    Tessellation,
    defaults,
    execution_space,
    finalize,
    minimize_interface,
    optimize_volume_mesh,
)

__all__ = [
    "Tessellation",
    "FlowSolver",
    "Simulation",
    "optimize_volume_mesh",
    "minimize_interface",
    "OptimizeResult",
    "InterfaceResult",
    "defaults",
    "execution_space",
    "finalize",
    "pore_mesh",
    "scenes",
]

# The distributed classes exist only in an MPI build of the extension.
if hasattr(_voro, "VoronoiHalo"):
    from ._voro import DistributedTessellation, VoronoiHalo  # noqa: F401

    __all__ += ["VoronoiHalo", "DistributedTessellation"]

# The installed distribution's metadata (pyproject.toml) is the single source of truth for the version;
# a build-tree import (PYTHONPATH=<build>) has no metadata and reports "0+unknown". This replaces a
# hand-maintained literal that had drifted behind pyproject.toml in every package at 0.6.0.
try:
    from importlib.metadata import PackageNotFoundError as _PNF, version as _dist_version
    try:
        __version__ = _dist_version("peclet-voro")
    except _PNF:  # the CUDA wheel installs the same module under the -cu13 distribution name
        __version__ = _dist_version("peclet-voro-cu13")
except Exception:  # PackageNotFoundError (dev build), or a broken metadata install
    __version__ = "0+unknown"

_LAZY = ("pore_mesh", "scenes")


def __getattr__(name):
    # PEP 562: `peclet.voro.pore_mesh` / `peclet.voro.scenes` load on first attribute access, so
    # importing the package does not pull the pore-space algorithms (and numpy) in.
    if name in _LAZY:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
