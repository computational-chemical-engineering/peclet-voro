"""peclet.voro — dynamic 3D Voronoi tessellation of moving particles.

A device-native (Kokkos) moving-cell Voronoi engine: periodic & Lees–Edwards boxes, incremental cell
repair, and compressible Euler / Navier–Stokes / multiphase dynamics on the moving cells. Also serves as
an unstructured-mesh generator that can feed an Eulerian solve in :mod:`peclet.flow`. The compiled
backend (Serial / OpenMP / CUDA / HIP) is chosen at build time — ``peclet.voro.execution_space`` reports
which one this build has.

* :class:`peclet.voro.Tessellation`, :class:`peclet.voro.Simulation`.

``peclet`` is an implicit (PEP 420) namespace shared with the other ``peclet-*`` packages, so it has no
top-level ``__init__.py``.
"""

from ._voro import *  # noqa: F401,F403

__version__ = "0.3.2"
