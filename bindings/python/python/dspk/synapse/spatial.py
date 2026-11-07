"""Space: where each spike comes from on the probe (four localisers), and how the probe drifted
(rigid or by depth blocks). Positions and depths are µm, in the probe's coordinates.
"""

from __future__ import annotations

import numpy as np

from dspk._native import synapse as _synapse
from dspk.synapse.io import Probe

_s = _synapse.spatial

DEFAULT_COM_POWER: float = _s.DEFAULT_COM_POWER
DEFAULT_MONOPOLAR_ITERATIONS: int = _s.DEFAULT_MONOPOLAR_ITERATIONS
DEFAULT_DIPOLE_ITERATIONS: int = _s.DEFAULT_DIPOLE_ITERATIONS
DEFAULT_GRID_XY_RADIUS_UM: float = _s.DEFAULT_GRID_XY_RADIUS_UM
DEFAULT_GRID_XY_STEP_UM: float = _s.DEFAULT_GRID_XY_STEP_UM
DEFAULT_GRID_Z_VALUES_UM: tuple[float, ...] = tuple(_s.DEFAULT_GRID_Z_VALUES_UM)
DEFAULT_GRID_SOFTMAX_TEMPERATURE: float = _s.DEFAULT_GRID_SOFTMAX_TEMPERATURE
DEFAULT_DRIFT_TIME_BIN_S: float = _s.DEFAULT_DRIFT_TIME_BIN_S
DEFAULT_DRIFT_DEPTH_BIN_UM: float = _s.DEFAULT_DRIFT_DEPTH_BIN_UM
DEFAULT_MAX_DRIFT_UM: float = _s.DEFAULT_MAX_DRIFT_UM
DEFAULT_DRIFT_DEPTH_BLOCKS: int = _s.DEFAULT_DRIFT_DEPTH_BLOCKS

METHODS = ("center_of_mass", "monopolar", "grid_convolution", "dipole")


def localize(
    snippets: dict,
    probe: Probe,
    *,
    method: str = "center_of_mass",
    power: float = DEFAULT_COM_POWER,
    max_iterations: int | None = None,
    xy_radius_um: float = DEFAULT_GRID_XY_RADIUS_UM,
    xy_step_um: float = DEFAULT_GRID_XY_STEP_UM,
    z_values_um=DEFAULT_GRID_Z_VALUES_UM,
    softmax_temperature: float = DEFAULT_GRID_SOFTMAX_TEMPERATURE,
) -> np.ndarray:
    """Where each spike comes from on the probe.

    Parameters
    ----------
    snippets : dict
        ``waveforms`` ``(spikes, k, samples)`` and ``channels`` ``(spikes, k)``, the spike's own
        channel first, as :func:`dspk.synapse.extraction.snippets` returns them.
    probe : Probe
        The probe.
    method : {"center_of_mass", "monopolar", "grid_convolution", "dipole"}
        ``"center_of_mass"``: the sites weighted by peak-to-peak height to ``power``.
        ``"monopolar"``: a point source fitted by Levenberg–Marquardt. ``"grid_convolution"``: a
        grid of candidate sources, weighted by a softmax. ``"dipole"``: a dipole fitted to the
        signed peaks.
    power : float
        Exponent of the weights (``"center_of_mass"``).
    max_iterations : int, optional
        Fit iterations (``"monopolar"``, ``"dipole"``); by default
        :data:`DEFAULT_MONOPOLAR_ITERATIONS` or :data:`DEFAULT_DIPOLE_ITERATIONS`.
    xy_radius_um, xy_step_um : float
        Extent and step of the candidate grid (``"grid_convolution"``).
    z_values_um : sequence of float
        Candidate distances from the probe plane (``"grid_convolution"``).
    softmax_temperature : float
        Sharpness of the grid weights (``"grid_convolution"``).

    Returns
    -------
    numpy.ndarray
        Positions ``(spikes, 3)`` in µm, in the probe's coordinates.

    Raises
    ------
    ValueError
        For an unknown ``method``.
    """
    if method not in METHODS:
        raise ValueError(f"method is one of {METHODS}, got {method!r}")
    if max_iterations is None:
        max_iterations = DEFAULT_DIPOLE_ITERATIONS if method == "dipole" else DEFAULT_MONOPOLAR_ITERATIONS
    w = np.ascontiguousarray(snippets["waveforms"], dtype=np.float32)
    c = np.ascontiguousarray(snippets["channels"], dtype=np.int64)
    return _s.localize(w, c, probe._native, method=method, power=power, max_iterations=max_iterations, xy_radius_um=xy_radius_um,
                       xy_step_um=xy_step_um, z_values_um=[float(z) for z in z_values_um], softmax_temperature=softmax_temperature)


def drift(
    samples,
    depths_um,
    amplitudes_uv,
    total_samples: int,
    sample_rate: float,
    *,
    depth_min_um: float,
    depth_max_um: float,
    time_bin_s: float = DEFAULT_DRIFT_TIME_BIN_S,
    depth_bin_um: float = DEFAULT_DRIFT_DEPTH_BIN_UM,
    max_drift_um: float = DEFAULT_MAX_DRIFT_UM,
    depth_blocks: int = DEFAULT_DRIFT_DEPTH_BLOCKS,
    rigid: bool = True,
) -> dict:
    """How the probe moved over the recording, from where and when spikes occurred.

    Parameters
    ----------
    samples : array_like
        Sample of every spike.
    depths_um : array_like
        Depth of every spike, in µm.
    amplitudes_uv : array_like
        Amplitude of every spike, in µV.
    total_samples : int
        Samples in the recording.
    sample_rate : float
        Sample rate in Hz.
    depth_min_um, depth_max_um : float
        The depth range considered.
    time_bin_s : float
        Time resolution of the estimate.
    depth_bin_um : float
        Depth resolution of the activity map.
    max_drift_um : float
        Largest shift searched.
    depth_blocks : int
        Blocks along depth, each with its own drift (``rigid=False``).
    rigid : bool
        One drift for the whole probe, or one per depth block.

    Returns
    -------
    dict
        Rigid: ``times_s``, ``drift_um``, ``activity`` ``(time bins, depth bins)``. Non-rigid:
        ``times_s``, ``block_centers_um``, ``drift_um`` ``(depth_blocks, time bins)``.
    """
    return _s.drift(np.ascontiguousarray(samples, dtype=np.uint64), np.ascontiguousarray(depths_um, dtype=np.float32),
                    np.ascontiguousarray(amplitudes_uv, dtype=np.float32), int(total_samples), float(sample_rate),
                    depth_min_um=depth_min_um, depth_max_um=depth_max_um, time_bin_s=time_bin_s, depth_bin_um=depth_bin_um,
                    max_drift_um=max_drift_um, depth_blocks=depth_blocks, rigid=rigid)


__all__ = [
    "DEFAULT_COM_POWER", "DEFAULT_DIPOLE_ITERATIONS", "DEFAULT_DRIFT_DEPTH_BIN_UM", "DEFAULT_DRIFT_DEPTH_BLOCKS", "DEFAULT_DRIFT_TIME_BIN_S",
    "DEFAULT_GRID_SOFTMAX_TEMPERATURE", "DEFAULT_GRID_XY_RADIUS_UM", "DEFAULT_GRID_XY_STEP_UM", "DEFAULT_GRID_Z_VALUES_UM",
    "DEFAULT_MAX_DRIFT_UM", "DEFAULT_MONOPOLAR_ITERATIONS", "METHODS", "drift", "localize",
]
