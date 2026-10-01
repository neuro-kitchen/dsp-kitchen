"""
PEP 484 type stubs for the compiled Rust `dsp_kitchen_bindings` PyO3 extension module.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import numpy.typing as npt

__version__: str

class ProbeLayout:
    @staticmethod
    def neuropixels_1_0() -> "ProbeLayout": ...
    @staticmethod
    def neuropixels_2_0() -> "ProbeLayout": ...
    @staticmethod
    def tetrode() -> "ProbeLayout": ...
    @staticmethod
    def utah_array() -> "ProbeLayout": ...
    @staticmethod
    def from_positions(
        name: str,
        positions: List[Tuple[float, float]],
        shank_ids: Optional[List[int]] = None,
    ) -> "ProbeLayout": ...
    @property
    def name(self) -> str: ...
    @property
    def num_channels(self) -> int: ...

class SpikeEvent:
    @property
    def channel_id(self) -> int: ...
    @property
    def sample_index(self) -> int: ...
    @property
    def peak_amplitude_uv(self) -> float: ...

class DeduplicatedSpike:
    @property
    def primary_channel(self) -> int: ...
    @property
    def sample_index(self) -> int: ...
    @property
    def peak_amplitude_uv(self) -> float: ...
    @property
    def participating_channels(self) -> List[int]: ...

class WaveformSnippet:
    @property
    def primary_channel(self) -> int: ...
    @property
    def center_sample(self) -> int: ...
    @property
    def subsample_offset(self) -> float: ...
    @property
    def channel_ids(self) -> List[int]: ...
    @property
    def num_samples(self) -> int: ...
    @property
    def num_channels(self) -> int: ...
    def waveform(self) -> npt.NDArray[np.float32]: ...

class MmapRecording:
    """Memory-mapped raw binary recording. Layout from the JSON sidecar (`rec.bin` -> `rec.meta`)
    or from `channels` + `sample_rate` (+ dtype / order / gain)."""
    def __init__(
        self,
        path: str,
        channels: Optional[int] = None,
        sample_rate: Optional[float] = None,
        dtype: Literal["float32", "int16", "uint16"] = "float32",
        order: Literal["channel_major", "time_major"] = "channel_major",
        gain_uv: float = 1.0,
        offset_uv: float = 0.0,
        header_bytes: int = 0,
        samples: Optional[int] = None,
    ) -> None: ...
    @property
    def path(self) -> str: ...
    @property
    def channels(self) -> int: ...
    @property
    def samples(self) -> int: ...
    @property
    def sample_rate(self) -> float: ...
    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def dtype(self) -> str: ...
    @property
    def total_bytes(self) -> int: ...
    def read(
        self,
        start_sample: int = 0,
        end_sample: Optional[int] = None,
        channels: Optional[List[int]] = None,
    ) -> npt.NDArray[np.float32]:
        """µV float32 copy [channels, samples]; default window 1 s."""
    def memoryview(self) -> memoryview:
        """Zero-copy read-only view of the stored bytes; keeps the recording alive."""
    def to_numpy(self) -> npt.NDArray[Any]:
        """Zero-copy read-only [channels, samples] view in the stored dtype; keeps the recording alive."""

class NwbZarrRecording:
    def __init__(self, path: str, series: Optional[str] = None) -> None: ...
    @property
    def path(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def channels(self) -> int: ...
    @property
    def samples(self) -> int: ...
    @property
    def sample_rate(self) -> float: ...
    @property
    def duration_sec(self) -> float: ...
    @property
    def start_time_sec(self) -> float: ...
    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def channel_names(self) -> List[str]: ...
    @property
    def unit(self) -> str: ...
    @property
    def series(self) -> Optional[str]: ...
    @property
    def metadata(self) -> Dict[str, str]: ...
    def slice_samples(
        self,
        start_sample: int = 0,
        end_sample: Optional[int] = None,
        channels: Optional[List[int]] = None,
    ) -> "NwbZarrRecording": ...
    def slice_time(
        self,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        channels: Optional[List[int]] = None,
    ) -> "NwbZarrRecording": ...
    def __getitem__(self, key: Any) -> "NwbZarrRecording": ...
    def read(
        self,
        start_sample: int = 0,
        end_sample: Optional[int] = None,
        channels: Optional[List[int]] = None,
    ) -> npt.NDArray[np.float32]:
        """µV float32 [channels, samples]; default window 1 s. IndexError / ValueError on bad input."""
    def read_window(
        self,
        start_sec: float = 0.0,
        duration_sec: float = 0.1,
        channels: Optional[List[int]] = None,
    ) -> npt.NDArray[np.float32]: ...

def list_nwb_series(path: str) -> List[Dict[str, Any]]: ...

class Scale:
    def __init__(self, gain: float) -> None: ...

class SubtractBaseline:
    def __init__(self) -> None: ...

class Clamp:
    def __init__(self, min_val: float, max_val: float) -> None: ...

Direction = Literal["forward-backward", "forward"]

class NotchFilter:
    """Second-order notch (scipy `iirnotch`)."""
    def __init__(
        self, freq_hz: float = 60.0, q: float = 30.0, direction: Direction = "forward-backward"
    ) -> None: ...

class BandpassFilter:
    """Butterworth band-pass; `order` per edge (scipy / SpikeInterface convention)."""
    def __init__(
        self,
        low_hz: float = 300.0,
        high_hz: float = 6000.0,
        order: int = 5,
        direction: Direction = "forward-backward",
    ) -> None: ...

class HighpassFilter:
    def __init__(
        self, cutoff_hz: float = 300.0, order: int = 5, direction: Direction = "forward-backward"
    ) -> None: ...

class LowpassFilter:
    def __init__(
        self, cutoff_hz: float = 300.0, order: int = 5, direction: Direction = "forward-backward"
    ) -> None: ...

class BandstopFilter:
    def __init__(
        self, low_hz: float, high_hz: float, order: int = 5, direction: Direction = "forward-backward"
    ) -> None: ...

class CommonAverageReference:
    def __init__(self) -> None: ...

class MedianFilter:
    def __init__(self) -> None: ...

class TeagerKaiser:
    def __init__(self) -> None: ...

class TemplateFilter:
    def __init__(self, template: List[float], event_indices: List[int]) -> None: ...

class Pipeline:
    def __init__(self, stages: Optional[List[object]] = None) -> None: ...
    def add(self, stage: object) -> None: ...
    @property
    def stages(self) -> List[str]: ...
    def __len__(self) -> int: ...
    def settling(self, fs: float = 30000.0) -> Tuple[int, int]:
        """(left, right) samples of context a chunk needs to match whole-recording filtering."""
    def run(
        self,
        data: npt.NDArray[np.float32],
        fs: float = 30000.0,
        channels: Optional[int] = None,
    ) -> npt.NDArray[np.float32]: ...

class StreamingSortResult:
    @property
    def channels(self) -> int: ...
    @property
    def total_samples(self) -> int: ...
    @property
    def sample_rate(self) -> float: ...
    @property
    def halos(self) -> Tuple[int, int]: ...
    @property
    def channel_sigmas_uv(self) -> List[float]: ...
    @property
    def total_raw_crossings(self) -> int: ...
    @property
    def total_dedup_spikes(self) -> int: ...
    @property
    def channel_spike_counts(self) -> List[int]: ...
    def spikes(self) -> List[DeduplicatedSpike]: ...
    def template(self, channel: int) -> Optional[Dict[str, Any]]: ...
    def all_templates(self) -> List[Optional[Dict[str, Any]]]: ...

def sort_recording(
    recording: NwbZarrRecording,
    pipeline: Pipeline,
    probe: ProbeLayout,
    threshold_factor: float = 5.0,
    refractory_ms: float = 1.0,
    spatial_radius_um: float = 150.0,
    k_neighbors: int = 4,
    pre_ms: float = 1.0,
    post_ms: float = 2.0,
    batch_duration_sec: float = 10.0,
    calibration_duration_sec: float = 5.0,
    calibration_chunks: int = 5,
    apply_sinc_shift: bool = True,
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
) -> StreamingSortResult: ...

class DspSession:
    def __init__(self, channels: int, chunk_samples: int, backend: str = "cpu") -> None: ...

class PCA:
    def __init__(self, n_components: int = 3) -> None: ...
    def fit_transform(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

class ModelHub:
    def __init__(self, cache_dir: Optional[str] = None) -> None: ...
    @property
    def cache_dir(self) -> str: ...
    def list(self, family: Optional[str] = None) -> List[Dict[str, Any]]: ...
    def info(self, model_id: str) -> Dict[str, Any]: ...
    def pull(self, model_id: str, force: bool = False) -> str: ...
    def verify(self, model_id: str, check_remote: bool = False) -> Dict[str, Any]: ...
    def remove(self, model_id: str) -> bool: ...
    def clean(self) -> int: ...

class Kilosort4BasisEmbedder:
    def __init__(
        self, path: Optional[str] = None, backend: Optional[str] = None
    ) -> None: ...
    @staticmethod
    def from_hub(backend: Optional[str] = None) -> "Kilosort4BasisEmbedder": ...
    @staticmethod
    def from_npy(
        path: str, backend: Optional[str] = None
    ) -> "Kilosort4BasisEmbedder": ...
    @property
    def num_components(self) -> int: ...
    @property
    def window_len(self) -> int: ...
    @property
    def source_path(self) -> str: ...
    def basis_matrix(self) -> npt.NDArray[np.float32]: ...
    def project(self, waveforms: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
    def embed(self, snippets: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
    def reconstruct(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

class Kilosort4Detector:
    def __init__(
        self,
        threshold_sigma: float = 4.5,
        refractory_samples: int = 30,
        path: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> None: ...
    @staticmethod
    def from_hub(
        threshold_sigma: float = 4.5,
        refractory_samples: int = 30,
        backend: Optional[str] = None,
    ) -> "Kilosort4Detector": ...
    @staticmethod
    def from_npy(
        path: str,
        threshold_sigma: float = 4.5,
        refractory_samples: int = 30,
        backend: Optional[str] = None,
    ) -> "Kilosort4Detector": ...
    @property
    def num_templates(self) -> int: ...
    @property
    def window_len(self) -> int: ...
    @property
    def center_offset(self) -> int: ...
    @property
    def source_path(self) -> str: ...
    def templates_matrix(self) -> npt.NDArray[np.float32]: ...
    def filter_energy(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
    def detect(
        self, data: npt.NDArray[np.float32], sample_rate_hz: float = 30000.0
    ) -> List[SpikeEvent]: ...



def notch_filter(
    data: npt.NDArray[np.float32],
    freq: float = 60.0,
    q: float = 30.0,
    fs: float = 30000.0,
    direction: Direction = "forward-backward",
) -> npt.NDArray[np.float32]: ...
def bandpass_filter(
    data: npt.NDArray[np.float32],
    low: float = 300.0,
    high: float = 6000.0,
    fs: float = 30000.0,
    order: int = 5,
    direction: Direction = "forward-backward",
) -> npt.NDArray[np.float32]: ...
def highpass_filter(
    data: npt.NDArray[np.float32],
    cutoff: float = 300.0,
    fs: float = 30000.0,
    order: int = 5,
    direction: Direction = "forward-backward",
) -> npt.NDArray[np.float32]: ...
def lowpass_filter(
    data: npt.NDArray[np.float32],
    cutoff: float = 300.0,
    fs: float = 30000.0,
    order: int = 5,
    direction: Direction = "forward-backward",
) -> npt.NDArray[np.float32]: ...
def common_average_reference(
    data: npt.NDArray[np.float32],
    channels: Optional[int] = None,
    backend: str = "cpu",
) -> npt.NDArray[np.float32]: ...
def scale_samples(
    data: npt.NDArray[np.float32],
    gain: float,
    channels: Optional[int] = None,
    backend: str = "cpu",
) -> npt.NDArray[np.float32]: ...
def median_filter_9p(
    data: npt.NDArray[np.float32],
    channels: Optional[int] = None,
    backend: str = "cpu",
) -> npt.NDArray[np.float32]: ...
def teager_kaiser_filter(
    data: npt.NDArray[np.float32],
    channels: Optional[int] = None,
    backend: str = "cpu",
) -> npt.NDArray[np.float32]: ...
def subtract_template(
    data: npt.NDArray[np.float32],
    template: List[float],
    event_indices: List[int],
    channels: Optional[int] = None,
) -> npt.NDArray[np.float32]: ...
def detect_spikes(
    data: npt.NDArray[np.float32],
    channels: Optional[int] = None,
    threshold_multiplier: float = 4.5,
    refractory_samples: int = 30,
) -> List[SpikeEvent]: ...
def deduplicate_spikes(
    events: List[SpikeEvent],
    probe: ProbeLayout,
    spatial_radius_um: float = 75.0,
    temporal_window_samples: int = 15,
) -> List[DeduplicatedSpike]: ...
def estimate_noise(
    data: npt.NDArray[np.float32],
    channels: Optional[int] = None,
) -> List[float]: ...
def extract_snippets(
    data: npt.NDArray[np.float32],
    spikes: List[DeduplicatedSpike],
    probe: ProbeLayout,
    channels: Optional[int] = None,
    k_neighbors: int = 7,
    pre_samples: int = 20,
    post_samples: int = 40,
    apply_sinc_shift: bool = True,
) -> List[WaveformSnippet]: ...
def compute_isi(
    spike_samples: List[int],
    sample_rate_hz: float = 30000.0,
    refractory_ms: float = 1.5,
    total_duration_sec: Optional[float] = None,
    min_isi_ms: float = 0.0,
) -> Dict[str, float]:
    """ISI violations (SpikeInterface `isi_violations`): total_spikes, violation_count,
    violation_rate_pct, isi_violations_ratio (Hill), violations_per_sec, firing_rate_hz."""
def compute_snr(
    snippets: List[WaveformSnippet],
    baseline_noise_uv: float,
) -> float: ...
def compute_template(
    snippets: List[WaveformSnippet],
) -> npt.NDArray[np.float32]: ...
