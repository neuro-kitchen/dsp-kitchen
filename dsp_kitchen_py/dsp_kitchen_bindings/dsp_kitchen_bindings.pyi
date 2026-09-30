"""
PEP 484 type stubs for the compiled Rust `dsp_kitchen_bindings` PyO3 extension module.
"""

from typing import List, Optional, Tuple
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
    def __init__(
        self,
        path: str,
        channels: int = 384,
        samples: int = 0,
        sample_rate: float = 30000.0,
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
    def total_bytes(self) -> int: ...
    def memoryview(self) -> memoryview: ...
    def to_numpy(self) -> npt.NDArray[np.float32]: ...

class Scale:
    def __init__(self, gain: float) -> None: ...

class SubtractBaseline:
    def __init__(self) -> None: ...

class Clamp:
    def __init__(self, min_val: float, max_val: float) -> None: ...

class NotchFilter:
    def __init__(self, freq_hz: float, sample_rate: float = 30000.0, q: float = 30.0) -> None: ...

class BandpassFilter:
    def __init__(
        self,
        low_hz: float = 300.0,
        high_hz: float = 6000.0,
        sample_rate: float = 30000.0,
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
    def __init__(self, backend: str = "cpu") -> None: ...
    def add_stage(self, stage: object) -> None: ...
    def run(
        self,
        data: npt.NDArray[np.float32],
        channels: Optional[int] = None,
    ) -> npt.NDArray[np.float32]: ...

class DspSession:
    def __init__(self, channels: int, chunk_samples: int, backend: str = "cpu") -> None: ...

class PCA:
    def __init__(self, n_components: int = 3) -> None: ...
    def fit_transform(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

class SpatiotemporalUnetDenoiser:
    def __init__(
        self,
        num_channels: int = 4,
        num_samples: int = 40,
        base_filters: int = 8,
        seed: int = 42,
    ) -> None: ...
    def denoise(self, snippets: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
    def save_safetensors(self, path: str) -> None: ...
    def load_safetensors(self, path: str) -> None: ...

class SingleChannelDenoiser:
    def __init__(self, hidden_channels: int = 16, seed: int = 42) -> None: ...
    def denoise(self, snippets: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

class DartsortVaeEmbedder:
    def __init__(
        self,
        num_channels: int = 4,
        num_samples: int = 40,
        latent_dim: int = 8,
        seed: int = 42,
    ) -> None: ...
    def embed(self, snippets: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

class ContrastiveWaveformEmbedder:
    def __init__(self, num_channels: int = 4, proj_dim: int = 8, seed: int = 42) -> None: ...
    def embed(self, snippets: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

class UnitQualityClassifier:
    def __init__(self, seed: int = 42) -> None: ...
    def classify(
        self, features: npt.NDArray[np.float32]
    ) -> List[Tuple[str, float, float, float]]: ...

class OnnxModelRunner:
    @staticmethod
    def from_file(path: str) -> "OnnxModelRunner": ...
    @staticmethod
    def from_bytes(bytes: bytes) -> "OnnxModelRunner": ...
    @property
    def node_count(self) -> int: ...
    def run_2d(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
    def run_3d(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
    def embed_with_profile(
        self,
        snippets: npt.NDArray[np.float32],
        sorter: str = "cebra",
        embedding_dim: int = 8,
    ) -> npt.NDArray[np.float32]: ...
    def denoise_dartsort(self, snippets: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

def notch_filter(
    data: npt.NDArray[np.float32],
    freq_hz: float = 60.0,
    sample_rate: float = 30000.0,
    q: float = 30.0,
    channels: Optional[int] = None,
    backend: str = "cpu",
) -> npt.NDArray[np.float32]: ...
def bandpass_filter(
    data: npt.NDArray[np.float32],
    low_hz: float = 300.0,
    high_hz: float = 6000.0,
    sample_rate: float = 30000.0,
    channels: Optional[int] = None,
    backend: str = "cpu",
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
    sample_rate: float = 30000.0,
    refractory_ms: float = 1.5,
) -> Tuple[int, float]: ...
def compute_snr(
    snippets: List[WaveformSnippet],
    baseline_noise_uv: float,
) -> float: ...
def compute_template(
    snippets: List[WaveformSnippet],
) -> npt.NDArray[np.float32]: ...
