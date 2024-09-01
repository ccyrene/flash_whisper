import numpy as np

from typing import Union, Optional

def hertz_to_mel(freq: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    elif mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0

    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep

    return mels

def mel_to_hertz(mels: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    elif mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0

    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))

    return freq

def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
    triangularize_in_mel_space: bool = False,
) -> np.ndarray:
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # center points of the triangular mel filters
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

    if triangularize_in_mel_space:
        # frequencies of FFT bins in Hz, but filters triangularized in mel space
        fft_bin_width = sampling_rate / (num_frequency_bins * 2)
        fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale)
        filter_freqs = mel_freqs
    else:
        # frequencies of FFT bins in Hz
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters *= np.expand_dims(enorm, 0)

    return mel_filters

def power_to_db(
    spectrogram: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
) -> np.ndarray:

    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")

    reference = max(min_value, reference)

    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    spectrogram = 10.0 * (np.log10(spectrogram) - np.log10(reference))

    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)

    return spectrogram

def amplitude_to_db(
    spectrogram: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-5,
    db_range: Optional[float] = None,
) -> np.ndarray:
    
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")

    reference = max(min_value, reference)

    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    spectrogram = 20.0 * (np.log10(spectrogram) - np.log10(reference))

    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)

    return spectrogram

def spectrogram(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: Optional[int] = None,
    power: Optional[float] = 1.0,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    preemphasis: Optional[float] = None,
    mel_filters: Optional[np.ndarray] = None,
    mel_floor: float = 1e-10,
    log_mel: Optional[str] = None,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
    remove_dc_offset: Optional[bool] = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:

    window_length = len(window)

    if fft_length is None:
        fft_length = frame_length

    if frame_length > fft_length:
        raise ValueError(f"frame_length ({frame_length}) may not be larger than fft_length ({fft_length})")

    if window_length != frame_length:
        raise ValueError(f"Length of the window ({window_length}) must equal frame_length ({frame_length})")

    if hop_length <= 0:
        raise ValueError("hop_length must be greater than zero")

    if waveform.ndim != 1:
        raise ValueError(f"Input waveform must have only one dimension, shape is {waveform.shape}")

    if np.iscomplexobj(waveform):
        raise ValueError("Complex-valued input waveforms are not currently supported")

    if power is None and mel_filters is not None:
        raise ValueError(
            "You have provided `mel_filters` but `power` is `None`. Mel spectrogram computation is not yet supported for complex-valued spectrogram."
            "Specify `power` to fix this issue."
        )

    # center pad the waveform
    if center:
        padding = [(int(frame_length // 2), int(frame_length // 2))]
        waveform = np.pad(waveform, padding, mode=pad_mode)

    # promote to float64, since np.fft uses float64 internally
    waveform = waveform.astype(np.float64)
    window = window.astype(np.float64)

    # split waveform into frames of frame_length size
    num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))

    num_frequency_bins = (fft_length // 2) + 1 if onesided else fft_length
    spectrogram = np.empty((num_frames, num_frequency_bins), dtype=np.complex64)

    # rfft is faster than fft
    fft_func = np.fft.rfft if onesided else np.fft.fft
    buffer = np.zeros(fft_length)

    timestep = 0
    for frame_idx in range(num_frames):
        buffer[:frame_length] = waveform[timestep : timestep + frame_length]

        if remove_dc_offset:
            buffer[:frame_length] = buffer[:frame_length] - buffer[:frame_length].mean()

        if preemphasis is not None:
            buffer[1:frame_length] -= preemphasis * buffer[: frame_length - 1]
            buffer[0] *= 1 - preemphasis

        buffer[:frame_length] *= window

        spectrogram[frame_idx] = fft_func(buffer)
        timestep += hop_length

    # note: ** is much faster than np.power
    if power is not None:
        spectrogram = np.abs(spectrogram, dtype=np.float64) ** power

    spectrogram = spectrogram.T

    if mel_filters is not None:
        spectrogram = np.maximum(mel_floor, np.dot(mel_filters.T, spectrogram))

    if power is not None and log_mel is not None:
        if log_mel == "log":
            spectrogram = np.log(spectrogram)
        elif log_mel == "log10":
            spectrogram = np.log10(spectrogram)
        elif log_mel == "dB":
            if power == 1.0:
                spectrogram = amplitude_to_db(spectrogram, reference, min_value, db_range)
            elif power == 2.0:
                spectrogram = power_to_db(spectrogram, reference, min_value, db_range)
            else:
                raise ValueError(f"Cannot use log_mel option '{log_mel}' with power {power}")
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")

        spectrogram = np.asarray(spectrogram, dtype)

    return spectrogram

def window_function(
    window_length: int,
    name: str = "hann",
    periodic: bool = True,
    frame_length: Optional[int] = None,
    center: bool = True,
) -> np.ndarray:
    
    length = window_length + 1 if periodic else window_length

    if name == "boxcar":
        window = np.ones(length)
    elif name in ["hamming", "hamming_window"]:
        window = np.hamming(length)
    elif name in ["hann", "hann_window"]:
        window = np.hanning(length)
    elif name in ["povey"]:
        window = np.power(np.hanning(length), 0.85)
    else:
        raise ValueError(f"Unknown window function '{name}'")

    if periodic:
        window = window[:-1]

    if frame_length is None:
        return window

    if window_length > frame_length:
        raise ValueError(
            f"Length of the window ({window_length}) may not be larger than frame_length ({frame_length})"
        )

    padded_window = np.zeros(frame_length)
    offset = (frame_length - window_length) // 2 if center else 0
    padded_window[offset : offset + window_length] = window
    return padded_window

def stft_np(waveform, n_fft, hop_length):
    
    pad = int(n_fft // 2)
    waveform = np.pad(waveform, ((0,0), (pad,pad)), "reflect")
    batch_size, signal_length = waveform.shape
    window_fn = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    num_frames = 1 + (signal_length - n_fft) // hop_length
    
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(batch_size, num_frames, n_fft),
        strides=(waveform.strides[0], waveform.strides[1] * hop_length, waveform.strides[1])
    ) * window_fn[np.newaxis, np.newaxis, :]
    
    stft_matrix = np.fft.rfft(frames, n=n_fft, axis=2)
    
    return stft_matrix.transpose(0, 2, 1)