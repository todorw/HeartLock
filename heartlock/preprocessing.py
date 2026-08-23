"""Cleans up raw ECG signal before beat detection and feature extraction."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# 0.5 Hz high-pass removes baseline wander from respiration and electrode
# motion (typically < 0.5 Hz) without eating into the QRS complex.
# 40 Hz low-pass removes EMG/muscle noise and high-frequency interference;
# QRS energy is concentrated below ~25 Hz so 40 Hz leaves it intact while
# still cutting most non-cardiac noise.
DEFAULT_LOWCUT = 0.5
DEFAULT_HIGHCUT = 40.0
DEFAULT_BANDPASS_ORDER = 4

# Powerline interference quality factor: higher Q means a narrower notch,
# which removes less of the surrounding signal. 30 is a common choice that
# suppresses the powerline tone without visibly distorting nearby QRS content.
DEFAULT_NOTCH_QUALITY = 30.0


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = DEFAULT_LOWCUT,
    highcut: float = DEFAULT_HIGHCUT,
    order: int = DEFAULT_BANDPASS_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    filtfilt (forward-backward) is used instead of a single-pass filter so
    the output has no phase shift, which matters here because we later
    measure interval timings (PR, QT) directly off the filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(
    signal: np.ndarray,
    fs: float,
    freq: float = 50.0,
    quality: float = DEFAULT_NOTCH_QUALITY,
) -> np.ndarray:
    """Remove powerline interference at `freq` Hz (50 for EU, 60 for US)."""
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, signal)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance scaling.

    Z-score (rather than min-max) is used because ECG amplitude is not
    bounded by a fixed physiological range and a single outlier beat
    shouldn't compress the rest of the signal's dynamic range.
    """
    std = np.std(signal)
    if std == 0:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def preprocess(
    signal: np.ndarray,
    fs: float,
    powerline_freq: float = 50.0,
    lowcut: float = DEFAULT_LOWCUT,
    highcut: float = DEFAULT_HIGHCUT,
) -> np.ndarray:
    """Full cleanup pipeline: bandpass -> notch -> normalize."""
    filtered = bandpass_filter(signal, fs, lowcut, highcut)
    filtered = notch_filter(filtered, fs, powerline_freq)
    return normalize(filtered)
