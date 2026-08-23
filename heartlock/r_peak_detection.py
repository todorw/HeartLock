"""Pan-Tompkins QRS detector and beat segmentation.

Reference: Pan, J. & Tompkins, W. (1985), "A Real-Time QRS Detection
Algorithm", IEEE Trans. Biomed. Eng.

The classic pipeline is: bandpass (emphasize QRS) -> derivative (emphasize
slope) -> square (emphasize large slopes, discard sign) -> moving-window
integrate (smear each QRS into one broad hump) -> adaptive threshold peak
picking. We run this on top of the already-cleaned signal from
`preprocessing.py`, then map the detected peak locations back onto that
cleaned signal (integration/derivative both shift and blur the peak, so the
raw detector output is not where the actual R peak sample is).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# Pan-Tompkins' original QRS-emphasis band. QRS energy peaks around 10-25 Hz;
# narrowing to 5-15 Hz suppresses P/T waves and most muscle noise while
# keeping the QRS slope information the derivative stage needs.
QRS_BANDPASS_LOW = 5.0
QRS_BANDPASS_HIGH = 15.0

# 150 ms integration window approximates the widest expected QRS complex,
# so the integrator produces one smooth hump per beat instead of multiple
# sub-peaks from a jagged squared derivative.
INTEGRATION_WINDOW_MS = 150.0

# No physiological heartbeat exceeds ~300 bpm, i.e. RR > 200 ms; using this
# as a refractory period stops the detector from double-firing on a single
# wide QRS complex.
REFRACTORY_MS = 200.0

# How far to search, in the cleaned signal, around a detected integrated
# peak for the true R-peak sample (covers the group delay introduced by the
# derivative + integration stages).
SEARCH_WINDOW_MS = 80.0

# Pan-Tompkins T-wave check: a candidate arriving less than 360 ms after the
# last accepted QRS is too soon to be the next beat at any normal resting
# heart rate (>166 bpm), so it's likely the T wave of the same beat being
# picked up as a second peak. We only reject it if its slope also looks
# T-wave-like (shallower than the preceding QRS).
T_WAVE_RR_MS = 360.0
SLOPE_WINDOW_MS = 50.0


def _qrs_bandpass(signal: np.ndarray, fs: float) -> np.ndarray:
    nyquist = 0.5 * fs
    b, a = butter(2, [QRS_BANDPASS_LOW / nyquist, QRS_BANDPASS_HIGH / nyquist], btype="band")
    return filtfilt(b, a, signal)


def _derivative(signal: np.ndarray, fs: float) -> np.ndarray:
    """Five-point derivative: emphasizes QRS slope, suppresses P/T waves."""
    T = 1.0 / fs
    kernel = np.array([-1, -2, 0, 2, 1]) / (8 * T)
    return np.convolve(signal, kernel, mode="same")


def _moving_window_integrate(signal: np.ndarray, fs: float) -> np.ndarray:
    window = max(1, int(round(INTEGRATION_WINDOW_MS / 1000.0 * fs)))
    return np.convolve(signal, np.ones(window) / window, mode="same")


def _local_slope(slope_signal: np.ndarray, idx: int, window: int) -> float:
    lo = max(0, idx - window)
    hi = min(len(slope_signal), idx + window + 1)
    return float(np.max(slope_signal[lo:hi])) if hi > lo else 0.0


def _adaptive_threshold_peaks(
    integrated: np.ndarray,
    slope_signal: np.ndarray,
    candidate_idx: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Classify candidate peaks in the integrated signal as QRS or noise.

    Implements Pan-Tompkins' dual running-average thresholds (SPKI for
    signal peaks, NPKI for noise peaks) with search-back for missed beats
    and a T-wave slope check, following the update rule from the original
    paper (0.125/0.875 IIR weighting on the peak history).
    """
    if len(candidate_idx) == 0:
        return np.array([], dtype=int)

    peak_vals = integrated[candidate_idx]

    init_window = integrated[: min(len(integrated), 2000)]
    spki = 0.25 * np.max(init_window) if len(init_window) else 0.0
    npki = 0.5 * np.mean(init_window) if len(init_window) else 0.0

    slope_window = max(1, int(round(SLOPE_WINDOW_MS / 1000.0 * fs)))
    t_wave_rr = T_WAVE_RR_MS / 1000.0 * fs

    qrs_idx = []
    rr_history = []
    last_qrs_slope = None

    def threshold1():
        return npki + 0.25 * (spki - npki)

    for idx, val in zip(candidate_idx, peak_vals):
        thr1 = threshold1()

        is_qrs = val > thr1

        # search-back: if the gap since the last accepted beat is much
        # longer than the recent average RR, re-examine skipped candidates
        # with the more permissive threshold2 = 0.5 * threshold1.
        if not is_qrs and qrs_idx and rr_history:
            avg_rr = np.mean(rr_history[-8:])
            gap = idx - qrs_idx[-1]
            if avg_rr > 0 and gap > 1.66 * avg_rr and val > 0.5 * thr1:
                is_qrs = True

        # T-wave check: a peak arriving implausibly soon after the last QRS
        # is only kept if its slope is as steep as that QRS (i.e. it looks
        # like a genuine fast beat, not the T wave of the previous beat).
        if is_qrs and qrs_idx and last_qrs_slope is not None:
            gap = idx - qrs_idx[-1]
            if gap < t_wave_rr:
                candidate_slope = _local_slope(slope_signal, idx, slope_window)
                if candidate_slope < 0.5 * last_qrs_slope:
                    is_qrs = False

        if is_qrs:
            spki = 0.125 * val + 0.875 * spki
            last_qrs_slope = _local_slope(slope_signal, idx, slope_window)
            if qrs_idx:
                rr_history.append(idx - qrs_idx[-1])
            qrs_idx.append(int(idx))
        else:
            npki = 0.125 * val + 0.875 * npki

    return np.array(qrs_idx, dtype=int)


def _refine_to_local_max(signal: np.ndarray, idx: int, window: int) -> int:
    lo = max(0, idx - window)
    hi = min(len(signal), idx + window + 1)
    local = signal[lo:hi]
    if len(local) == 0:
        return idx
    return lo + int(np.argmax(local))


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return sample indices of R peaks in `signal` (Pan-Tompkins)."""
    if len(signal) < 20:
        return np.array([], dtype=int)

    qrs_band = _qrs_bandpass(signal, fs)
    derived = _derivative(qrs_band, fs)
    squared = derived**2
    integrated = _moving_window_integrate(squared, fs)

    min_distance = max(1, int(round(REFRACTORY_MS / 1000.0 * fs)))
    candidate_idx, _ = find_peaks(integrated, distance=min_distance)

    qrs_idx = _adaptive_threshold_peaks(integrated, squared, candidate_idx, fs)

    search_window = max(1, int(round(SEARCH_WINDOW_MS / 1000.0 * fs)))
    refined = np.array(
        [_refine_to_local_max(signal, idx, search_window) for idx in qrs_idx],
        dtype=int,
    )
    return np.unique(refined)


def segment_beats(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    pre_ms: float = 200.0,
    post_ms: float = 400.0,
) -> list[np.ndarray]:
    """Cut fixed-length beat windows around each R peak.

    Default window is 200 ms before / 400 ms after the R peak, wide enough
    to capture the P wave (precedes R by ~120-200 ms) and the T wave
    (follows R by up to ~400 ms) for a typical resting heart rate.
    """
    pre = int(round(pre_ms / 1000.0 * fs))
    post = int(round(post_ms / 1000.0 * fs))

    beats = []
    for peak in r_peaks:
        lo, hi = peak - pre, peak + post
        if lo < 0 or hi > len(signal):
            continue
        beats.append(signal[lo:hi])
    return beats
