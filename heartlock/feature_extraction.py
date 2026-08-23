"""Turn segmented beats into per-subject biometric features.

Two independent approaches, so they can be compared for identification
accuracy:

- Fiducial: locate the P, Q, R, S, T wave landmarks on each beat and
  measure clinically-meaningful durations/amplitudes/intervals from them.
  This is the classic, fully interpretable ECG feature set.
- Non-fiducial: skip landmark detection entirely and just take the
  median waveform across a subject's beats as a "template" to be compared
  directly (via correlation/DTW in `matching.py`). More robust to
  delineation errors, less interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from heartlock.r_peak_detection import detect_r_peaks, segment_beats

# Wider than r_peak_detection's default segmentation window: fiducial
# delineation needs enough pre-R margin to reliably capture a full P wave
# (P peaks are typically 120-250 ms ahead of R) and enough post-R margin
# for the T wave to fully return to baseline (up to ~400 ms after R).
BEAT_PRE_MS = 300.0
BEAT_POST_MS = 450.0

# Search windows for each landmark, relative to the R peak, in ms.
Q_SEARCH_MS = 60.0
S_SEARCH_MS = 80.0
T_SEARCH_GAP_MS = 40.0

# First N ms of the beat window is assumed to be isoelectric baseline
# (comfortably before the P wave given the BEAT_PRE_MS margin above).
BASELINE_MS = 40.0

ONSET_OFFSET_THRESHOLD_FRAC = 0.1


@dataclass
class FiducialFeatures:
    """One beat's landmark-based measurements, in ms (durations/intervals)
    or the beat's own normalized signal units (amplitudes)."""

    p_amplitude: float
    q_amplitude: float
    r_amplitude: float
    s_amplitude: float
    t_amplitude: float
    p_duration_ms: float
    qrs_duration_ms: float
    t_duration_ms: float
    pr_interval_ms: float
    qt_interval_ms: float

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        return [f.name for f in fields(FiducialFeatures)]


def _r_offset(fs: float) -> int:
    return int(round(BEAT_PRE_MS / 1000.0 * fs))


def _samples(ms: float, fs: float) -> int:
    return max(1, int(round(ms / 1000.0 * fs)))


def _find_onset_offset(
    beat: np.ndarray,
    peak_idx: int,
    baseline: float,
    lo: int,
    hi: int,
    threshold_frac: float = ONSET_OFFSET_THRESHOLD_FRAC,
) -> tuple[int, int]:
    """Walk outward from a wave's peak until the signal drops back within
    `threshold_frac` of its amplitude above baseline, in either direction.
    """
    amplitude = beat[peak_idx] - baseline
    threshold = baseline + threshold_frac * amplitude

    onset = peak_idx
    if amplitude >= 0:
        while onset > lo and beat[onset] > threshold:
            onset -= 1
    else:
        while onset > lo and beat[onset] < threshold:
            onset -= 1

    offset = peak_idx
    if amplitude >= 0:
        while offset < hi - 1 and beat[offset] > threshold:
            offset += 1
    else:
        while offset < hi - 1 and beat[offset] < threshold:
            offset += 1

    return onset, offset


def delineate_beat(beat: np.ndarray, fs: float) -> FiducialFeatures | None:
    """Locate P/Q/R/S/T landmarks on one R-aligned beat and measure them.

    This is a lightweight heuristic delineator (peak search in
    physiologically-motivated windows + threshold-crossing onset/offset),
    not a validated clinical-grade algorithm - adequate for biometric
    feature comparison but not for diagnostic use.
    """
    r_idx = _r_offset(fs)
    if r_idx >= len(beat):
        return None

    baseline_n = _samples(BASELINE_MS, fs)
    baseline = float(np.median(beat[:baseline_n]))

    q_window = _samples(Q_SEARCH_MS, fs)
    q_lo = max(0, r_idx - q_window)
    if q_lo >= r_idx:
        return None
    q_idx = q_lo + int(np.argmin(beat[q_lo:r_idx]))

    s_window = _samples(S_SEARCH_MS, fs)
    s_hi = min(len(beat), r_idx + s_window)
    if s_hi <= r_idx + 1:
        return None
    s_idx = r_idx + 1 + int(np.argmin(beat[r_idx + 1 : s_hi]))

    if q_idx < baseline_n:
        return None
    p_idx = int(np.argmax(beat[baseline_n:q_idx])) + baseline_n

    t_gap = _samples(T_SEARCH_GAP_MS, fs)
    t_lo = min(len(beat) - 1, s_idx + t_gap)
    if t_lo >= len(beat) - 1:
        return None
    t_idx = t_lo + int(np.argmax(beat[t_lo:]))

    p_onset, p_offset = _find_onset_offset(beat, p_idx, baseline, baseline_n, q_idx)
    t_onset, t_offset = _find_onset_offset(beat, t_idx, baseline, s_idx, len(beat))

    dt_ms = 1000.0 / fs

    return FiducialFeatures(
        p_amplitude=float(beat[p_idx] - baseline),
        q_amplitude=float(beat[q_idx] - baseline),
        r_amplitude=float(beat[r_idx] - baseline),
        s_amplitude=float(beat[s_idx] - baseline),
        t_amplitude=float(beat[t_idx] - baseline),
        p_duration_ms=float((p_offset - p_onset) * dt_ms),
        qrs_duration_ms=float((s_idx - q_idx) * dt_ms),
        t_duration_ms=float((t_offset - t_onset) * dt_ms),
        pr_interval_ms=float((q_idx - p_onset) * dt_ms),
        qt_interval_ms=float((t_offset - q_idx) * dt_ms),
    )


def extract_fiducial_features(signal: np.ndarray, fs: float) -> np.ndarray:
    """R-detect, segment, delineate every beat, and return the per-subject
    feature vector as the median across beats (robust to occasional
    delineation failures on noisy individual beats).
    """
    r_peaks = detect_r_peaks(signal, fs)
    beats = segment_beats(signal, r_peaks, fs, pre_ms=BEAT_PRE_MS, post_ms=BEAT_POST_MS)

    per_beat = [delineate_beat(b, fs) for b in beats]
    vectors = [f.to_vector() for f in per_beat if f is not None]

    if not vectors:
        return np.full(len(FiducialFeatures.feature_names()), np.nan)

    return np.median(np.stack(vectors), axis=0)


def build_average_template(signal: np.ndarray, fs: float) -> np.ndarray | None:
    """Non-fiducial feature: the median R-aligned beat waveform.

    Median (not mean) so a handful of noisy/misaligned beats don't drag the
    template shape away from the typical beat.
    """
    r_peaks = detect_r_peaks(signal, fs)
    beats = segment_beats(signal, r_peaks, fs, pre_ms=BEAT_PRE_MS, post_ms=BEAT_POST_MS)

    if not beats:
        return None

    return np.median(np.stack(beats), axis=0)
