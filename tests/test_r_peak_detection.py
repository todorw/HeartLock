import numpy as np

from heartlock import r_peak_detection as rpd


def _synthetic_ecg(fs, duration_s, heart_rate_bpm, r_amplitude=1.5, noise_std=0.02, seed=0):
    """Build a crude but deterministic ECG-like signal: periodic sharp
    R-peaks (narrow gaussian) plus a broader T-wave gaussian per beat, with
    light noise. Not physiologically exact, just peaky enough to exercise
    the detector's timing accuracy.
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    signal = np.zeros(n)

    rr = 60.0 / heart_rate_bpm
    beat_times = np.arange(0.3, duration_s - 0.3, rr)

    for bt in beat_times:
        signal += r_amplitude * np.exp(-0.5 * ((t - bt) / 0.015) ** 2)
        signal += 0.3 * r_amplitude * np.exp(-0.5 * ((t - (bt + 0.25)) / 0.08) ** 2)

    rng = np.random.default_rng(seed)
    signal += rng.normal(0, noise_std, size=n)
    return signal, beat_times


def test_detect_r_peaks_matches_known_heart_rate():
    fs = 500
    signal, beat_times = _synthetic_ecg(fs, duration_s=20, heart_rate_bpm=72)
    peaks = rpd.detect_r_peaks(signal, fs)

    assert abs(len(peaks) - len(beat_times)) <= 1

    detected_times = peaks / fs
    for bt in beat_times[1:-1]:
        closest = np.min(np.abs(detected_times - bt))
        assert closest < 0.03  # within 30 ms of the true R peak


def test_detect_r_peaks_handles_fast_heart_rate_without_double_counting():
    fs = 500
    signal, beat_times = _synthetic_ecg(fs, duration_s=15, heart_rate_bpm=150)
    peaks = rpd.detect_r_peaks(signal, fs)

    assert abs(len(peaks) - len(beat_times)) <= 2


def test_detect_r_peaks_empty_signal():
    peaks = rpd.detect_r_peaks(np.zeros(0), fs=500)
    assert len(peaks) == 0


def test_segment_beats_returns_expected_window_length():
    fs = 500
    signal, _ = _synthetic_ecg(fs, duration_s=20, heart_rate_bpm=72)
    peaks = rpd.detect_r_peaks(signal, fs)
    beats = rpd.segment_beats(signal, peaks, fs, pre_ms=200, post_ms=400)

    assert len(beats) > 0
    expected_len = int(round(200 / 1000 * fs)) + int(round(400 / 1000 * fs))
    for beat in beats:
        assert len(beat) == expected_len


def test_segment_beats_drops_peaks_too_close_to_edges():
    fs = 500
    signal = np.zeros(100)
    r_peaks = np.array([5, 50, 95])
    beats = rpd.segment_beats(signal, r_peaks, fs, pre_ms=200, post_ms=400)
    # all three peaks are too close to an edge for a full 300-sample window
    assert len(beats) == 0
