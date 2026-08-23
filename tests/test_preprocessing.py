import numpy as np

from heartlock import preprocessing as pp


def _make_signal(fs, duration_s, freqs_amps, noise_std=0.0, seed=0):
    t = np.arange(0, duration_s, 1 / fs)
    signal = np.zeros_like(t)
    for freq, amp in freqs_amps:
        signal += amp * np.sin(2 * np.pi * freq * t)
    if noise_std:
        rng = np.random.default_rng(seed)
        signal += rng.normal(0, noise_std, size=signal.shape)
    return t, signal


def test_bandpass_attenuates_baseline_wander():
    fs = 500
    # 0.1 Hz baseline wander well below the 0.5 Hz cutoff, plus 10 Hz
    # "cardiac-like" content inside the passband.
    _, signal = _make_signal(fs, 10, [(0.1, 5.0), (10, 1.0)])
    filtered = pp.bandpass_filter(signal, fs)

    baseline_power_before = np.abs(np.fft.rfft(signal))
    baseline_power_after = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(signal), 1 / fs)

    idx_baseline = np.argmin(np.abs(freqs - 0.1))
    idx_passband = np.argmin(np.abs(freqs - 10))

    assert baseline_power_after[idx_baseline] < 0.1 * baseline_power_before[idx_baseline]
    # passband content should survive with most of its amplitude
    assert baseline_power_after[idx_passband] > 0.5 * baseline_power_before[idx_passband]


def test_bandpass_attenuates_high_frequency_noise():
    fs = 500
    _, signal = _make_signal(fs, 10, [(10, 1.0), (100, 2.0)])
    filtered = pp.bandpass_filter(signal, fs)

    freqs = np.fft.rfftfreq(len(signal), 1 / fs)
    power_before = np.abs(np.fft.rfft(signal))
    power_after = np.abs(np.fft.rfft(filtered))

    idx_hf = np.argmin(np.abs(freqs - 100))
    assert power_after[idx_hf] < 0.1 * power_before[idx_hf]


def test_notch_filter_attenuates_powerline_frequency():
    fs = 500
    _, signal = _make_signal(fs, 10, [(10, 1.0), (50, 3.0)])
    filtered = pp.notch_filter(signal, fs, freq=50.0)

    freqs = np.fft.rfftfreq(len(signal), 1 / fs)
    power_before = np.abs(np.fft.rfft(signal))
    power_after = np.abs(np.fft.rfft(filtered))

    idx_50 = np.argmin(np.abs(freqs - 50))
    idx_10 = np.argmin(np.abs(freqs - 10))

    assert power_after[idx_50] < 0.1 * power_before[idx_50]
    assert power_after[idx_10] > 0.7 * power_before[idx_10]


def test_normalize_zero_mean_unit_variance():
    rng = np.random.default_rng(1)
    signal = rng.normal(5, 2, size=1000)
    normed = pp.normalize(signal)
    assert np.isclose(np.mean(normed), 0, atol=1e-8)
    assert np.isclose(np.std(normed), 1, atol=1e-8)


def test_normalize_handles_constant_signal():
    signal = np.full(100, 3.0)
    normed = pp.normalize(signal)
    assert np.allclose(normed, 0)


def test_preprocess_pipeline_runs_and_normalizes():
    fs = 500
    _, signal = _make_signal(fs, 10, [(0.1, 5.0), (10, 1.0), (50, 2.0)], noise_std=0.1)
    out = pp.preprocess(signal, fs)
    assert out.shape == signal.shape
    assert np.isclose(np.mean(out), 0, atol=1e-6)
