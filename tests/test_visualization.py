import numpy as np

from heartlock import visualization as viz


def _synthetic_multi_beat_signal(fs, duration_s, heart_rate_bpm, t_amplitude=0.3, seed=0):
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    signal = np.zeros(n)
    rr = 60.0 / heart_rate_bpm
    for bt in np.arange(0.5, duration_s - 0.5, rr):
        signal += 1.5 * np.exp(-0.5 * ((t - bt) / 0.015) ** 2)
        signal += t_amplitude * np.exp(-0.5 * ((t - (bt + 0.25)) / 0.08) ** 2)
    rng = np.random.default_rng(seed)
    signal += rng.normal(0, 0.01, size=n)
    return signal


def test_plot_pipeline_writes_image_file(tmp_path):
    fs = 500
    signal = _synthetic_multi_beat_signal(fs, 20, 72, seed=1)
    out_path = tmp_path / "pipeline.png"

    viz.plot_pipeline(signal, fs, "test_subject", out_path)

    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_plot_pipeline_handles_no_beats_gracefully(tmp_path):
    fs = 500
    flat_signal = np.zeros(fs * 5)
    out_path = tmp_path / "pipeline_flat.png"

    viz.plot_pipeline(flat_signal, fs, "flat_subject", out_path)

    assert out_path.is_file()
