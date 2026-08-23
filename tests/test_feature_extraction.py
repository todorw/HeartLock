import numpy as np

from heartlock import feature_extraction as fe


def _synthetic_beat(fs, r_time_s, pr_ms=160, qrs_ms=50, qt_ms=360, t_offset_ms=250):
    n = int(round((fe.BEAT_PRE_MS + fe.BEAT_POST_MS) / 1000.0 * fs))
    t = np.arange(n) / fs
    r_idx = fe._r_offset(fs)
    r_time = r_idx / fs

    beat = np.zeros(n)
    beat += 0.15 * np.exp(-0.5 * ((t - (r_time - pr_ms / 1000.0)) / 0.02) ** 2)
    beat -= 0.08 * np.exp(-0.5 * ((t - (r_time - 0.02)) / 0.008) ** 2)
    beat += 1.0 * np.exp(-0.5 * ((t - r_time) / 0.008) ** 2)
    beat -= 0.2 * np.exp(-0.5 * ((t - (r_time + 0.025)) / 0.008) ** 2)
    beat += 0.3 * np.exp(-0.5 * ((t - (r_time + t_offset_ms / 1000.0)) / 0.04) ** 2)
    return beat


def test_delineate_beat_recovers_approximate_clinical_intervals():
    fs = 500
    beat = _synthetic_beat(fs, r_time_s=fe.BEAT_PRE_MS / 1000.0)
    feats = fe.delineate_beat(beat, fs)

    assert feats is not None
    assert 100 < feats.pr_interval_ms < 220
    assert 20 < feats.qrs_duration_ms < 90
    assert 250 < feats.qt_interval_ms < 420
    assert feats.r_amplitude > feats.p_amplitude > 0
    assert feats.q_amplitude < 0
    assert feats.s_amplitude < 0


def test_delineate_beat_returns_none_for_too_short_beat():
    fs = 500
    tiny_beat = np.zeros(10)
    assert fe.delineate_beat(tiny_beat, fs) is None


def test_feature_vector_length_matches_names():
    fs = 500
    beat = _synthetic_beat(fs, r_time_s=fe.BEAT_PRE_MS / 1000.0)
    feats = fe.delineate_beat(beat, fs)
    vec = feats.to_vector()
    assert len(vec) == len(fe.FiducialFeatures.feature_names())


def test_extract_fiducial_features_handles_flat_signal():
    fs = 500
    flat = np.zeros(5000)
    vec = fe.extract_fiducial_features(flat, fs)
    assert np.all(np.isnan(vec))


def test_build_average_template_none_when_no_beats():
    fs = 500
    flat = np.zeros(5000)
    tmpl = fe.build_average_template(flat, fs)
    assert tmpl is None


def test_build_average_template_shape_and_alignment():
    fs = 500
    r_period = int(0.8 * fs)
    n_beats = 10
    total_len = fe._r_offset(fs) + r_period * n_beats
    signal = np.zeros(total_len)

    beat = _synthetic_beat(fs, r_time_s=fe.BEAT_PRE_MS / 1000.0)
    r_idx = fe._r_offset(fs)
    for i in range(n_beats):
        center = r_idx + i * r_period
        lo = center - r_idx
        hi = lo + len(beat)
        if hi <= len(signal):
            signal[lo:hi] += beat

    tmpl = fe.build_average_template(signal, fs)
    assert tmpl is not None
    expected_len = int(round((fe.BEAT_PRE_MS + fe.BEAT_POST_MS) / 1000.0 * fs))
    assert len(tmpl) == expected_len
    # the template's own R peak should land at the expected R offset
    assert np.argmax(tmpl) == fe._r_offset(fs)
