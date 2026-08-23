import numpy as np
import pytest

from heartlock import matching as mm


def _gaussian_template(n, peak_idx, width=8, amplitude=1.0, seed=None):
    t = np.arange(n)
    template = amplitude * np.exp(-0.5 * ((t - peak_idx) / width) ** 2)
    if seed is not None:
        rng = np.random.default_rng(seed)
        template = template + rng.normal(0, 0.02, size=n)
    return template


def _synthetic_multi_beat_signal(
    fs, duration_s, heart_rate_bpm, r_width_ms=15, t_amplitude=0.3, noise_std=0.01, seed=0
):
    """Realistic-enough multi-beat ECG signal (sharp R spike + broad T
    bump, repeated at a fixed rate), so it round-trips correctly through
    the real R-detection + segmentation pipeline `identify()` uses
    internally, unlike a single bare template array.
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    signal = np.zeros(n)

    rr = 60.0 / heart_rate_bpm
    beat_times = np.arange(0.5, duration_s - 0.5, rr)

    for bt in beat_times:
        signal += 1.5 * np.exp(-0.5 * ((t - bt) / (r_width_ms / 1000.0)) ** 2)
        signal += t_amplitude * np.exp(-0.5 * ((t - (bt + 0.25)) / 0.08) ** 2)

    rng = np.random.default_rng(seed)
    signal += rng.normal(0, noise_std, size=n)
    return signal


@pytest.fixture
def synthetic_enrollments():
    n = 375
    r_idx = 150
    templates = {
        "Person_A": _gaussian_template(n, r_idx, width=8, amplitude=1.0),
        "Person_B": _gaussian_template(n, r_idx, width=20, amplitude=0.6),
        "Person_C": _gaussian_template(n, r_idx - 5, width=12, amplitude=1.3),
    }
    fiducial = {
        "Person_A": np.array([0.2, -0.1, 1.0, -0.3, 0.4, 80, 90, 180, 160, 380]),
        "Person_B": np.array([0.5, -0.2, 0.8, -0.5, 0.6, 100, 110, 200, 190, 420]),
        "Person_C": np.array([0.1, -0.05, 1.2, -0.2, 0.3, 70, 80, 160, 140, 350]),
    }
    return [
        mm.Enrollment(subject_id=sid, fiducial_vector=fiducial[sid], template=templates[sid])
        for sid in templates
    ], templates


def test_template_correlation_perfect_match_scores_near_one():
    n = 375
    template = _gaussian_template(n, 150)
    score = mm.template_correlation_score(template, template)
    assert score > 0.999


def test_template_correlation_dissimilar_shapes_score_lower():
    n = 375
    narrow = _gaussian_template(n, 150, width=5)
    wide = _gaussian_template(n, 150, width=40)
    same = mm.template_correlation_score(narrow, narrow)
    different = mm.template_correlation_score(narrow, wide)
    assert same > different


def test_template_dtw_score_perfect_match_is_high():
    n = 375
    template = _gaussian_template(n, 150)
    score = mm.template_dtw_score(template, template)
    assert score > 0.99


def test_fiducial_score_identical_vectors_is_zero():
    vec = np.array([1.0, 2.0, 3.0])
    std = np.array([1.0, 1.0, 1.0])
    assert mm.fiducial_score(vec, vec, std) == 0.0


def test_fiducial_score_nan_query_returns_neg_inf():
    vec = np.array([np.nan, 2.0, 3.0])
    other = np.array([1.0, 2.0, 3.0])
    std = np.array([1.0, 1.0, 1.0])
    assert mm.fiducial_score(vec, other, std) == -np.inf


@pytest.fixture
def enrolled_synthetic_subjects():
    """Two subjects enrolled via the real `enroll_subject` pipeline (raw
    signal -> R-detect -> segment -> template/fiducial vector), each with
    a distinct T-wave amplitude so their beat morphology differs.
    """
    fs = 500
    subjects = {
        "Person_Low_T": _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.15, seed=1),
        "Person_High_T": _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=2),
    }
    enrollments = [mm.enroll_subject(sig, fs, sid) for sid, sig in subjects.items()]
    return enrollments, fs


def test_identify_template_corr_picks_correct_subject(enrolled_synthetic_subjects):
    enrollments, fs = enrolled_synthetic_subjects
    # different random seed = a different "session" for the same subject
    query = _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=99)
    result = mm.identify(query, fs=fs, enrollments=enrollments, method="template_corr")
    assert result.subject_id == "Person_High_T"
    assert set(result.scores) == {"Person_Low_T", "Person_High_T"}


def test_identify_template_dtw_picks_correct_subject(enrolled_synthetic_subjects):
    enrollments, fs = enrolled_synthetic_subjects
    query = _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.15, seed=99)
    result = mm.identify(query, fs=fs, enrollments=enrollments, method="template_dtw")
    assert result.subject_id == "Person_Low_T"


def test_identify_fiducial_picks_correct_subject(enrolled_synthetic_subjects):
    enrollments, fs = enrolled_synthetic_subjects
    query = _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=99)
    result = mm.identify(query, fs=fs, enrollments=enrollments, method="fiducial")
    assert result.subject_id == "Person_High_T"


def test_identify_empty_enrollments_returns_none():
    result = mm.identify(np.zeros(100), fs=500, enrollments=[], method="template_corr")
    assert result.subject_id is None
    assert result.confidence == 0.0


def test_identify_unknown_method_raises(synthetic_enrollments):
    enrollments, templates = synthetic_enrollments
    with pytest.raises(ValueError):
        mm.identify(templates["Person_A"], fs=500, enrollments=enrollments, method="bogus")


def test_save_and_load_enrollments_roundtrip(tmp_path, synthetic_enrollments):
    enrollments, _ = synthetic_enrollments
    path = tmp_path / "enrollments.npz"
    mm.save_enrollments(enrollments, path)
    loaded = mm.load_enrollments(path)

    assert [e.subject_id for e in loaded] == [e.subject_id for e in enrollments]
    for original, reloaded in zip(enrollments, loaded):
        np.testing.assert_array_equal(original.template, reloaded.template)
        np.testing.assert_array_equal(original.fiducial_vector, reloaded.fiducial_vector)


def test_fiducial_feature_stats_shapes(synthetic_enrollments):
    enrollments, _ = synthetic_enrollments
    mean, std = mm.fiducial_feature_stats(enrollments)
    assert mean.shape == (10,)
    assert std.shape == (10,)
    assert np.all(std > 0)


def test_verify_accepts_genuine_claim(enrolled_synthetic_subjects):
    enrollments, fs = enrolled_synthetic_subjects
    query = _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=99)
    result = mm.verify(query, fs=fs, claimed_id="Person_High_T", enrollments=enrollments, threshold=0.95)
    assert result.accepted
    assert result.claimed_id == "Person_High_T"
    assert result.score >= result.threshold


def test_verify_rejects_impostor_claim(enrolled_synthetic_subjects):
    enrollments, fs = enrolled_synthetic_subjects
    query = _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=99)
    result = mm.verify(query, fs=fs, claimed_id="Person_Low_T", enrollments=enrollments, threshold=0.95)
    assert not result.accepted
    assert result.score < result.threshold


def test_verify_unknown_claimed_id_raises(enrolled_synthetic_subjects):
    enrollments, fs = enrolled_synthetic_subjects
    query = _synthetic_multi_beat_signal(fs, 20, 72, seed=99)
    with pytest.raises(ValueError, match="no enrollment found"):
        mm.verify(query, fs=fs, claimed_id="Person_Nobody", enrollments=enrollments, threshold=0.5)


def test_verify_unknown_method_raises(enrolled_synthetic_subjects):
    enrollments, fs = enrolled_synthetic_subjects
    query = _synthetic_multi_beat_signal(fs, 20, 72, seed=99)
    with pytest.raises(ValueError, match="unknown matching method"):
        mm.verify(
            query, fs=fs, claimed_id="Person_High_T", enrollments=enrollments, threshold=0.5, method="bogus"
        )
