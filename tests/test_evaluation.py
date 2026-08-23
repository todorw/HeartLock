import numpy as np
import pytest

from heartlock import evaluation as ev


def test_roc_curve_perfectly_separated_scores_gives_zero_eer():
    genuine = np.full(50, 0.9)
    impostor = np.full(50, 0.1)
    thresholds, far, frr = ev.roc_curve(genuine, impostor, n_thresholds=500)
    eer, eer_threshold = ev.equal_error_rate(thresholds, far, frr)
    assert eer < 0.01
    assert 0.1 < eer_threshold < 0.9


def test_roc_curve_identical_distributions_gives_high_eer():
    rng = np.random.default_rng(0)
    genuine = rng.normal(0.5, 0.1, 500)
    impostor = rng.normal(0.5, 0.1, 500)
    thresholds, far, frr = ev.roc_curve(genuine, impostor, n_thresholds=500)
    eer, _ = ev.equal_error_rate(thresholds, far, frr)
    assert eer > 0.3


def test_roc_curve_requires_both_score_types():
    with pytest.raises(ValueError):
        ev.roc_curve(np.array([]), np.array([0.5]))
    with pytest.raises(ValueError):
        ev.roc_curve(np.array([0.5]), np.array([]))


def test_far_decreases_and_frr_increases_with_threshold():
    rng = np.random.default_rng(1)
    genuine = rng.normal(0.8, 0.05, 200)
    impostor = rng.normal(0.3, 0.05, 200)
    thresholds, far, frr = ev.roc_curve(genuine, impostor)
    assert far[0] >= far[-1]
    assert frr[0] <= frr[-1]


def test_compute_auc_perfect_separation_is_one():
    genuine = np.full(50, 0.9)
    impostor = np.full(50, 0.1)
    _, far, frr = ev.roc_curve(genuine, impostor)
    auc = ev.compute_auc(far, frr)
    assert auc > 0.99


def test_compute_auc_random_scores_near_half():
    rng = np.random.default_rng(2)
    genuine = rng.normal(0.5, 0.15, 1000)
    impostor = rng.normal(0.5, 0.15, 1000)
    _, far, frr = ev.roc_curve(genuine, impostor)
    auc = ev.compute_auc(far, frr)
    assert 0.35 < auc < 0.65


def test_qualifying_subjects_respects_max_subjects(tmp_path, monkeypatch):
    import heartlock.data_loader as dl

    monkeypatch.setattr(dl, "list_subjects", lambda data_dir=None: ["A", "B", "C"])
    monkeypatch.setattr(
        dl,
        "list_sessions",
        lambda sid, data_dir=None: [1, 2] if sid != "C" else [1],
    )

    result = ev.qualifying_subjects(1, 2, data_dir=tmp_path, max_subjects=1)
    assert result == ["A"]

    result_all = ev.qualifying_subjects(1, 2, data_dir=tmp_path)
    assert result_all == ["A", "B"]


def _pair_scores_two_subjects():
    """4 subjects enrolled; A and B have noticeably noisier/higher impostor
    scores than C and D, so per-subject normalization should matter.
    """
    pair_scores = []
    # genuine scores, one per subject
    pair_scores += [("A", "A", 0.95), ("B", "B", 0.93), ("C", "C", 0.9), ("D", "D", 0.88)]
    # A and B run "hot": their impostor scores hover close to their genuine
    for other in ["B", "C", "D"]:
        pair_scores.append((other, "A", 0.8))
    for other in ["A", "C", "D"]:
        pair_scores.append((other, "B", 0.78))
    # C and D run "cool": clearly separated impostor scores
    for other in ["A", "B", "D"]:
        pair_scores.append((other, "C", 0.3))
    for other in ["A", "B", "C"]:
        pair_scores.append((other, "D", 0.28))
    return pair_scores


def test_subject_score_stats_excludes_genuine_scores():
    pair_scores = _pair_scores_two_subjects()
    stats = ev.subject_score_stats(pair_scores, min_impostor_samples=3)
    assert set(stats) == {"A", "B", "C", "D"}
    mean_a, std_a = stats["A"]
    assert mean_a == pytest.approx(0.8, abs=1e-9)
    assert std_a == pytest.approx(0.0, abs=1e-6) or std_a == 1.0  # constant scores -> std floored to 1.0


def test_subject_score_stats_skips_sparse_subjects():
    pair_scores = [("A", "A", 0.9), ("B", "A", 0.5)]  # only 1 impostor probe
    stats = ev.subject_score_stats(pair_scores, min_impostor_samples=3)
    assert stats == {}


def test_per_subject_thresholds_differ_by_subject_noise_level():
    pair_scores = _pair_scores_two_subjects()
    thresholds = ev.per_subject_thresholds(pair_scores)
    assert set(thresholds) == {"A", "B", "C", "D"}
    # A/B's impostor scores sit right at their genuine score (0.8 vs 0.95,
    # 0.78 vs 0.93) so their calibrated threshold should land well above
    # C/D's, whose impostors are far below their genuine score.
    assert thresholds["A"] > thresholds["C"]
    assert thresholds["B"] > thresholds["D"]


def test_per_subject_thresholds_empty_without_enough_data():
    pair_scores = [("A", "A", 0.9), ("B", "A", 0.5)]
    assert ev.per_subject_thresholds(pair_scores) == {}


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


def test_run_cross_session_evaluation_reports_misclassifications(monkeypatch):
    import heartlock.data_loader as dl
    from heartlock.data_loader import ECGRecord

    fs = 500
    signal_a = _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.15, seed=1)
    signal_b = _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=2)

    def fake_load_record(subject_id, session, data_dir=None):
        if session == 1:
            signal = signal_a if subject_id == "A" else signal_b
        else:
            # both subjects' "session 2" recording is actually B's signal,
            # so A's query is guaranteed to be misidentified as B while
            # B's own query is still correctly identified as B.
            signal = signal_b
        return ECGRecord(subject_id=subject_id, session=session, signal=signal, fs=fs)

    monkeypatch.setattr(dl, "load_record", fake_load_record)

    scores = ev.run_cross_session_evaluation(
        ["A", "B"], enroll_session=1, test_session=2, method="template_corr"
    )

    assert scores.misclassifications == [("A", "B")]
    assert scores.rank1_accuracy == pytest.approx(0.5)


def test_run_cross_session_evaluation_no_misclassifications_when_all_correct(monkeypatch):
    import heartlock.data_loader as dl
    from heartlock.data_loader import ECGRecord

    fs = 500
    signal_a = _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.15, seed=1)
    signal_b = _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=2)
    signals = {"A": signal_a, "B": signal_b}

    def fake_load_record(subject_id, session, data_dir=None):
        return ECGRecord(subject_id=subject_id, session=session, signal=signals[subject_id], fs=fs)

    monkeypatch.setattr(dl, "load_record", fake_load_record)

    scores = ev.run_cross_session_evaluation(
        ["A", "B"], enroll_session=1, test_session=2, method="template_corr"
    )

    assert scores.misclassifications == []
    assert scores.rank1_accuracy == pytest.approx(1.0)


@pytest.mark.network
def test_run_evaluation_end_to_end(tmp_path):
    summary = ev.run_evaluation(
        method="template_corr",
        enroll_session=1,
        test_session=2,
        max_subjects=5,
        data_dir=tmp_path,
        results_dir=tmp_path / "results",
    )
    assert summary.n_subjects > 0
    assert 0.0 <= summary.eer <= 1.0
    assert (tmp_path / "results" / "template_corr_summary.json").exists()
    assert (tmp_path / "results" / "template_corr_roc.png").exists()
