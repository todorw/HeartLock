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
