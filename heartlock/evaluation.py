"""Biometric evaluation: cross-session genuine/impostor scoring, ROC, EER.

Raw accuracy is a poor way to judge a biometric matcher because it hides
the actual decision trade-off: at whatever threshold you'd deploy, how
often does a real match get rejected (FRR) versus how often does an
impostor get accepted (FAR)? The Equal Error Rate (the point where those
two are equal) is the standard single-number summary of that trade-off.

We specifically test *cross-session* identification - enrolling on one
recording session and testing on a different session of the same
subject - because within-session accuracy is inflated by session-specific
noise/electrode-placement artifacts that a real deployment wouldn't get
to rely on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from heartlock import data_loader as dl
from heartlock import matching as mm
from heartlock import preprocessing as pp


@dataclass
class CrossSessionScores:
    genuine: np.ndarray
    impostor: np.ndarray
    rank1_accuracy: float
    n_subjects: int
    n_queries: int
    # every (query subject, candidate subject, score) triple scored during
    # the run, kept alongside the flattened genuine/impostor arrays above
    # so per-subject threshold calibration can re-group them by candidate.
    pair_scores: list[tuple[str, str, float]]


@dataclass
class EvaluationSummary:
    method: str
    enroll_session: int
    test_session: int
    n_subjects: int
    n_queries: int
    n_genuine_scores: int
    n_impostor_scores: int
    rank1_accuracy: float
    eer: float
    eer_threshold: float
    auc: float
    per_subject_thresholds: dict[str, float]


def qualifying_subjects(
    enroll_session: int,
    test_session: int,
    data_dir: Path | None = None,
    max_subjects: int | None = None,
) -> list[str]:
    """Subjects that have recordings for both sessions, so a genuine
    cross-session comparison is actually possible for them."""
    subjects = dl.list_subjects(data_dir)
    qualifying = []
    for sid in subjects:
        sessions = dl.list_sessions(sid, data_dir)
        if enroll_session in sessions and test_session in sessions:
            qualifying.append(sid)
    if max_subjects is not None:
        qualifying = qualifying[:max_subjects]
    return qualifying


def run_cross_session_evaluation(
    subject_ids: list[str],
    enroll_session: int,
    test_session: int,
    method: str = "template_corr",
    data_dir: Path | None = None,
    powerline_freq: float = 50.0,
) -> CrossSessionScores:
    """Enroll every subject on `enroll_session`, identify each subject's
    `test_session` recording against the whole enrolled set, and collect
    genuine (query subject == candidate) vs impostor score pairs.
    """
    enrollments = []
    for sid in subject_ids:
        rec = dl.load_record(sid, enroll_session, data_dir)
        clean = pp.preprocess(rec.signal, rec.fs, powerline_freq)
        try:
            enrollments.append(mm.enroll_subject(clean, rec.fs, sid))
        except ValueError:
            continue  # no beats could be segmented from this recording

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    pair_scores: list[tuple[str, str, float]] = []
    rank1_correct = 0
    rank1_total = 0

    for sid in subject_ids:
        if sid not in {e.subject_id for e in enrollments}:
            continue
        rec = dl.load_record(sid, test_session, data_dir)
        clean = pp.preprocess(rec.signal, rec.fs, powerline_freq)
        result = mm.identify(clean, rec.fs, enrollments, method=method)
        if not result.scores:
            continue

        rank1_total += 1
        if result.subject_id == sid:
            rank1_correct += 1

        for candidate_id, score in result.scores.items():
            if not np.isfinite(score):
                continue
            pair_scores.append((sid, candidate_id, score))
            if candidate_id == sid:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    rank1_accuracy = rank1_correct / rank1_total if rank1_total else float("nan")

    return CrossSessionScores(
        genuine=np.array(genuine_scores),
        impostor=np.array(impostor_scores),
        rank1_accuracy=rank1_accuracy,
        n_subjects=len(enrollments),
        n_queries=rank1_total,
        pair_scores=pair_scores,
    )


def roc_curve(
    genuine: np.ndarray, impostor: np.ndarray, n_thresholds: int = 500
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep a threshold across the observed score range and compute, at
    each point, the False Accept Rate (fraction of impostor scores that
    would be wrongly accepted) and False Reject Rate (fraction of genuine
    scores that would be wrongly rejected).
    """
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("need at least one genuine and one impostor score")

    all_scores = np.concatenate([genuine, impostor])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)

    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine < t).mean() for t in thresholds])
    return thresholds, far, frr


def equal_error_rate(
    thresholds: np.ndarray, far: np.ndarray, frr: np.ndarray
) -> tuple[float, float]:
    """Approximate EER as the point where FAR and FRR are closest.

    far is decreasing and frr is increasing in threshold, so their
    difference crosses zero near the true EER; when the sweep resolution
    doesn't land exactly on a crossing, linear interpolation between the
    two bracketing points is used.
    """
    diff = far - frr
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(sign_changes) == 0:
        idx = int(np.argmin(np.abs(diff)))
        return float((far[idx] + frr[idx]) / 2), float(thresholds[idx])

    i = sign_changes[0]
    d0, d1 = diff[i], diff[i + 1]
    t0, t1 = thresholds[i], thresholds[i + 1]
    f0, f1 = far[i], far[i + 1]

    frac = d0 / (d0 - d1) if d0 != d1 else 0.0
    eer_threshold = t0 + frac * (t1 - t0)
    eer = f0 + frac * (f1 - f0)
    return float(eer), float(eer_threshold)


def compute_auc(far: np.ndarray, frr: np.ndarray) -> float:
    """Area under the ROC curve (TPR vs FPR), via the trapezoid rule."""
    fpr = far[::-1]
    tpr = (1 - frr)[::-1]
    return float(np.trapezoid(tpr, fpr))


MIN_IMPOSTOR_SAMPLES_FOR_NORMALIZATION = 3


def subject_score_stats(
    pair_scores: list[tuple[str, str, float]],
    min_impostor_samples: int = MIN_IMPOSTOR_SAMPLES_FOR_NORMALIZATION,
) -> dict[str, tuple[float, float]]:
    """Per-enrolled-subject (mean, std) of the *impostor* scores scored
    against them - i.e. every other subject's query, excluding their own
    genuine query.

    This is the "cohort"/background distribution a score is normalized
    against (Z-norm, standard in biometric score fusion): some enrolled
    templates just run noisier or score higher against everyone, and a
    single global raw threshold silently over- or under-protects them.
    Subjects with too few impostor probes to estimate a stable std are
    skipped, so callers should fall back to the global threshold for them.
    """
    by_subject: dict[str, list[float]] = {}
    for query_id, candidate_id, score in pair_scores:
        if query_id == candidate_id:
            continue
        by_subject.setdefault(candidate_id, []).append(score)

    stats: dict[str, tuple[float, float]] = {}
    for sid, scores in by_subject.items():
        if len(scores) < min_impostor_samples:
            continue
        arr = np.array(scores)
        std = float(np.std(arr))
        stats[sid] = (float(np.mean(arr)), std if std > 1e-9 else 1.0)
    return stats


def normalized_pair_scores(
    pair_scores: list[tuple[str, str, float]], stats: dict[str, tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Z-normalize every (query, candidate) score against the candidate's
    own impostor-score distribution from `subject_score_stats`, then split
    into genuine/impostor - putting every subject's scores on the same
    scale before a single decision threshold is chosen in that space.
    """
    genuine: list[float] = []
    impostor: list[float] = []
    for query_id, candidate_id, score in pair_scores:
        if candidate_id not in stats:
            continue
        mean, std = stats[candidate_id]
        z = (score - mean) / std
        (genuine if query_id == candidate_id else impostor).append(z)
    return np.array(genuine), np.array(impostor)


def per_subject_thresholds(
    pair_scores: list[tuple[str, str, float]],
    min_impostor_samples: int = MIN_IMPOSTOR_SAMPLES_FOR_NORMALIZATION,
) -> dict[str, float]:
    """One raw-score accept/reject threshold per enrolled subject.

    Finds a single EER threshold in Z-normalized score space (comparable
    across subjects), then maps it back to each subject's own raw score
    scale via that subject's impostor mean/std: threshold_i = mean_i +
    z_threshold * std_i. Subjects without enough impostor probes to
    normalize are simply omitted - callers should fall back to the global
    (non-normalized) EER threshold for them.
    """
    stats = subject_score_stats(pair_scores, min_impostor_samples)
    if not stats:
        return {}

    z_genuine, z_impostor = normalized_pair_scores(pair_scores, stats)
    if len(z_genuine) == 0 or len(z_impostor) == 0:
        return {}

    thresholds, far, frr = roc_curve(z_genuine, z_impostor)
    _, z_threshold = equal_error_rate(thresholds, far, frr)

    return {sid: mean + z_threshold * std for sid, (mean, std) in stats.items()}


def plot_far_frr(
    thresholds: np.ndarray,
    far: np.ndarray,
    frr: np.ndarray,
    eer: float,
    eer_threshold: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(thresholds, far, label="FAR (false accept rate)")
    ax.plot(thresholds, frr, label="FRR (false reject rate)")
    ax.axvline(eer_threshold, color="gray", linestyle="--", linewidth=1)
    ax.plot([eer_threshold], [eer], "ko", label=f"EER = {eer:.3f}")
    ax.set_xlabel("match score threshold")
    ax.set_ylabel("error rate")
    ax.set_title("FAR / FRR vs threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_roc(far: np.ndarray, frr: np.ndarray, auc: float, save_path: Path) -> None:
    fpr = far[::-1]
    tpr = (1 - frr)[::-1]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("false accept rate")
    ax.set_ylabel("true accept rate")
    ax.set_title("ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def run_evaluation(
    method: str = "template_corr",
    enroll_session: int = 1,
    test_session: int = 2,
    max_subjects: int | None = 30,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
) -> EvaluationSummary:
    """End-to-end cross-session evaluation: pick qualifying subjects, run
    identification, compute ROC/EER, and (if `results_dir` is given) save
    plots and a JSON summary.
    """
    subject_ids = qualifying_subjects(enroll_session, test_session, data_dir, max_subjects)
    if not subject_ids:
        raise ValueError(
            f"no subjects have both session {enroll_session} and session {test_session}"
        )

    scores = run_cross_session_evaluation(
        subject_ids, enroll_session, test_session, method, data_dir
    )
    thresholds, far, frr = roc_curve(scores.genuine, scores.impostor)
    eer, eer_threshold = equal_error_rate(thresholds, far, frr)
    auc = compute_auc(far, frr)
    subject_thresholds = per_subject_thresholds(scores.pair_scores)

    summary = EvaluationSummary(
        method=method,
        enroll_session=enroll_session,
        test_session=test_session,
        n_subjects=scores.n_subjects,
        n_queries=scores.n_queries,
        n_genuine_scores=len(scores.genuine),
        n_impostor_scores=len(scores.impostor),
        rank1_accuracy=scores.rank1_accuracy,
        eer=eer,
        eer_threshold=eer_threshold,
        auc=auc,
        per_subject_thresholds=subject_thresholds,
    )

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        plot_far_frr(
            thresholds, far, frr, eer, eer_threshold, results_dir / f"{method}_far_frr.png"
        )
        plot_roc(far, frr, auc, results_dir / f"{method}_roc.png")
        (results_dir / f"{method}_summary.json").write_text(
            json.dumps(asdict(summary), indent=2)
        )

    return summary
