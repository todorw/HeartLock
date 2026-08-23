"""Enroll subjects and identify a query ECG snippet against them.

All scoring functions here return "higher is better" scores so
`identify()` and the evaluation module can treat every method uniformly:
argmax over candidates picks the best match, and the winning score itself
doubles as the confidence value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from heartlock.feature_extraction import build_average_template, extract_fiducial_features

# Small lag window (in samples) searched when correlating two templates,
# so a few samples of residual R-alignment jitter between recordings
# doesn't tank an otherwise good match.
CORR_MAX_LAG = 10


@dataclass
class Enrollment:
    subject_id: str
    fiducial_vector: np.ndarray
    template: np.ndarray


@dataclass
class IdentificationResult:
    subject_id: str | None
    confidence: float
    scores: dict[str, float]


def enroll_subject(signal: np.ndarray, fs: float, subject_id: str) -> Enrollment:
    """Extract both feature representations for one subject's recording."""
    fiducial_vector = extract_fiducial_features(signal, fs)
    template = build_average_template(signal, fs)
    if template is None:
        raise ValueError(f"no beats could be segmented for subject {subject_id!r}")
    return Enrollment(subject_id=subject_id, fiducial_vector=fiducial_vector, template=template)


def save_enrollments(enrollments: list[Enrollment], path: Path) -> None:
    """Persist enrollments as a plain numpy archive (no pickle involved)."""
    arrays = {}
    subject_order = []
    for e in enrollments:
        arrays[f"{e.subject_id}__fiducial"] = e.fiducial_vector
        arrays[f"{e.subject_id}__template"] = e.template
        subject_order.append(e.subject_id)
    arrays["__subject_order"] = np.array(subject_order)
    np.savez(path, **arrays)


def load_enrollments(path: Path) -> list[Enrollment]:
    with np.load(path, allow_pickle=False) as data:
        subject_order = [str(s) for s in data["__subject_order"]]
        return [
            Enrollment(
                subject_id=sid,
                fiducial_vector=data[f"{sid}__fiducial"],
                template=data[f"{sid}__template"],
            )
            for sid in subject_order
        ]


def template_correlation_score(query: np.ndarray, template: np.ndarray) -> float:
    """Max Pearson correlation between the two templates over a small lag
    window. Templates are already R-aligned by construction, so only a
    small search is needed to absorb residual jitter.
    """
    n = min(len(query), len(template))
    q = query[:n]
    t = template[:n]

    best = -1.0
    for lag in range(-CORR_MAX_LAG, CORR_MAX_LAG + 1):
        if lag >= 0:
            a, b = q[lag:], t[: n - lag]
        else:
            a, b = q[: n + lag], t[-lag:]
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        best = max(best, corr)
    return best


def template_dtw_score(query: np.ndarray, template: np.ndarray) -> float:
    """DTW alignment distance, converted to a bounded higher-is-better score.

    fastdtw distance is unbounded and grows with beat length, so it's
    normalized by path length before being folded into 1/(1+d).
    """
    # fastdtw compares individual samples (scalars) for a 1-D series;
    # scipy's euclidean() expects vectors, so a plain abs-difference is used.
    distance, path = fastdtw(query, template, dist=lambda a, b: abs(a - b))
    normalized = distance / max(1, len(path))
    return 1.0 / (1.0 + normalized)


def fiducial_feature_stats(enrollments: list[Enrollment]) -> tuple[np.ndarray, np.ndarray]:
    """Population mean/std per fiducial feature, for z-scoring distances
    so features on very different scales (ms vs mV) contribute comparably.
    """
    stacked = np.stack([e.fiducial_vector for e in enrollments])
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    std[std == 0] = 1.0
    return mean, std


def fiducial_score(
    query_vector: np.ndarray, enrolled_vector: np.ndarray, feature_std: np.ndarray
) -> float:
    """Negative z-scored euclidean distance (higher = more similar)."""
    if np.any(np.isnan(query_vector)) or np.any(np.isnan(enrolled_vector)):
        return -np.inf
    z_query = query_vector / feature_std
    z_enrolled = enrolled_vector / feature_std
    return -float(euclidean(z_query, z_enrolled))


def identify(
    query_signal: np.ndarray,
    fs: float,
    enrollments: list[Enrollment],
    method: str = "template_corr",
) -> IdentificationResult:
    """Compare a query ECG snippet against every enrolled subject.

    method: "template_corr" (cross-correlation of averaged beat template,
    default and generally most robust for short snippets), "template_dtw"
    (DTW alignment of the same template), or "fiducial" (landmark-based
    feature distance).
    """
    if not enrollments:
        return IdentificationResult(subject_id=None, confidence=0.0, scores={})

    query_template = build_average_template(query_signal, fs)
    scores: dict[str, float] = {}

    if method == "template_corr":
        for e in enrollments:
            scores[e.subject_id] = (
                template_correlation_score(query_template, e.template)
                if query_template is not None
                else -1.0
            )
    elif method == "template_dtw":
        for e in enrollments:
            scores[e.subject_id] = (
                template_dtw_score(query_template, e.template) if query_template is not None else 0.0
            )
    elif method == "fiducial":
        query_vector = extract_fiducial_features(query_signal, fs)
        _, feature_std = fiducial_feature_stats(enrollments)
        for e in enrollments:
            scores[e.subject_id] = fiducial_score(query_vector, e.fiducial_vector, feature_std)
    else:
        raise ValueError(f"unknown matching method {method!r}")

    best_subject = max(scores, key=scores.get)
    return IdentificationResult(
        subject_id=best_subject, confidence=scores[best_subject], scores=scores
    )
