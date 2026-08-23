"""Fetch and cache ECG-ID Database records from PhysioNet.

We keep the raw ("ECG I") lead rather than PhysioNet's pre-filtered
("ECG I filtered") lead, since HeartLock does its own bandpass/notch
filtering in `preprocessing.py` and we don't want to double-filter or
depend on PhysioNet's unknown filter parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

PN_DB_NAME = "ecgiddb"
RAW_LEAD_NAME = "ECG I"

# Person_01 .. Person_90, rec_1 .. rec_20 (bounds seen in the live database).
_SUBJECT_RE = re.compile(r"^Person_(\d{2})$")
_MAX_SESSION = 50


@dataclass(frozen=True)
class ECGRecord:
    """A single ECG-ID recording: one subject, one session."""

    subject_id: str
    session: int
    signal: np.ndarray
    fs: int
    age: str | None = None
    sex: str | None = None


def default_data_dir() -> Path:
    """Project-relative `data/` directory, kept out of git via .gitignore."""
    return Path(__file__).resolve().parent.parent / "data"


def _validate_subject_id(subject_id: str) -> str:
    """Reject anything that isn't a bare `Person_NN` token.

    This is the only place a caller-supplied string reaches disk/network
    path construction, so it doubles as our path-traversal guard: no
    slashes, dots, or other path metacharacters can survive this check.
    """
    if not _SUBJECT_RE.match(subject_id):
        raise ValueError(
            f"invalid subject id {subject_id!r}, expected format 'Person_01'"
        )
    return subject_id


def _validate_session(session: int) -> int:
    session = int(session)
    if not (1 <= session <= _MAX_SESSION):
        raise ValueError(f"invalid session number {session!r}")
    return session


def list_subjects(data_dir: Path | None = None) -> list[str]:
    """Return sorted subject ids (e.g. ['Person_01', ..., 'Person_90']).

    Queries PhysioNet's record index and caches it locally so repeat calls
    don't require network access.
    """
    data_dir = data_dir or default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / "RECORDS"

    if cache_file.exists():
        lines = cache_file.read_text().splitlines()
    else:
        lines = wfdb.get_record_list(PN_DB_NAME)
        cache_file.write_text("\n".join(lines) + "\n")

    subjects = sorted({line.split("/")[0] for line in lines if line.strip()})
    return subjects


def list_sessions(subject_id: str, data_dir: Path | None = None) -> list[int]:
    """Return the sorted session numbers available for one subject."""
    subject_id = _validate_subject_id(subject_id)
    data_dir = data_dir or default_data_dir()
    cache_file = data_dir / "RECORDS"

    if cache_file.exists():
        lines = cache_file.read_text().splitlines()
    else:
        list_subjects(data_dir)  # populates the cache
        lines = (data_dir / "RECORDS").read_text().splitlines()

    sessions = []
    for line in lines:
        if line.startswith(f"{subject_id}/rec_"):
            sessions.append(int(line.split("rec_")[1]))
    return sorted(sessions)


def load_record(
    subject_id: str,
    session: int,
    data_dir: Path | None = None,
    download: bool = True,
) -> ECGRecord:
    """Load one subject/session recording, downloading it if not cached."""
    subject_id = _validate_subject_id(subject_id)
    session = _validate_session(session)
    data_dir = data_dir or default_data_dir()

    record_dir = data_dir / subject_id
    record_name = f"rec_{session}"
    local_path = record_dir / record_name

    if not (local_path.with_suffix(".hea")).exists():
        if not download:
            raise FileNotFoundError(
                f"no cached record at {local_path} and download=False"
            )
        record_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database(
            PN_DB_NAME,
            dl_dir=str(data_dir),
            records=[f"{subject_id}/{record_name}"],
        )

    rec = wfdb.rdrecord(str(local_path))

    if RAW_LEAD_NAME not in rec.sig_name:
        raise ValueError(
            f"record {subject_id}/{record_name} missing lead {RAW_LEAD_NAME!r}, "
            f"has {rec.sig_name!r}"
        )
    lead_idx = rec.sig_name.index(RAW_LEAD_NAME)
    signal = rec.p_signal[:, lead_idx].astype(np.float64)

    age = sex = None
    for comment in rec.comments or []:
        if comment.startswith("Age:"):
            age = comment.split(":", 1)[1].strip()
        elif comment.startswith("Sex:"):
            sex = comment.split(":", 1)[1].strip()

    return ECGRecord(
        subject_id=subject_id,
        session=session,
        signal=signal,
        fs=int(rec.fs),
        age=age,
        sex=sex,
    )
