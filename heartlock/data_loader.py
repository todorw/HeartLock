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

# MIT-BIH Arrhythmia Database: the documented fallback when ECG-ID is
# unavailable (see prompt.md / README). Unlike ECG-ID it has no repeat
# sessions per subject (one continuous ~30min recording per record) and
# far fewer subjects (48 vs 90), so it only supports enroll/identify/verify,
# not the cross-session evaluation ECG-ID is built for.
MITBIH_DB_NAME = "mitdb"
MITBIH_LEAD_NAME = "MLII"

# Fixed set of valid MIT-BIH record ids (PhysioNet mitdb). Record ids are
# user-facing and get woven into file paths, so - like ECG-ID subject ids -
# they're validated against this known-good set rather than a loose regex.
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


@dataclass(frozen=True)
class ECGRecord:
    """A single ECG recording: one subject, one session."""

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


def _validate_mitbih_record(record_id: str) -> str:
    """Reject anything that isn't one of the fixed, known MIT-BIH record
    ids - like `_validate_subject_id`, this doubles as the path-traversal
    guard since `record_id` reaches disk/network path construction.
    """
    if record_id not in MITBIH_RECORDS:
        raise ValueError(
            f"invalid mit-bih record id {record_id!r}, expected one of {MITBIH_RECORDS}"
        )
    return record_id


def _ecgid_unavailable_error(exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"could not reach PhysioNet for the ECG-ID database ({exc}). "
        "As a fallback, the MIT-BIH Arrhythmia Database can be used instead "
        "via --source mitbih (e.g. `heartlock enroll 100 --source mitbih`), "
        "though it has no repeat sessions per subject so `evaluate` isn't "
        "supported with it."
    )


def list_subjects(data_dir: Path | None = None, source: str = "ecgid") -> list[str]:
    """Return sorted subject/record ids available from `source`.

    ECG-ID ids look like 'Person_01'..'Person_90'; MIT-BIH ids are the
    fixed 3-digit PhysioNet record numbers in `MITBIH_RECORDS` and need no
    network access to list, since that set is hardcoded rather than fetched.
    """
    if source == "mitbih":
        return list(MITBIH_RECORDS)
    if source != "ecgid":
        raise ValueError(f"unknown data source {source!r}, expected 'ecgid' or 'mitbih'")

    data_dir = data_dir or default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / "RECORDS"

    if cache_file.exists():
        lines = cache_file.read_text().splitlines()
    else:
        try:
            lines = wfdb.get_record_list(PN_DB_NAME)
        except Exception as exc:
            raise _ecgid_unavailable_error(exc) from exc
        cache_file.write_text("\n".join(lines) + "\n")

    subjects = sorted({line.split("/")[0] for line in lines if line.strip()})
    return subjects


def list_sessions(subject_id: str, data_dir: Path | None = None, source: str = "ecgid") -> list[int]:
    """Return the sorted session numbers available for one subject/record.

    MIT-BIH records have no repeat-session structure, so this always
    returns `[1]` for `source="mitbih"`.
    """
    if source == "mitbih":
        _validate_mitbih_record(subject_id)
        return [1]
    if source != "ecgid":
        raise ValueError(f"unknown data source {source!r}, expected 'ecgid' or 'mitbih'")

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


def _load_ecgid_record(
    subject_id: str, session: int, data_dir: Path, download: bool
) -> ECGRecord:
    subject_id = _validate_subject_id(subject_id)
    session = _validate_session(session)

    record_dir = data_dir / subject_id
    record_name = f"rec_{session}"
    local_path = record_dir / record_name

    if not (local_path.with_suffix(".hea")).exists():
        if not download:
            raise FileNotFoundError(
                f"no cached record at {local_path} and download=False"
            )
        record_dir.mkdir(parents=True, exist_ok=True)
        try:
            wfdb.dl_database(
                PN_DB_NAME,
                dl_dir=str(data_dir),
                records=[f"{subject_id}/{record_name}"],
            )
        except Exception as exc:
            raise _ecgid_unavailable_error(exc) from exc

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


def _load_mitbih_record(record_id: str, session: int, data_dir: Path, download: bool) -> ECGRecord:
    record_id = _validate_mitbih_record(record_id)
    if session != 1:
        raise ValueError(
            f"mit-bih records have no repeat sessions; got session={session}, expected 1"
        )

    record_dir = data_dir / "mitbih"
    local_path = record_dir / record_id

    if not local_path.with_suffix(".hea").exists():
        if not download:
            raise FileNotFoundError(f"no cached record at {local_path} and download=False")
        record_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database(MITBIH_DB_NAME, dl_dir=str(record_dir), records=[record_id])

    rec = wfdb.rdrecord(str(local_path))

    if MITBIH_LEAD_NAME not in rec.sig_name:
        raise ValueError(
            f"mit-bih record {record_id} missing lead {MITBIH_LEAD_NAME!r}, has {rec.sig_name!r}"
        )
    lead_idx = rec.sig_name.index(MITBIH_LEAD_NAME)
    signal = rec.p_signal[:, lead_idx].astype(np.float64)

    return ECGRecord(subject_id=record_id, session=1, signal=signal, fs=int(rec.fs))


def load_record(
    subject_id: str,
    session: int,
    data_dir: Path | None = None,
    download: bool = True,
    source: str = "ecgid",
) -> ECGRecord:
    """Load one subject/session recording, downloading it if not cached.

    `source="mitbih"` loads from the fallback MIT-BIH Arrhythmia Database
    instead of ECG-ID (see `MITBIH_RECORDS`); `session` must be 1 there
    since MIT-BIH has no repeat-session structure.
    """
    data_dir = data_dir or default_data_dir()
    if source == "mitbih":
        return _load_mitbih_record(subject_id, session, data_dir, download)
    if source != "ecgid":
        raise ValueError(f"unknown data source {source!r}, expected 'ecgid' or 'mitbih'")
    return _load_ecgid_record(subject_id, session, data_dir, download)
