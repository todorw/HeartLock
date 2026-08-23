import pytest

from heartlock import data_loader as dl


def test_validate_subject_id_accepts_well_formed():
    assert dl._validate_subject_id("Person_01") == "Person_01"


@pytest.mark.parametrize(
    "bad_id",
    ["../../etc", "Person_01/../../x", "Person_1", "person_01", "Person_01.hea", ""],
)
def test_validate_subject_id_rejects_path_traversal_and_malformed(bad_id):
    with pytest.raises(ValueError):
        dl._validate_subject_id(bad_id)


@pytest.mark.parametrize("bad_session", [0, -1, 51, "abc"])
def test_validate_session_rejects_out_of_range(bad_session):
    with pytest.raises(ValueError):
        dl._validate_session(bad_session)


def test_validate_session_accepts_in_range():
    assert dl._validate_session(1) == 1
    assert dl._validate_session("5") == 5


def test_validate_mitbih_record_accepts_known_id():
    assert dl._validate_mitbih_record("100") == "100"


@pytest.mark.parametrize("bad_id", ["../../etc", "999", "100/../200", "", "MLII"])
def test_validate_mitbih_record_rejects_unknown_or_traversal(bad_id):
    with pytest.raises(ValueError):
        dl._validate_mitbih_record(bad_id)


def test_list_subjects_mitbih_needs_no_network(tmp_path):
    subjects = dl.list_subjects(tmp_path, source="mitbih")
    assert subjects == dl.MITBIH_RECORDS
    assert "100" in subjects


def test_list_sessions_mitbih_is_always_session_one():
    assert dl.list_sessions("100", source="mitbih") == [1]


def test_list_subjects_rejects_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="unknown data source"):
        dl.list_subjects(tmp_path, source="bogus")


def test_load_record_mitbih_rejects_non_one_session(tmp_path):
    with pytest.raises(ValueError, match="no repeat sessions"):
        dl.load_record("100", 2, data_dir=tmp_path, source="mitbih", download=False)


@pytest.mark.network
def test_load_record_mitbih_returns_expected_shape(tmp_path):
    rec = dl.load_record("100", 1, data_dir=tmp_path, source="mitbih")
    assert rec.subject_id == "100"
    assert rec.session == 1
    assert rec.fs == 360
    assert rec.signal.ndim == 1
    assert rec.signal.shape[0] > 0


@pytest.mark.network
def test_load_record_returns_expected_shape(tmp_path):
    rec = dl.load_record("Person_01", 1, data_dir=tmp_path)
    assert rec.subject_id == "Person_01"
    assert rec.session == 1
    assert rec.fs == 500
    assert rec.signal.ndim == 1
    assert rec.signal.shape[0] > 0


@pytest.mark.network
def test_list_subjects_and_sessions(tmp_path):
    subjects = dl.list_subjects(tmp_path)
    assert "Person_01" in subjects
    assert len(subjects) == 90

    sessions = dl.list_sessions("Person_01", tmp_path)
    assert sessions == sorted(sessions)
    assert 1 in sessions
