import numpy as np
import pytest

from heartlock import cli


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


def test_build_parser_requires_command():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_main_requires_subject_or_file(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["identify", "--store", "/tmp/does-not-matter.npz"])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "subject id or --file" in captured.err


def test_load_signal_from_file_requires_fs(tmp_path):
    npy_path = tmp_path / "signal.npy"
    np.save(npy_path, np.zeros(100))
    args = cli.build_parser().parse_args(["identify", "--file", str(npy_path)])
    with pytest.raises(ValueError, match="--fs is required"):
        cli._load_signal(args)


def test_load_signal_from_file_rejects_non_npy(tmp_path):
    bad_path = tmp_path / "signal.txt"
    bad_path.write_text("not a numpy file")
    args = cli.build_parser().parse_args(["identify", "--file", str(bad_path), "--fs", "500"])
    with pytest.raises(ValueError, match=r"\.npy"):
        cli._load_signal(args)


def test_load_signal_from_file_rejects_missing_file(tmp_path):
    missing = tmp_path / "nope.npy"
    args = cli.build_parser().parse_args(["identify", "--file", str(missing), "--fs", "500"])
    with pytest.raises(FileNotFoundError):
        cli._load_signal(args)


def test_load_signal_from_file_rejects_2d_array(tmp_path):
    npy_path = tmp_path / "signal.npy"
    np.save(npy_path, np.zeros((10, 10)))
    args = cli.build_parser().parse_args(["identify", "--file", str(npy_path), "--fs", "500"])
    with pytest.raises(ValueError, match="1-D"):
        cli._load_signal(args)


def test_enroll_and_identify_roundtrip_via_files(tmp_path, capsys):
    fs = 500
    store = tmp_path / "store.npz"

    low_t_path = tmp_path / "low_t.npy"
    high_t_path = tmp_path / "high_t.npy"
    np.save(low_t_path, _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.15, seed=1))
    np.save(high_t_path, _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=2))

    assert cli.main(["enroll", "--file", str(low_t_path), "--fs", str(fs), "--store", str(store)]) == 0
    assert cli.main(["enroll", "--file", str(high_t_path), "--fs", str(fs), "--store", str(store)]) == 0
    capsys.readouterr()

    query_path = tmp_path / "query.npy"
    np.save(query_path, _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=99))

    exit_code = cli.main(
        ["identify", "--file", str(query_path), "--fs", str(fs), "--store", str(store)]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "best match: high_t" in out


def test_identify_without_enrollment_store_fails_gracefully(tmp_path, capsys):
    fs = 500
    query_path = tmp_path / "query.npy"
    np.save(query_path, _synthetic_multi_beat_signal(fs, 20, 72))

    exit_code = cli.main(
        [
            "identify",
            "--file",
            str(query_path),
            "--fs",
            str(fs),
            "--store",
            str(tmp_path / "missing.npz"),
        ]
    )
    assert exit_code == 1
    assert "enroll subjects first" in capsys.readouterr().err


def test_verify_accepts_genuine_claim_via_files(tmp_path, capsys):
    fs = 500
    store = tmp_path / "store.npz"

    low_t_path = tmp_path / "low_t.npy"
    high_t_path = tmp_path / "high_t.npy"
    np.save(low_t_path, _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.15, seed=1))
    np.save(high_t_path, _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=2))

    assert cli.main(["enroll", "--file", str(low_t_path), "--fs", str(fs), "--store", str(store)]) == 0
    assert cli.main(["enroll", "--file", str(high_t_path), "--fs", str(fs), "--store", str(store)]) == 0
    capsys.readouterr()

    query_path = tmp_path / "query.npy"
    np.save(query_path, _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=99))

    exit_code = cli.main(
        [
            "verify",
            "--file",
            str(query_path),
            "--fs",
            str(fs),
            "--store",
            str(store),
            "--claim",
            "high_t",
            "--threshold",
            "0.95",
        ]
    )
    assert exit_code == 0
    assert "ACCEPTED" in capsys.readouterr().out


def test_verify_rejects_impostor_claim_via_files(tmp_path, capsys):
    fs = 500
    store = tmp_path / "store.npz"

    low_t_path = tmp_path / "low_t.npy"
    high_t_path = tmp_path / "high_t.npy"
    np.save(low_t_path, _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.15, seed=1))
    np.save(high_t_path, _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=2))

    assert cli.main(["enroll", "--file", str(low_t_path), "--fs", str(fs), "--store", str(store)]) == 0
    assert cli.main(["enroll", "--file", str(high_t_path), "--fs", str(fs), "--store", str(store)]) == 0
    capsys.readouterr()

    query_path = tmp_path / "query.npy"
    np.save(query_path, _synthetic_multi_beat_signal(fs, 20, 72, t_amplitude=0.6, seed=99))

    exit_code = cli.main(
        [
            "verify",
            "--file",
            str(query_path),
            "--fs",
            str(fs),
            "--store",
            str(store),
            "--claim",
            "low_t",
            "--threshold",
            "0.95",
        ]
    )
    assert exit_code == 1
    assert "REJECTED" in capsys.readouterr().out


def test_verify_unknown_claim_fails_gracefully(tmp_path, capsys):
    fs = 500
    store = tmp_path / "store.npz"
    signal_path = tmp_path / "sig.npy"
    np.save(signal_path, _synthetic_multi_beat_signal(fs, 20, 72, seed=1))

    assert cli.main(["enroll", "--file", str(signal_path), "--fs", str(fs), "--store", str(store)]) == 0
    capsys.readouterr()

    exit_code = cli.main(
        [
            "verify",
            "--file",
            str(signal_path),
            "--fs",
            str(fs),
            "--store",
            str(store),
            "--claim",
            "nobody",
            "--threshold",
            "0.5",
        ]
    )
    assert exit_code == 1
    assert "no enrollment found" in capsys.readouterr().err


def test_verify_without_threshold_or_summary_fails_gracefully(tmp_path, capsys):
    fs = 500
    store = tmp_path / "store.npz"
    signal_path = tmp_path / "sig.npy"
    np.save(signal_path, _synthetic_multi_beat_signal(fs, 20, 72, seed=1))

    assert cli.main(["enroll", "--file", str(signal_path), "--fs", str(fs), "--store", str(store)]) == 0
    capsys.readouterr()

    exit_code = cli.main(
        [
            "verify",
            "--file",
            str(signal_path),
            "--fs",
            str(fs),
            "--store",
            str(store),
            "--claim",
            "sig",
            "--results-dir",
            str(tmp_path / "no_results_here"),
        ]
    )
    assert exit_code == 1
    assert "no --threshold given" in capsys.readouterr().err


def test_verify_without_enrollment_store_fails_gracefully(tmp_path, capsys):
    fs = 500
    query_path = tmp_path / "query.npy"
    np.save(query_path, _synthetic_multi_beat_signal(fs, 20, 72))

    exit_code = cli.main(
        [
            "verify",
            "--file",
            str(query_path),
            "--fs",
            str(fs),
            "--store",
            str(tmp_path / "missing.npz"),
            "--claim",
            "query",
            "--threshold",
            "0.5",
        ]
    )
    assert exit_code == 1
    assert "enroll subjects first" in capsys.readouterr().err


def test_plot_writes_image_file_via_files(tmp_path, capsys):
    fs = 500
    signal_path = tmp_path / "sig.npy"
    np.save(signal_path, _synthetic_multi_beat_signal(fs, 20, 72, seed=1))
    out_path = tmp_path / "pipeline.png"

    exit_code = cli.main(
        ["plot", "--file", str(signal_path), "--fs", str(fs), "--out", str(out_path)]
    )
    assert exit_code == 0
    assert out_path.is_file()
    assert "wrote pipeline plot" in capsys.readouterr().out


def test_enroll_rejects_path_traversal_subject_id(capsys):
    exit_code = cli.main(["enroll", "../../etc", "--store", "/tmp/store.npz"])
    assert exit_code == 1
    assert "invalid subject id" in capsys.readouterr().err


@pytest.mark.network
def test_enroll_and_identify_real_subjects_via_cli(tmp_path, capsys):
    store = tmp_path / "store.npz"
    data_dir = tmp_path / "data"

    for sid in ["Person_01", "Person_02"]:
        code = cli.main(
            ["enroll", sid, "--session", "1", "--store", str(store), "--data-dir", str(data_dir)]
        )
        assert code == 0

    capsys.readouterr()
    code = cli.main(
        [
            "identify",
            "Person_02",
            "--session",
            "2",
            "--store",
            str(store),
            "--data-dir",
            str(data_dir),
        ]
    )
    assert code == 0
    assert "best match: Person_02" in capsys.readouterr().out
