"""Command-line entry point: `heartlock enroll|identify|evaluate`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from heartlock import data_loader as dl
from heartlock import evaluation as ev
from heartlock import matching as mm
from heartlock import preprocessing as pp
from heartlock import visualization as viz

DEFAULT_STORE = "enrollments.npz"


def _resolve_path(raw: str) -> Path:
    """Normalize a user-supplied path. Subject ids/sessions (the only
    values that get woven into PhysioNet file paths) are validated
    separately in `data_loader`; this just expands `~` and resolves `..`
    so downstream code always works with a clean absolute path.
    """
    return Path(raw).expanduser().resolve()


def _load_signal(args: argparse.Namespace) -> tuple[np.ndarray, float, str]:
    """Get a raw ECG signal + sample rate, either from a local .npy file
    or by fetching a subject/session recording from the ECG-ID database.
    """
    if args.file:
        file_path = _resolve_path(args.file)
        if file_path.suffix != ".npy":
            raise ValueError("--file must point to a .npy array (1-D float signal)")
        if not file_path.is_file():
            raise FileNotFoundError(f"no such file: {file_path}")
        signal = np.load(file_path, allow_pickle=False)
        if signal.ndim != 1:
            raise ValueError("--file must contain a 1-D signal array")
        if not args.fs:
            raise ValueError("--fs is required when loading a signal from --file")
        label = file_path.stem
        return signal.astype(np.float64), float(args.fs), label

    data_dir = _resolve_path(args.data_dir) if args.data_dir else None
    rec = dl.load_record(args.subject, args.session, data_dir=data_dir, source=args.source)
    return rec.signal, float(rec.fs), rec.subject_id


def cmd_enroll(args: argparse.Namespace) -> int:
    signal, fs, subject_id = _load_signal(args)
    clean = pp.preprocess(signal, fs, powerline_freq=args.powerline_freq)

    store_path = _resolve_path(args.store)
    enrollments = mm.load_enrollments(store_path) if store_path.exists() else []
    enrollments = [e for e in enrollments if e.subject_id != subject_id]
    enrollments.append(mm.enroll_subject(clean, fs, subject_id))

    mm.save_enrollments(enrollments, store_path)
    print(f"enrolled {subject_id!r} ({len(enrollments)} subject(s) now in {store_path})")
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    signal, fs, label = _load_signal(args)
    clean = pp.preprocess(signal, fs, powerline_freq=args.powerline_freq)

    store_path = _resolve_path(args.store)
    if not store_path.exists():
        print(f"no enrollment store found at {store_path}; enroll subjects first", file=sys.stderr)
        return 1

    enrollments = mm.load_enrollments(store_path)
    result = mm.identify(clean, fs, enrollments, method=args.method)

    if result.subject_id is None:
        print("no enrolled subjects to match against")
        return 1

    print(f"query: {label}")
    print(f"best match: {result.subject_id}  (confidence: {result.confidence:.4f})")
    print("all scores:")
    for sid, score in sorted(result.scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {sid}: {score:.4f}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    signal, fs, label = _load_signal(args)
    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    viz.plot_pipeline(signal, fs, label, out_path, powerline_freq=args.powerline_freq)
    print(f"wrote pipeline plot for {label!r} to {out_path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    signal, fs, label = _load_signal(args)
    clean = pp.preprocess(signal, fs, powerline_freq=args.powerline_freq)

    store_path = _resolve_path(args.store)
    if not store_path.exists():
        print(f"no enrollment store found at {store_path}; enroll subjects first", file=sys.stderr)
        return 1
    enrollments = mm.load_enrollments(store_path)

    threshold = args.threshold
    threshold_source = "--threshold"
    if threshold is None:
        results_dir = _resolve_path(args.results_dir)
        summary_path = results_dir / f"{args.method}_summary.json"
        if not summary_path.is_file():
            print(
                f"no --threshold given and no {summary_path} found; "
                "either pass --threshold or run `heartlock evaluate` first",
                file=sys.stderr,
            )
            return 1
        summary = json.loads(summary_path.read_text())
        per_subject = summary.get("per_subject_thresholds", {})
        if not args.no_per_subject_threshold and args.claim in per_subject:
            threshold = per_subject[args.claim]
            threshold_source = "per-subject EER threshold"
        else:
            threshold = summary["eer_threshold"]
            threshold_source = "global EER threshold"

    try:
        result = mm.verify(clean, fs, args.claim, enrollments, threshold, method=args.method)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    decision = "ACCEPTED" if result.accepted else "REJECTED"
    print(f"query: {label}  claimed identity: {result.claimed_id}")
    print(
        f"score: {result.score:.4f}  threshold: {result.threshold:.4f} "
        f"({threshold_source})  -> {decision}"
    )
    return 0 if result.accepted else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    data_dir = _resolve_path(args.data_dir) if args.data_dir else None
    results_dir = _resolve_path(args.results_dir) if args.results_dir else None

    summary = ev.run_evaluation(
        method=args.method,
        enroll_session=args.enroll_session,
        test_session=args.test_session,
        max_subjects=args.max_subjects,
        data_dir=data_dir,
        results_dir=results_dir,
    )

    print(f"method: {summary.method}")
    print(f"subjects evaluated: {summary.n_subjects}  queries: {summary.n_queries}")
    print(f"rank-1 accuracy: {summary.rank1_accuracy:.4f}")
    print(f"EER: {summary.eer:.4f}  (at threshold {summary.eer_threshold:.4f})")
    print(f"AUC: {summary.auc:.4f}")
    print(
        f"per-subject thresholds calibrated for {len(summary.per_subject_thresholds)}"
        f"/{summary.n_subjects} subjects (used automatically by `heartlock verify`)"
    )
    if results_dir:
        print(f"plots and summary written to {results_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heartlock", description="ECG-based biometric identification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_signal_args(sub: argparse.ArgumentParser, include_store: bool = True) -> None:
        sub.add_argument(
            "subject",
            nargs="?",
            help="subject/record id: ECG-ID 'Person_01' style, or a MIT-BIH record "
            "number with --source mitbih",
        )
        sub.add_argument("--session", type=int, default=1, help="recording session number (default: 1)")
        sub.add_argument("--file", help="load a raw signal from a local .npy file instead")
        sub.add_argument("--fs", type=float, help="sample rate in Hz, required with --file")
        sub.add_argument(
            "--source",
            choices=["ecgid", "mitbih"],
            default="ecgid",
            help="dataset to load subject/record from (default: ecgid). mitbih is the "
            "documented fallback when ECG-ID is unreachable - it has no repeat "
            "sessions per subject, so --session must be 1",
        )
        sub.add_argument("--data-dir", help="local ECG-ID cache directory (default: ./data)")
        sub.add_argument(
            "--powerline-freq", type=float, default=50.0, help="powerline notch frequency (default: 50)"
        )
        if include_store:
            sub.add_argument(
                "--store", default=DEFAULT_STORE, help="enrollment store file (default: enrollments.npz)"
            )

    enroll_parser = subparsers.add_parser("enroll", help="enroll a subject's ECG recording")
    add_common_signal_args(enroll_parser)
    enroll_parser.set_defaults(func=cmd_enroll)

    identify_parser = subparsers.add_parser("identify", help="identify an ECG snippet against enrolled subjects")
    add_common_signal_args(identify_parser)
    identify_parser.add_argument(
        "--method",
        choices=["template_corr", "template_dtw", "fiducial", "fusion"],
        default="template_corr",
        help="matching method (default: template_corr)",
    )
    identify_parser.set_defaults(func=cmd_identify)

    verify_parser = subparsers.add_parser(
        "verify", help="1:1 verify an ECG snippet against a claimed identity"
    )
    add_common_signal_args(verify_parser)
    verify_parser.add_argument("--claim", required=True, help="subject id being claimed")
    verify_parser.add_argument(
        "--method",
        choices=["template_corr", "template_dtw", "fiducial", "fusion"],
        default="template_corr",
        help="matching method (default: template_corr)",
    )
    verify_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="accept/reject score threshold (default: the claimed subject's per-subject "
        "EER threshold from results/<method>_summary.json if available, else the global "
        "EER threshold from the same file, written by `heartlock evaluate`)",
    )
    verify_parser.add_argument(
        "--no-per-subject-threshold",
        action="store_true",
        help="always use the global EER threshold, even if a per-subject one is available",
    )
    verify_parser.add_argument(
        "--results-dir", default="results", help="where to look for the evaluation summary (default: results)"
    )
    verify_parser.set_defaults(func=cmd_verify)

    plot_parser = subparsers.add_parser(
        "plot", help="visualize the pipeline (raw signal, R-peaks, beat template) for one recording"
    )
    add_common_signal_args(plot_parser, include_store=False)
    plot_parser.add_argument(
        "--out", default="results/pipeline.png", help="output image path (default: results/pipeline.png)"
    )
    plot_parser.set_defaults(func=cmd_plot)

    evaluate_parser = subparsers.add_parser("evaluate", help="run cross-session ROC/EER evaluation")
    evaluate_parser.add_argument(
        "--method", choices=["template_corr", "template_dtw", "fiducial", "fusion"], default="template_corr"
    )
    evaluate_parser.add_argument("--enroll-session", type=int, default=1)
    evaluate_parser.add_argument("--test-session", type=int, default=2)
    evaluate_parser.add_argument("--max-subjects", type=int, default=30)
    evaluate_parser.add_argument("--data-dir", help="local ECG-ID cache directory (default: ./data)")
    evaluate_parser.add_argument("--results-dir", default="results", help="where to write plots/summary")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in ("enroll", "identify", "verify", "plot") and not args.file and not args.subject:
        parser.error("either a subject id or --file must be given")

    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
