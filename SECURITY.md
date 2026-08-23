# Security

## Threat model

HeartLock is a research/portfolio project exploring ECG waveform morphology as
a biometric signal. **It is not a production authentication system and must
never be used to gate access to anything real.** There is no liveness
detection, no protection against replay of a recorded/synthesized ECG
signal, no template encryption, and no defense against an adversary who can
supply arbitrary input signals. A biometric feature this easy to bypass with
a recording is not a security control.

Within that scope, the things this project does try to get right:

- **No raw biometric data in the repository.** `data/` (all downloaded
  PhysioNet recordings) is gitignored from the first commit, before any
  data is ever pulled.
- **No secrets in the codebase.** There are none needed for this project
  (PhysioNet's open-access databases require no credentials), but `.env`
  is gitignored in case that ever changes.
- **CLI input validation.** Subject ids and session numbers supplied on the
  command line are validated against a strict pattern before being used to
  construct any file path or PhysioNet URL, to prevent path traversal (see
  `heartlock/data_loader.py::_validate_subject_id`).
- **No `eval`/`exec`/`pickle` on any input.** Enrollment data is persisted
  with `numpy.savez`/`numpy.load(..., allow_pickle=False)`, never `pickle`.
- **Pinned dependencies.** `requirements.txt` and `pyproject.toml` pin exact
  versions to avoid unreviewed floating-version supply-chain surprises.
- **Static analysis.** `bandit` is run over the codebase as part of the
  development workflow.

## Reporting

This is a personal portfolio project with no production deployment. If you
spot a security issue anyway, please open an issue on the repository.
