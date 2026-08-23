# HeartLock

ECG-based biometric identification, built as a signal-processing pipeline
rather than a black-box classifier: it filters a raw ECG lead, detects
heartbeats, extracts interpretable per-beat features, and matches an
unknown recording against a set of enrolled subjects.

> **This is a research/portfolio project, not a real authentication
> system.** See [SECURITY.md](SECURITY.md) for the threat model. There is
> no liveness detection and no protection against a replayed or
> synthesized signal — do not use this to gate access to anything.

## Why ECG biometrics

Every person's ECG has a distinctive P-QRS-T morphology shaped by their
individual heart geometry and conduction pathways — how the electrical
signal travels through *their* heart, not just *that* it beats. Unlike a
fingerprint, it's also a live physiological signal, and unlike a face it's
not something a photo can capture. That combination makes it an interesting
biometric research problem: can heartbeat *shape* alone tell two people
apart, even across recording sessions taken on different days?

That last part is the hard part. Within a single recording, telling two
people's beats apart is comparatively easy — but the real test of a
biometric feature is whether it survives being re-measured later, with
different electrode placement, hydration, stress level, and skin
conductivity. This project evaluates that cross-session case specifically,
because it's the case that actually matters.

## Dataset

[ECG-ID Database](https://physionet.org/content/ecgiddb/1.0.0/) on
PhysioNet: 90 subjects, 310 recordings total (2-20 sessions per subject,
20 seconds each at 500 Hz, single lead), purpose-built for ECG biometric
research. Fetched on demand via `wfdb` and cached locally in `data/`
(gitignored — never committed).

## Pipeline

```
raw signal → preprocessing → R-peak detection → feature extraction → matching → identification
```

1. **`data_loader.py`** — fetches and caches ECG-ID recordings from
   PhysioNet, one clean interface per subject/session.
2. **`preprocessing.py`** — 0.5-40 Hz bandpass (removes baseline wander and
   high-frequency noise) + notch filter (removes 50/60 Hz powerline hum) +
   z-score normalization.
3. **`r_peak_detection.py`** — Pan-Tompkins QRS detector (bandpass →
   derivative → square → moving-window integrate → adaptive threshold,
   with T-wave discrimination to avoid double-counting a beat), then
   segments the signal into fixed-length, R-aligned beat windows.
4. **`feature_extraction.py`** — two independent feature representations:
   - *Fiducial*: P/QRS/T amplitudes, durations, and the PR/QT intervals,
     from a heuristic landmark delineator.
   - *Non-fiducial*: the median beat waveform (a per-subject "template"),
     used directly for shape matching without any landmark detection.
5. **`matching.py`** — enrolls a subject as (template, fiducial vector).
   Two distinct operations are exposed: `identify()`, 1:N — compare a
   query against every enrolled subject via cross-correlation, DTW
   (`fastdtw`), or z-scored fiducial distance, and return the best match
   **and its score** (never a bare label); and `verify()`, 1:1 — score a
   query against a single *claimed* identity and accept/reject against a
   threshold, the operation an access-control-style system would
   actually use.
6. **`evaluation.py`** — the actual biometric test: enroll on one session,
   identify on a *different* session of the same subjects, and report
   rank-1 accuracy plus a proper ROC curve and Equal Error Rate (EER). Also
   calibrates a **per-subject decision threshold** for `verify`: some
   enrolled templates score consistently higher/noisier against everyone
   than others do, so a single global threshold over- or under-protects
   them. Each subject's impostor scores are Z-normalized (mean/std of the
   scores every *other* subject got against them), a single EER threshold
   is found in that normalized space, and it's mapped back to each
   subject's own raw score scale.
7. **`cli.py`** — `heartlock enroll` / `identify` / `verify` / `evaluate`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.11+.

## Usage

```bash
# enroll a subject (fetches and caches their ECG-ID recording)
heartlock enroll Person_01 --session 1

# identify a different session against everyone enrolled so far
heartlock identify Person_01 --session 2

# identify an arbitrary local signal instead of a database subject
heartlock identify --file my_snippet.npy --fs 500

# 1:1 verify a claimed identity (accept/reject, not a ranked match)
heartlock verify Person_01 --session 2 --claim Person_01

# cross-session ROC/EER evaluation over a batch of subjects
heartlock evaluate --method template_corr --enroll-session 1 --test-session 2 --max-subjects 30
```

`--method` on `identify`/`verify`/`evaluate` selects the matcher:
`template_corr` (cross-correlation of the averaged beat template, the
default), `template_dtw` (DTW alignment of the same template), or
`fiducial` (landmark-feature distance).

`identify` is 1:N: "who, out of everyone enrolled, does this look like?"
`verify` is the different, 1:1 question an access-control-style system
actually asks: "does this match the identity being *claimed*?" It accepts
or rejects based on a score threshold rather than returning a ranked
list. If `--threshold` isn't given, it's read from
`results/<method>_summary.json` — the claimed subject's own per-subject
EER threshold if that subject was in the last `evaluate` run (pass
`--no-per-subject-threshold` to always use the global one instead), else
the global EER threshold. Run `evaluate` first, or pass `--threshold`
explicitly.

```bash
# visualize the pipeline: raw signal, filtered signal + detected
# R-peaks, and the averaged beat template used for matching
heartlock plot Person_01 --session 1 --out results/person_01_pipeline.png
```

## Interpreting the evaluation output

`heartlock evaluate` writes `<method>_summary.json`, `<method>_roc.png`,
and `<method>_far_frr.png` to `results/`:

- **Rank-1 accuracy** — how often the top-scoring match is actually
  correct. Easy to inflate, so treat it as a headline number only.
- **FAR / FRR vs threshold** — the actual trade-off a deployed system
  would face: at any given match-score threshold, how often does an
  impostor get falsely accepted (FAR) versus how often does the real
  subject get falsely rejected (FRR)? Lower is better for both,
  necessarily traded off against each other.
- **EER (Equal Error Rate)** — the point where FAR and FRR are equal; the
  standard single-number summary of a biometric matcher's separability.
  Lower is better; sub-1% is very good, over 20% is close to unusable.
  This is the number to look at first.
- **ROC curve / AUC** — true-accept rate vs false-accept rate across all
  thresholds; AUC close to 1.0 means genuine and impostor score
  distributions barely overlap.

With `template_corr` on a 15-subject cross-session run (session 1 enrolled,
session 2 tested), this pipeline gets roughly 90%+ rank-1 accuracy and a
single-digit-percent EER — solid for a from-scratch signal-processing
system on a snippet as short as 20 seconds, but nowhere near the
reliability a real authentication factor would need.

## Testing

```bash
pytest                       # runs everything, including tests that fetch from PhysioNet
pytest -m "not network"      # skip tests that need internet access
```

## Known limitations

- The fiducial delineator is a lightweight heuristic (peak search in
  physiologically-motivated windows + threshold crossing), not a
  clinically validated wave-boundary algorithm — expect it to
  occasionally misfire on noisy or atypical beats.
- R-peak detection accuracy varies by subject; a few ECG-ID recordings
  are noisy enough that Pan-Tompkins' adaptive thresholding produces
  irregular RR intervals. Evaluation aggregates the median beat, which is
  fairly robust to a handful of bad beats, but very noisy recordings will
  still drag down that subject's match quality.
- No MIT-BIH fallback is implemented; ECG-ID has been reliably available.
