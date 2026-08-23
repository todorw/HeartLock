"""Waveform visualization: inspect what each pipeline stage actually did.

This project is explicitly signal-processing-first, not a black-box
classifier, so it should be possible to *look at* what the R-peak
detector and template builder are doing on a given recording rather than
just trusting the final match score.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from heartlock import preprocessing as pp
from heartlock import r_peak_detection as rpd
from heartlock.feature_extraction import BEAT_PRE_MS, build_average_template

DEFAULT_PREVIEW_SECONDS = 6.0


def plot_pipeline(
    raw_signal: np.ndarray,
    fs: float,
    label: str,
    save_path: Path,
    powerline_freq: float = 50.0,
    preview_seconds: float = DEFAULT_PREVIEW_SECONDS,
) -> None:
    """Three-panel figure: raw signal, filtered signal with detected
    R-peaks marked, and the averaged beat template used for matching.

    Only the first `preview_seconds` are plotted for the top two panels
    (a 20s recording at 500Hz is unreadable at full width), but R-peak
    detection and the template still run over the *entire* signal so the
    plot reflects what matching/evaluation actually use.
    """
    clean = pp.preprocess(raw_signal, fs, powerline_freq)
    r_peaks = rpd.detect_r_peaks(clean, fs)
    template = build_average_template(raw_signal, fs)

    n_preview = min(len(raw_signal), int(round(preview_seconds * fs)))
    t = np.arange(n_preview) / fs

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    axes[0].plot(t, raw_signal[:n_preview], color="tab:gray", linewidth=0.8)
    axes[0].set_title(f"{label}: raw signal (first {preview_seconds:.0f}s)")
    axes[0].set_ylabel("amplitude (mV)")
    axes[0].set_xlabel("time (s)")

    axes[1].plot(t, clean[:n_preview], color="tab:blue", linewidth=0.8)
    preview_peaks = r_peaks[r_peaks < n_preview]
    axes[1].plot(
        preview_peaks / fs, clean[preview_peaks], "ro", markersize=5, label="detected R-peak"
    )
    axes[1].set_title(f"filtered signal ({len(r_peaks)} R-peaks detected total)")
    axes[1].set_ylabel("amplitude (z-score)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()

    if template is not None:
        r_offset_samples = int(round(BEAT_PRE_MS / 1000.0 * fs))
        beat_t_ms = (np.arange(len(template)) - r_offset_samples) / fs * 1000.0
        axes[2].plot(beat_t_ms, template, color="tab:green")
        axes[2].axvline(0, color="gray", linestyle="--", linewidth=1, label="R-peak")
        axes[2].set_title(f"averaged beat template ({label})")
        axes[2].set_xlabel("time relative to R-peak (ms)")
        axes[2].legend()
    else:
        axes[2].set_title("averaged beat template: no beats could be segmented")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
