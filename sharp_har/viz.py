"""Training/eval visualization: plots built from the run artifacts
(history.csv, harness *_confusion.csv) — never from live training state,
so any committed run can be re-plotted offline. Ref. §0.4 (run
artifacts), §6-C2 (GRL monitoring), §9 (per-set reporting, confusion
matrix for the primary rotation).

Thin-notebook contract: notebooks only call `plot_history`,
`plot_confusion`, `compare_runs`, `metrics_table` and display the
output; all plotting/tabulation logic lives here.

Panels are driven by the columns actually present in history.csv, so
the same function covers every config:
- train_loss                       -> always (CE and SupCon phase A)
- val_macro_f1                     -> CE runs only (§6-C3: no in-loop
                                      selection in phase A), with the
                                      best (val-selected) epoch marked
- arset_train_acc + grl_lambda     -> GRL runs (C2/C4), the mandatory
                                      §6-C2 monitoring pair (both in
                                      [0, 1], shared axis)
- lr, s_per_step                   -> always (the gates read s_per_step)

`plot_embeddings` is the one exception to the history.csv contract: it
reads a `harness.cache_features` .npz directly (§9 v5.2 key figure).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Fixed categorical order (assigned by position, never cycled); recessive
# chrome for grid/axes so the data ink dominates.
PALETTE = (
    "#2a78d6",  # blue
    "#008300",  # green
    "#e87ba4",  # magenta
    "#eda100",  # yellow
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#4a3aa7",  # violet
    "#e34948",  # red
)
_GRID = "#e1e0d9"
_MUTED = "#898781"
_INK = "#0b0b0b"


def _style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=_INK)
    ax.set_xlabel("epoch", fontsize=9, color=_MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=_MUTED)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.tick_params(labelsize=8, colors=_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_GRID)


def load_history(run_dir: str | Path) -> pd.DataFrame:
    """Loads a run's history.csv (written per epoch by train.train_run,
    §0.4). Accepts the run directory or the CSV path itself."""
    path = Path(run_dir)
    if path.is_dir():
        path = path / "history.csv"
    assert path.exists(), f"no history.csv at {path} — has the run started?"
    return pd.read_csv(path)


def plot_history(run_dir: str | Path, save_path: str | Path | None = None) -> Figure:
    """One figure with a panel per metric available in the run's
    history.csv (see module docstring for the column -> panel map).
    `run_dir` is ckpt_root/<config name>. Optionally saves to
    `save_path` (png); always returns the Figure."""
    run_dir = Path(run_dir)
    hist = load_history(run_dir)
    name = run_dir.name if run_dir.is_dir() else run_dir.parent.name
    epochs = hist["epoch"]

    def loss_panel(ax: plt.Axes) -> None:
        ax.plot(epochs, hist["train_loss"], color=PALETTE[0], linewidth=1.8)
        _style_axis(ax, "Train loss (epoch mean)", "loss")

    def val_panel(ax: plt.Axes) -> None:
        ax.plot(epochs, hist["val_macro_f1"], color=PALETTE[1], linewidth=1.8)
        best = hist["val_macro_f1"].idxmax()
        ax.scatter(
            [epochs[best]], [hist["val_macro_f1"][best]],
            color=PALETTE[1], s=36, zorder=3,
        )
        ax.annotate(
            f"best: {hist['val_macro_f1'][best]:.3f} (epoch {int(epochs[best])})",
            (epochs[best], hist["val_macro_f1"][best]),
            textcoords="offset points", xytext=(6, 6), fontsize=8, color=_INK,
        )
        _style_axis(ax, "Val macro-F1 (fused, selection metric)", "macro-F1")
        ax.set_ylim(0.0, 1.0)

    def grl_panel(ax: plt.Axes) -> None:
        # §6-C2 monitoring: the AR-set head accuracy must fall toward the
        # majority baseline as lambda ramps. Both series live in [0, 1].
        ax.plot(epochs, hist["arset_train_acc"], color=PALETTE[2], linewidth=1.8,
                label="AR-set train acc")
        ax.plot(epochs, hist["grl_lambda"], color=_MUTED, linewidth=1.4,
                linestyle="--", label="GRL λ(p)")
        ax.legend(fontsize=8, frameon=False)
        _style_axis(ax, "GRL adversary monitoring (§6-C2)", "value")
        ax.set_ylim(0.0, 1.05)

    def lr_panel(ax: plt.Axes) -> None:
        ax.plot(epochs, hist["lr"], color=PALETTE[3], linewidth=1.8)
        _style_axis(ax, "Learning rate", "lr")

    def speed_panel(ax: plt.Axes) -> None:
        ax.plot(epochs, hist["s_per_step"], color=PALETTE[5], linewidth=1.8)
        _style_axis(ax, "Throughput (gates read this)", "s / step")
        ax.set_ylim(bottom=0.0)

    panels: list[Callable[[plt.Axes], None]] = [loss_panel]
    if "val_macro_f1" in hist.columns:
        panels.append(val_panel)
    if "arset_train_acc" in hist.columns:
        panels.append(grl_panel)
    panels += [lr_panel, speed_panel]

    n_cols = 2
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for panel, ax in zip(panels, axes):
        panel(ax)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"{name} — {len(hist)} epochs", fontsize=12, color=_INK)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion(
    confusion_csv: str | Path,
    normalize: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Heatmap of a harness `*_confusion.csv` (rows = true class,
    columns = predicted, §9). `normalize` shows row-normalized shades
    (per-class recall structure); the annotated numbers are always the
    raw window counts."""
    confusion_csv = Path(confusion_csv)
    cm = pd.read_csv(confusion_csv, index_col=0)
    counts = cm.to_numpy(dtype=np.int64)
    shade = counts.astype(np.float64)
    if normalize:
        shade = shade / np.maximum(shade.sum(axis=1, keepdims=True), 1)

    cmap = LinearSegmentedColormap.from_list(
        "seq_blue", ["#fcfcfb", "#86b6ef", "#2a78d6", "#0d366b"]
    )
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(cm.columns), 1.0 + 0.65 * len(cm)))
    im = ax.imshow(shade, cmap=cmap, vmin=0.0, aspect="equal")
    ax.set_xticks(range(len(cm.columns)), cm.columns, fontsize=9)
    ax.set_yticks(range(len(cm.index)), cm.index, fontsize=9)
    ax.set_xlabel("predicted", fontsize=9, color=_MUTED)
    ax.set_ylabel("true", fontsize=9, color=_MUTED)
    ax.set_title(confusion_csv.stem, fontsize=10, color=_INK)
    threshold = shade.max() * 0.55 if shade.max() > 0 else 1.0
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(
                j, i, str(counts[i, j]), ha="center", va="center", fontsize=8,
                color="white" if shade[i, j] > threshold else _INK,
            )
    fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compare_runs(
    run_dirs: dict[str, str | Path],
    metric: str = "val_macro_f1",
    save_path: str | Path | None = None,
) -> Figure:
    """Overlays one history.csv column across runs ({label: run_dir}),
    e.g. C0-C4 val macro-F1 side by side. Colors follow the insertion
    order of `run_dirs` (fixed assignment, §0.5 comparability: remember
    differences under ~2 points are 'comparable', not improvements).
    Runs whose history lacks `metric` (e.g. phase A vs val_macro_f1)
    are skipped with a note."""
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = 0
    for i, (label, run_dir) in enumerate(run_dirs.items()):
        hist = load_history(run_dir)
        if metric not in hist.columns:
            print(f"{label}: no {metric!r} column in history.csv, skipped")
            continue
        ax.plot(hist["epoch"], hist[metric], color=PALETTE[i % len(PALETTE)],
                linewidth=1.8, label=label)
        plotted += 1
    assert plotted > 0, f"no run in {sorted(run_dirs)} has a {metric!r} column"
    if plotted > 1:
        ax.legend(fontsize=9, frameon=False)
    _style_axis(ax, f"{metric} across runs", metric)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def metrics_table(
    metrics_csvs: Mapping[str, str | Path],
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """Per-AR-set comparison table from harness `*_metrics.csv` files
    ({run label: csv path}): fused rows only (the per-antenna appendix
    stays in the CSVs), pivoted to run × AR-set with `metric` values
    ("macro_f1" or "accuracy"). Rows keep the insertion order of
    `metrics_csvs`, "ALL" is the first column. Ref. §9 (never
    aggregate-only); §0.5: differences under ~2 points are
    "comparable", not improvements. Runs from different protocols
    (e.g. C0 on P1) simply leave NaN in the AR-sets they don't cover."""
    rows = []
    for label, path in metrics_csvs.items():
        df = pd.read_csv(path)
        fused = df[df["aggregation"].str.startswith("fused")]
        assert len(fused) > 0, f"{path}: no fused rows — not a harness metrics CSV?"
        for _, r in fused.iterrows():
            rows.append({"run": label, "ar_set": r["ar_set"], metric: float(r[metric])})
    table = pd.DataFrame(rows).pivot(index="run", columns="ar_set", values=metric)
    cols = ["ALL"] + sorted(c for c in table.columns if c != "ALL")
    return table.reindex(index=list(metrics_csvs), columns=cols)


def plot_embeddings(
    feature_caches: Mapping[str, str | Path],
    *,
    windows_per_trace: int = 8,
    pca_components: int = 50,
    perplexity: float = 30.0,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> Figure:
    """PCA(50) -> t-SNE of TRAIN features (§9 v5.2 key figure, replaces
    the underpowered §7 ar_set probe as the qualitative invariance
    picture): one row per encoder (e.g. {"C1": ..., "C3": ...}), two
    panels each — colored by activity and by AR-set.

    Declared recipe, identical across encoders so panels are
    comparable (§9): features L2-normalized (SupCon optimizes angles,
    not scale — same treatment for CE keeps the comparison
    apples-to-apples); subsampled to `windows_per_trace` windows per
    trace, all if fewer (bounds how much a single long recording's
    near-duplicate windows can dominate the picture); PCA-50 -> t-SNE
    perplexity 30, seed 42.

    TRAIN features only: the only split with every AR-set and both
    environments present (val is 9 traces / 5 AR-sets, AR-3 absent;
    test is a single domain, S7) — same declared scope as the §7
    train-feature domain diagnostics, memorization confound included.
    No test contact.

    Qualitative figure, read alongside the probe/diagnostic numbers,
    never a standalone claim: t-SNE preserves local neighborhoods, not
    global inter-cluster distances or cluster sizes.

    `feature_caches`: {label: path to a harness.cache_features "*_train.npz"}.
    """
    rows = list(feature_caches.items())
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 4.6 * len(rows)), squeeze=False)

    for row_idx, (label, path) in enumerate(rows):
        data = np.load(path, allow_pickle=False)
        trace_id = data["trace_id"]
        y = data["y"]
        ar = data["ar_set"]
        assert (y >= 0).all() and (ar >= 0).all(), (
            f"{label}: negative label on TRAIN features (held-out sentinel?) — "
            "plot_embeddings is train-only by design (§9)."
        )
        act_labels = [str(l) for l in data["labels"]]
        arset_labels = [str(l) for l in data["arset_labels"]]

        rng = np.random.default_rng(seed)
        keep = []
        for tid in sorted(set(trace_id.tolist())):
            idx = np.flatnonzero(trace_id == tid)
            if len(idx) > windows_per_trace:
                idx = rng.choice(idx, size=windows_per_trace, replace=False)
            keep.append(idx)
        keep = np.concatenate(keep)

        x = data["features"][keep].astype(np.float64)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        n_comp = min(pca_components, x.shape[0] - 1, x.shape[1])
        x_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(x)
        emb = TSNE(
            n_components=2, perplexity=perplexity, random_state=seed, init="pca",
        ).fit_transform(x_pca)

        for col, (values, all_labels, title) in enumerate((
            (y[keep], act_labels, "by activity"),
            (ar[keep], arset_labels, "by AR-set"),
        )):
            ax = axes[row_idx][col]
            for k, cls_name in enumerate(all_labels):
                m = values == k
                if not m.any():
                    continue
                ax.scatter(
                    emb[m, 0], emb[m, 1], s=10, alpha=0.75, linewidths=0,
                    color=PALETTE[k % len(PALETTE)], label=cls_name,
                )
            ax.legend(fontsize=7, frameon=False, markerscale=1.5, ncol=2)
            ax.set_title(f"{label} — {title}", fontsize=10, color=_INK)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(_GRID)

    fig.suptitle(
        f"t-SNE of train features (PCA-{pca_components}, perplexity {perplexity}, "
        f"seed {seed}, {windows_per_trace} windows/trace)", fontsize=11, color=_INK,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _demo(argv: list[str]) -> None:  # pragma: no cover
    """`python -m sharp_har.viz <run_dir>`: quick local check."""
    fig = plot_history(argv[0])
    fig.savefig(Path(argv[0]) / "history.png", dpi=150, bbox_inches="tight")
    print("wrote", Path(argv[0]) / "history.png")


if __name__ == "__main__":  # pragma: no cover
    import sys

    _demo(sys.argv[1:])
