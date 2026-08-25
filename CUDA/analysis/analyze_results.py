#!/usr/bin/env python3

"""Analizza i tempi dei kernel CUDA naïve e tiled/shared.

Il benchmark contiene una sola metrica: il tempo della regione dei kernel,
misurato sul lato host tra il lancio e cudaDeviceSynchronize(). Trasferimenti
H2D/D2H, allocazioni e preparazione delle immagini non fanno parte dei dati.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"

CONFIG_COLUMNS = [
    "mode", "implementation", "block_dim", "sm_count",
    "use_shared", "rows_per_block", "se_size", "image_rows", "image_cols",
    "batch", "operation",
]
REQUIRED_COLUMNS = {"run_id", "run_index", "seconds", *CONFIG_COLUMNS}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRID_COLOR,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.frameon": False,
})

_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
}


def t95(n: int) -> float:
    dof = int(n) - 1
    if dof < 1:
        return 0.0
    if dof in _T95:
        return _T95[dof]
    if dof > 30:
        return 1.960
    return _T95[min(k for k in _T95 if k > dof)]


def load_data(path: Path, requested_run: str | None) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise SystemExit(f"File non trovato: {path}")
    raw = pd.read_csv(path, dtype={"run_id": str})
    missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise SystemExit(f"CSV non compatibile ({path}): mancano {missing}")
    if raw.empty:
        raise SystemExit(f"CSV vuoto: {path}")

    run_ids = sorted(raw["run_id"].dropna().unique())
    run_id = requested_run or run_ids[-1]
    if run_id not in run_ids:
        raise SystemExit(f"run_id {run_id!r} non trovato. Disponibili: {run_ids}")

    selected = raw[raw["run_id"] == run_id].copy()
    modes = set(selected["mode"].dropna().unique())
    implementations = set(selected["implementation"].dropna().unique())
    if modes != {"problem_size"} or implementations != {"cuda"}:
        raise SystemExit(
            f"Il run {run_id!r} appartiene al vecchio benchmark "
            f"(mode={sorted(modes)}, implementation={sorted(implementations)}). "
            "Esegui nuovamente CUDA/main: sono accettati soltanto i nuovi run "
            "problem_size/CUDA con griglia naturale."
        )

    selected["image_size"] = (
        selected["image_rows"].astype(str) + "x" + selected["image_cols"].astype(str)
    )
    selected["image_area"] = selected["image_rows"] * selected["image_cols"]
    selected["image_megapixels"] = selected["image_area"] / 1e6
    selected["memory"] = selected["use_shared"].map({0: "naive", 1: "shared"})
    if selected["memory"].isna().any():
        raise SystemExit("use_shared deve contenere soltanto 0 o 1.")
    naive_r = set(selected.loc[selected["use_shared"] == 0, "rows_per_block"].unique())
    if naive_r != {1}:
        raise SystemExit(
            f"Il run {run_id!r} usa rows_per_block={sorted(naive_r)} anche nel global: "
            "è il precedente kernel row-coarsened, non la nuova baseline naïve. "
            "Esegui nuovamente CUDA/main."
        )
    return selected, run_id


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    groups = CONFIG_COLUMNS + ["image_size", "image_area", "image_megapixels", "memory"]
    stats = (
        df.groupby(groups, dropna=False)["seconds"]
        .agg(mean="mean", std="std", min="min", max="max", samples="count")
        .reset_index()
    )
    stats["ci95"] = [
        t95(n) * sd / n**0.5 if n > 1 and pd.notna(sd) else 0.0
        for sd, n in zip(stats["std"], stats["samples"])
    ]
    return stats


def ratio_with_ci(a, a_ci, b, b_ci):
    ratio = a / b
    ci = ratio * ((a_ci / a) ** 2 + (b_ci / b) ** 2) ** 0.5
    return ratio, ci


def compute_speedup(stats: pd.DataFrame) -> pd.DataFrame:
    # rows_per_block è 1 nella baseline naïve e R nello shared: non fa parte
    # della chiave, perché descrive proprio una differenza fra le implementazioni.
    keys = [
        "operation", "block_dim", "sm_count", "se_size",
        "image_rows", "image_cols", "image_size", "image_area",
        "image_megapixels", "batch",
    ]
    columns = keys + ["mean", "ci95", "samples"]
    naive = stats[stats["memory"] == "naive"][columns].rename(columns={
        "mean": "naive_mean", "ci95": "naive_ci95",
        "samples": "naive_samples",
    })
    shared = stats[stats["memory"] == "shared"][columns].rename(columns={
        "mean": "shared_mean", "ci95": "shared_ci95",
        "samples": "shared_samples",
    })
    result = naive.merge(shared, on=keys, how="inner", validate="one_to_one")
    if result.empty:
        raise SystemExit("Nessuna coppia naïve/shared confrontabile trovata.")
    result["speedup_naive_over_shared"], result["speedup_ci95"] = ratio_with_ci(
        result["naive_mean"], result["naive_ci95"],
        result["shared_mean"], result["shared_ci95"],
    )
    return result


def compute_se_cost(stats: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "operation", "use_shared", "memory", "block_dim", "sm_count",
        "rows_per_block", "image_rows", "image_cols", "image_size", "image_area",
        "image_megapixels", "batch",
    ]
    base_se = int(stats["se_size"].min())
    base = stats[stats["se_size"] == base_se][keys + ["mean", "ci95"]].rename(
        columns={"mean": "se_base_mean", "ci95": "se_base_ci95"}
    )
    result = stats.merge(base, on=keys, how="left", validate="many_to_one")
    if result["se_base_mean"].isna().any():
        raise SystemExit(f"Configurazioni senza riferimento SE {base_se}x{base_se}.")
    result["relative_to_smallest_se"], result["relative_to_smallest_se_ci95"] = ratio_with_ci(
        result["mean"], result["ci95"], result["se_base_mean"], result["se_base_ci95"]
    )
    result.loc[result["se_size"] == base_se, "relative_to_smallest_se"] = 1.0
    result.loc[result["se_size"] == base_se, "relative_to_smallest_se_ci95"] = 0.0
    result["base_se_size"] = base_se
    result["dense_window_ratio"] = (result["se_size"] / base_se) ** 2
    return result


def operations(df: pd.DataFrame) -> list[str]:
    preferred = ["erosion", "opening"]
    present = list(df["operation"].drop_duplicates())
    return [op for op in preferred if op in present] + sorted(set(present) - set(preferred))


def implementation_label(memory: str) -> str:
    return "naïve (global)" if memory == "naive" else "tiled (shared)"


def machine_label(stats: pd.DataFrame, override: str | None) -> str:
    if override:
        return override
    sms = sorted(int(x) for x in stats["sm_count"].dropna().unique() if int(x) > 0)
    return f"GPU CUDA ({', '.join(map(str, sms))} SM)" if sms else "GPU CUDA"


def data_caption(stats: pd.DataFrame, machine: str, extra: str = "") -> str:
    batches = sorted(int(x) for x in stats["batch"].unique())
    samples = sorted(int(x) for x in stats["samples"].unique())
    parts = [
        f"batch {', '.join(map(str, batches))}",
        f"{min(samples)}-{max(samples)} campioni per configurazione",
        "timer host + cudaDeviceSynchronize; trasferimenti esclusi",
    ]
    if extra:
        parts.append(extra)
    return "Dati: " + " · ".join(parts) + f" — {machine}"


def style_axis(ax, xlabel: str, ylabel: str, title: str, log_y: bool = False):
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if log_y:
        ax.set_yscale("log")


def new_facet_figure(title: str, subtitle: str, caption: str, columns: list[str],
                     row_labels: list[str], facet_w: float = 4.8, facet_h: float = 3.8):
    rows = len(row_labels)
    fig, axes = plt.subplots(
        rows, len(columns), figsize=(facet_w * len(columns), facet_h * rows), squeeze=False
    )
    wrapped = textwrap.fill(caption, width=max(80, int(facet_w * len(columns) * 16)))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.85, bottom=0.10,
                        wspace=0.34, hspace=0.48)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.972)
    fig.text(0.5, 0.94, subtitle, ha="center", fontsize=9, color=INK_SECONDARY)
    fig.text(0.5, 0.018, wrapped, ha="center", va="bottom", fontsize=7.5, color=INK_MUTED)
    for row, label in enumerate(row_labels):
        pos = axes[row][0].get_position()
        fig.text(0.018, pos.y0 + pos.height / 2, label, rotation=90,
                 va="center", ha="center", fontsize=10.5, fontweight="bold")
    return fig, axes


def add_legend(fig, handles, labels, columns=5):
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
                   ncol=min(columns, len(labels)), fontsize=8)


def plot_time_by_problem_size(stats: pd.DataFrame, out: Path, machine: str):
    ops = operations(stats)
    se_sizes = sorted(int(x) for x in stats["se_size"].unique())
    fig, axes = new_facet_figure(
        "Tempo dei kernel al crescere del problema",
        "Confronto fra accesso diretto in global memory e tiling in shared memory.",
        data_caption(stats, machine), ops, [f"SE {s}x{s}" for s in se_sizes],
    )
    handles, labels = [], []
    for row, se_size in enumerate(se_sizes):
        for col, operation in enumerate(ops):
            ax = axes[row][col]
            subset = stats[(stats["se_size"] == se_size) &
                           (stats["operation"] == operation)]
            for index, memory in enumerate(("naive", "shared")):
                line = subset[subset["memory"] == memory].sort_values("image_area")
                if line.empty:
                    continue
                artist = ax.errorbar(
                    line["image_megapixels"], line["mean"], yerr=line["ci95"],
                    marker="o", linewidth=1.8, capsize=3, color=CATEGORICAL[index],
                    label=implementation_label(memory),
                )
                if row == 0 and col == 0:
                    handles.append(artist)
                    labels.append(implementation_label(memory))
            style_axis(ax, "dimensione immagine [MPixel]", "secondi / immagine",
                       operation, log_y=True)
    add_legend(fig, handles, labels)
    fig.savefig(out / "kernel_time_by_problem_size.png", dpi=180)
    plt.close(fig)


def plot_time_by_se_size(stats: pd.DataFrame, out: Path, machine: str):
    ops = operations(stats)
    memories = [m for m in ("naive", "shared") if m in set(stats["memory"])]
    problems = stats[["image_size", "image_area"]].drop_duplicates().sort_values("image_area")
    fig, axes = new_facet_figure(
        "Tempo dei kernel al crescere dell'elemento strutturante",
        "Ogni curva rappresenta una diversa dimensione del problema.",
        data_caption(stats, machine), ops, [implementation_label(m) for m in memories],
    )
    handles, labels = [], []
    for row, memory in enumerate(memories):
        for col, operation in enumerate(ops):
            ax = axes[row][col]
            subset = stats[(stats["memory"] == memory) &
                           (stats["operation"] == operation)]
            for index, problem in enumerate(problems.itertuples(index=False)):
                line = subset[subset["image_size"] == problem.image_size].sort_values("se_size")
                if line.empty:
                    continue
                label = problem.image_size
                artist = ax.errorbar(
                    line["se_size"], line["mean"], yerr=line["ci95"], marker="o",
                    linewidth=1.5, capsize=3, color=CATEGORICAL[index % len(CATEGORICAL)],
                    label=label,
                )
                if row == 0 and col == 0:
                    handles.append(artist)
                    labels.append(label)
            ax.set_xticks(sorted(int(x) for x in stats["se_size"].unique()))
            style_axis(ax, "lato dell'elemento strutturante", "secondi / immagine",
                       operation, log_y=True)
    add_legend(fig, handles, labels, columns=4)
    fig.savefig(out / "kernel_time_by_se_size.png", dpi=180)
    plt.close(fig)


def plot_speedup(speedup: pd.DataFrame, out: Path, machine: str):
    ops = operations(speedup)
    se_sizes = sorted(int(x) for x in speedup["se_size"].unique())
    fig, axes = new_facet_figure(
        "Speedup relativo tiled rispetto a naïve",
        "S = T_naive / T_shared; sopra 1 il tiling è vantaggioso.",
        f"Confronto a parità di immagine, batch, thread per blocco e SE — {machine}", ops,
        [f"SE {s}x{s}" for s in se_sizes],
    )
    for row, se_size in enumerate(se_sizes):
        for col, operation in enumerate(ops):
            ax = axes[row][col]
            line = speedup[(speedup["se_size"] == se_size) &
                           (speedup["operation"] == operation)].sort_values("image_area")
            ax.errorbar(
                line["image_megapixels"], line["speedup_naive_over_shared"],
                yerr=line["speedup_ci95"], marker="o", linewidth=1.8, capsize=3,
                color=CATEGORICAL[2],
            )
            ax.axhline(1.0, color=INK_MUTED, linestyle="--", linewidth=1)
            style_axis(ax, "dimensione immagine [MPixel]", "speedup", operation)
    fig.savefig(out / "naive_vs_shared_speedup.png", dpi=180)
    plt.close(fig)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=script_dir.parent / "experiment_run" / "kernel_results.csv",
    )
    parser.add_argument("--run-id", help="run_id da analizzare; default: il più recente")
    parser.add_argument("--out", type=Path, default=script_dir / "output")
    parser.add_argument("--machine", help="etichetta descrittiva della GPU")
    return parser.parse_args()


def main():
    args = parse_args()
    samples, run_id = load_data(args.input, args.run_id)
    stats = aggregate(samples)
    speedup = compute_speedup(stats)
    se_cost = compute_se_cost(stats)
    machine = machine_label(stats, args.machine)

    args.out.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.out / "summary_stats.csv", index=False)
    speedup.to_csv(args.out / "naive_vs_shared_speedup.csv", index=False)
    se_cost.to_csv(args.out / "se_size_comparison.csv", index=False)

    plot_time_by_problem_size(stats, args.out, machine)
    plot_time_by_se_size(stats, args.out, machine)
    plot_speedup(speedup, args.out, machine)

    print(f"Analizzato run_id={run_id!r}: {len(samples)} campioni, {len(stats)} configurazioni.")
    print(f"Risultati scritti in {args.out.resolve()}")


if __name__ == "__main__":
    main()
