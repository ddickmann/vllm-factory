"""Publication-ready chart generator with brand palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .results import BenchResult

PALETTE = {
    "vllm_factory": "#7C3AED",
    "vanilla": "#9CA3AF",
    "accent": "#10B981",
    "accent_warm": "#F59E0B",
    "bg": "#FFFFFF",
    "text": "#1F2937",
    "text_light": "#6B7280",
    "grid": "#F3F4F6",
    "pass_green": "#059669",
    "warn_amber": "#D97706",
}

FONT_FAMILIES = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"]
MODE_STYLES = {
    "saturate": {"color": PALETTE["vllm_factory"], "marker": "o", "label": "vLLM Saturate"},
    "staggered": {"color": PALETTE["accent"], "marker": "s", "label": "vLLM Staggered"},
}


def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILIES,
        "font.size": 11,
        "axes.facecolor": PALETTE["bg"],
        "figure.facecolor": PALETTE["bg"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "text.color": PALETTE["text"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
    })


def _save(fig, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        path = output_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def _display_name(plugin: str) -> str:
    names = {
        "lfm2_colbert": "LFM2-ColBERT",
        "mt5_gliner": "MT5 GLiNER",
        "embeddinggemma": "EmbeddingGemma",
        "moderncolbert": "ModernColBERT",
        "colbert_zero": "ColBERT-Zero",
        "colqwen3": "ColQwen3",
        "collfm2": "ColLFM2",
        "nemotron_colembed": "Nemotron ColEmbed",
        "mmbert_gliner": "MMBert GLiNER",
        "deberta_gliner": "DeBERTa GLiNER",
        "deberta_gliner2": "DeBERTa GLiNER2",
        "deberta_gliner_linker": "GLiNER Linker",
        "modernbert_gliner_rerank": "GLiNER Rerank",
    }
    return names.get(plugin, plugin)


def chart_parity(results: list[BenchResult], output_dir: Path):
    _apply_style()

    results = sorted(results, key=lambda r: r.plugin)

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.8), 3))
    ax.set_xlim(0, len(results))
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, r in enumerate(results):
        cx = i + 0.5
        name = _display_name(r.plugin)
        score = r.parity_score
        metric = r.parity_metric.replace("_", " ").title()

        color = PALETTE["pass_green"] if score >= 0.95 else PALETTE["warn_amber"]

        ax.add_patch(plt.Rectangle(
            (i + 0.05, 0.15), 0.9, 0.7,
            facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.5,
            transform=ax.transData, zorder=2,
        ))

        ax.text(cx, 0.72, name, ha="center", va="center",
                fontsize=9, fontweight="bold", color=PALETTE["text"])
        ax.text(cx, 0.50, f"{score:.4f}", ha="center", va="center",
                fontsize=16, fontweight="bold", color=color)
        ax.text(cx, 0.30, metric, ha="center", va="center",
                fontsize=8, color=PALETTE["text_light"])

    fig.suptitle("Parity: vLLM Factory vs Reference", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    _save(fig, output_dir, "parity")
    print("  Chart saved: parity.svg / .png")


# ---------------------------------------------------------------------------
# Summary chart: best throughput factor by model and mode
# ---------------------------------------------------------------------------

def chart_best_throughput(results: list[BenchResult], output_dir: Path, mode: str):
    _apply_style()

    best_points = []
    for result in results:
        best = result.best_sweep(mode)
        if best is not None:
            best_points.append((result, best))

    if not best_points:
        return

    best_points.sort(key=lambda item: item[1].throughput_factor, reverse=True)
    labels = [_display_name(result.plugin) for result, _ in best_points]
    values = [best.throughput_factor for _, best in best_points]
    colors = [MODE_STYLES[mode]["color"]] * len(best_points)

    fig, ax = plt.subplots(figsize=(max(8, len(best_points) * 2.0), 5))
    x = np.arange(len(best_points))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.6, zorder=3)

    for bar, (_, best) in zip(bars, best_points):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"{best.throughput_factor:.1f}x @ {best.concurrency}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=PALETTE["text"],
        )

    ax.set_ylabel("Best throughput factor vs vanilla")
    ax.set_title(f"Peak Throughput Factors ({mode.title()})", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    _save(fig, output_dir, f"throughput_{mode}")
    print(f"  Chart saved: throughput_{mode}.svg / .png")


def _baseline_series(result: BenchResult) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    per_level = {}
    for sweep in result.sweeps:
        per_level.setdefault(sweep.concurrency, sweep)
    levels = sorted(per_level)
    vanilla_req = [per_level[level].vanilla_req_per_s for level in levels]
    vanilla_p50 = [per_level[level].vanilla_p50_ms for level in levels]
    vanilla_p99 = [per_level[level].vanilla_p99_ms for level in levels]
    vanilla_latency_factor = [per_level[level].latency_factor for level in levels]
    return levels, vanilla_req, vanilla_p50, vanilla_p99, vanilla_latency_factor


def _get_sweep(result: BenchResult, mode: str, concurrency: int):
    for sweep in result.sweeps:
        if sweep.mode == mode and sweep.concurrency == concurrency:
            return sweep
    return None


def chart_social_batching_card(
    result: BenchResult,
    output_dir: Path,
    levels: tuple[int, ...] = (16, 32, 64),
):
    _apply_style()

    selected = []
    for level in levels:
        saturate = _get_sweep(result, "saturate", level)
        staggered = _get_sweep(result, "staggered", level)
        if saturate is None or staggered is None:
            continue
        selected.append((level, saturate, staggered))

    if not selected:
        return

    fig, ax = plt.subplots(figsize=(10.8, 8.2))
    fig.subplots_adjust(top=0.72, bottom=0.16, left=0.10, right=0.97)

    levels = [level for level, _, _ in selected]
    vanilla_vals = [sat.vanilla_req_per_s for _, sat, _ in selected]
    staggered_vals = [stg.vllm_req_per_s for _, _, stg in selected]
    saturate_vals = [sat.vllm_req_per_s for _, sat, _ in selected]

    x = np.arange(len(levels))
    width = 0.22

    bars_vanilla = ax.bar(
        x - width,
        vanilla_vals,
        width,
        label="Vanilla",
        color=PALETTE["vanilla"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    bars_staggered = ax.bar(
        x,
        staggered_vals,
        width,
        label="vLLM Staggered",
        color=PALETTE["accent"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    bars_saturate = ax.bar(
        x + width,
        saturate_vals,
        width,
        label="vLLM Saturated",
        color=PALETTE["vllm_factory"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ymax = max(saturate_vals + staggered_vals + vanilla_vals) * 1.24
    ax.set_ylim(0, ymax)

    for bars in (bars_vanilla, bars_staggered, bars_saturate):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.015,
                f"{bar.get_height():.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=PALETTE["text"],
            )

    for idx, (_, sat, stg) in enumerate(selected):
        ax.text(
            x[idx],
            stg.vllm_req_per_s + ymax * 0.07,
            f"{stg.throughput_factor:.1f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=PALETTE["accent"],
        )
        ax.text(
            x[idx] + width,
            sat.vllm_req_per_s + ymax * 0.07,
            f"{sat.throughput_factor:.1f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=PALETTE["vllm_factory"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"BS {level}" for level in levels], fontsize=12, fontweight="bold")
    ax.set_ylabel("Requests / second", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.02))

    name = _display_name(result.plugin)
    fig.text(
        0.10,
        0.95,
        f"{name}: Throughput With Batching",
        fontsize=21,
        fontweight="bold",
        color=PALETTE["text"],
    )
    fig.text(
        0.10,
        0.915,
        "Vanilla vs vLLM under staggered async arrivals and saturated batching",
        fontsize=11.5,
        color=PALETTE["text_light"],
    )

    fig.text(
        0.10,
        0.875,
        f"Model: {result.model_id}",
        fontsize=10.5,
        color=PALETTE["text_light"],
    )
    ds_label = result.dataset_label or "synthetic benchmark data"
    fig.text(
        0.10,
        0.848,
        f"Dataset: {ds_label}",
        fontsize=10.5,
        color=PALETTE["text_light"],
    )

    badge_bbox = dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor=PALETTE["grid"])
    fig.text(
        0.72,
        0.92,
        f"GPU\n{result.gpu}",
        ha="left",
        va="top",
        fontsize=10.5,
        color=PALETTE["text"],
        bbox=badge_bbox,
    )
    fig.text(
        0.72,
        0.84,
        f"Parity\n{result.parity_score:.4f}",
        ha="left",
        va="top",
        fontsize=10.5,
        color=PALETTE["pass_green"] if result.parity_score >= 0.95 else PALETTE["warn_amber"],
        bbox=badge_bbox,
    )

    latency_wins = []
    for level, sat, stg in selected:
        if sat.latency_factor > 1.0:
            latency_wins.append(f"saturated BS {level}: {sat.latency_factor:.1f}x p50")
        if stg.latency_factor > 1.0:
            latency_wins.append(f"staggered BS {level}: {stg.latency_factor:.1f}x p50")

    footer = "Numbers above vLLM bars show throughput gain vs vanilla."
    if latency_wins:
        footer += " Latency wins: " + " | ".join(latency_wins)
    fig.text(0.10, 0.08, footer, fontsize=10.5, color=PALETTE["text_light"])

    _save(fig, output_dir, f"social_{result.plugin}")
    print(f"  Chart saved: social_{result.plugin}.svg / .png")


def chart_model_card(result: BenchResult, output_dir: Path):
    _apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    name = _display_name(result.plugin)
    levels, vanilla_req, vanilla_p50, vanilla_p99, _ = _baseline_series(result)

    ax = axes[0, 0]
    ax.plot(
        levels,
        vanilla_req,
        color=PALETTE["vanilla"],
        marker="o",
        linestyle="--",
        linewidth=2.0,
        label="Vanilla",
        zorder=3,
    )
    for mode in result.modes:
        sweeps = result.sweeps_for_mode(mode)
        if not sweeps:
            continue
        style = MODE_STYLES[mode]
        ax.plot(
            [s.concurrency for s in sweeps],
            [s.vllm_req_per_s for s in sweeps],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.3,
            label=style["label"],
            zorder=4,
        )
    ax.set_title("Throughput vs Batch Size", fontweight="bold")
    ax.set_xlabel("Batch size / concurrency")
    ax.set_ylabel("Requests / second")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    ax = axes[0, 1]
    ax.plot(
        levels,
        vanilla_p50,
        color=PALETTE["vanilla"],
        marker="o",
        linestyle="--",
        linewidth=2.0,
        label="Vanilla p50",
        zorder=3,
    )
    for mode in result.modes:
        sweeps = result.sweeps_for_mode(mode)
        if not sweeps:
            continue
        style = MODE_STYLES[mode]
        ax.plot(
            [s.concurrency for s in sweeps],
            [s.vllm_p50_ms for s in sweeps],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.3,
            label=f"{style['label']} p50",
            zorder=4,
        )
    ax.set_title("P50 Latency vs Batch Size", fontweight="bold")
    ax.set_xlabel("Batch size / concurrency")
    ax.set_ylabel("Latency (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    ax = axes[1, 0]
    for mode in result.modes:
        sweeps = result.sweeps_for_mode(mode)
        if not sweeps:
            continue
        style = MODE_STYLES[mode]
        ax.plot(
            [s.concurrency for s in sweeps],
            [s.throughput_factor for s in sweeps],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.3,
            label=f"{mode.title()} throughput factor",
            zorder=4,
        )
    ax.axhline(1.0, color=PALETTE["grid"], linewidth=1.0)
    ax.set_title("Throughput Factor vs Vanilla", fontweight="bold")
    ax.set_xlabel("Batch size / concurrency")
    ax.set_ylabel("Factor (x)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    ax = axes[1, 1]
    ax.plot(
        levels,
        vanilla_p99,
        color=PALETTE["vanilla"],
        marker="o",
        linestyle="--",
        linewidth=2.0,
        label="Vanilla p99",
        zorder=3,
    )
    for mode in result.modes:
        sweeps = result.sweeps_for_mode(mode)
        if not sweeps:
            continue
        style = MODE_STYLES[mode]
        ax.plot(
            [s.concurrency for s in sweeps],
            [s.vllm_p99_ms for s in sweeps],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.3,
            label=f"{style['label']} p99",
            zorder=4,
        )
    ax.set_title("P99 Latency vs Batch Size", fontweight="bold")
    ax.set_xlabel("Batch size / concurrency")
    ax.set_ylabel("Latency (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"{name}  —  {result.model_id}", fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.02,
        f"Parity: {result.parity_score:.4f} ({result.parity_metric.replace('_', ' ')})  |  GPU: {result.gpu}",
        ha="center",
        fontsize=10,
        color=PALETTE["pass_green"] if result.parity_score >= 0.95 else PALETTE["warn_amber"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))

    _save(fig, output_dir, f"card_{result.plugin}")
    print(f"  Chart saved: card_{result.plugin}.svg / .png")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_charts(results: list[BenchResult], output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        print("No results to chart.")
        return

    print(f"\nGenerating charts for {len(results)} results -> {output_dir}/\n")

    chart_parity(results, output_dir)
    chart_best_throughput(results, output_dir, "saturate")
    chart_best_throughput(results, output_dir, "staggered")

    for r in results:
        chart_model_card(r, output_dir)
        chart_social_batching_card(r, output_dir)

    print(f"\nAll charts written to {output_dir}/")
