from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit(
        "plotly is required. Run with: uv run --with plotly python compare_benchmark_reports.py"
    ) from exc


BG = "#f1f5f9"
CARD = "#ffffff"
TEXT = "#1e293b"
MUTED = "#64748b"
GRID = "#e2e8f0"
BASELINE = "#6366f1"
CURRENT = "#10b981"
POS = "#22c55e"
NEG = "#ef4444"
ACCENT = "#3b82f6"
HEADER_BG = "#1e293b"

CONCURRENCY_ORDER = [1, 8, 16, 32, 64]


@dataclass(frozen=True)
class Report:
    label: str
    path: Path
    data: dict[str, Any]

    @property
    def overall(self) -> dict[str, Any]:
        return self.data["overall_summary"]

    @property
    def scenarios(self) -> list[dict[str, Any]]:
        return self.data["scenarios"]

    @property
    def scenario_map(self) -> dict[str, dict[str, Any]]:
        return {item["scenario"]: item for item in self.scenarios}


def infer_label(path: Path) -> str:
    name = path.stem.lower()
    for key in ("minimal", "original", "baseline", "bloated", "fixed"):
        if key in name:
            if key == "original":
                return "baseline"
            elif key == "minimal":
                return "cached"
            return key
    return path.stem


def load_report(path: Path) -> Report:
    return Report(label=infer_label(path), path=path, data=json.loads(path.read_text()))


def delta_pct(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100.0


def build_latency_table(baseline: Report, minimal: Report) -> go.Table:
    """Build latency comparison table - shows mixed+long @c64 (batch scenario)."""
    base_ml = get_mixed_long_by_concurrency(baseline)
    mini_ml = get_mixed_long_by_concurrency(minimal)

    # Get c64 mixed+long data (the batch scenario)
    base_c64 = base_ml.get(64, {})
    mini_c64 = mini_ml.get(64, {})

    base_overall = baseline.overall
    mini_overall = minimal.overall

    rows = [
        [
            "Batch c64 Mean",
            f"{base_c64.get('mean_ms', 0):.0f}ms",
            f"{mini_c64.get('mean_ms', 0):.0f}ms",
            f"{delta_pct(mini_c64.get('mean_ms', 0), base_c64.get('mean_ms', 0)):+.1f}%",
        ],
        [
            "Batch c64 P99",
            f"{base_c64.get('p99_ms', 0):.0f}ms",
            f"{mini_c64.get('p99_ms', 0):.0f}ms",
            f"{delta_pct(mini_c64.get('p99_ms', 0), base_c64.get('p99_ms', 0)):+.1f}%",
        ],
        [
            "Overall Mean",
            f"{base_overall['mean_ms']:.0f}",
            f"{mini_overall['mean_ms']:.0f}",
            f"{delta_pct(mini_overall['mean_ms'], base_overall['mean_ms']):+.1f}%",
        ],
        [
            "Overall P95",
            f"{base_overall['p95_ms']:.0f}",
            f"{mini_overall['p95_ms']:.0f}",
            f"{delta_pct(mini_overall['p95_ms'], base_overall['p95_ms']):+.1f}%",
        ],
        [
            "Overall P99",
            f"{base_overall['p99_ms']:.0f}",
            f"{mini_overall['p99_ms']:.0f}",
            f"{delta_pct(mini_overall['p99_ms'], base_overall['p99_ms']):+.1f}%",
        ],
    ]

    return go.Table(
        header=dict(
            values=["<b>Latency</b>", "<b>Baseline</b>", "<b>Cached</b>", "<b>Δ</b>"],
            fill_color=HEADER_BG,
            font=dict(color="white", size=10),
            align=["left", "center", "center", "center"],
            height=24,
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=CARD,
            align=["left", "center", "center", "center"],
            height=20,
            font=dict(color=TEXT, size=10),
            line_color=GRID,
        ),
    )


def build_throughput_table(baseline: Report, minimal: Report) -> go.Table:
    """Build throughput comparison table - shows mixed+long @c64 (batch scenario)."""
    base_ml = get_mixed_long_by_concurrency(baseline)
    mini_ml = get_mixed_long_by_concurrency(minimal)

    # Get c64 mixed+long data (the batch scenario)
    base_c64 = base_ml.get(64, {})
    mini_c64 = mini_ml.get(64, {})

    base_overall = baseline.overall
    mini_overall = minimal.overall

    rows = [
        [
            "Batch c64 Req Tok/s",
            f"{base_c64.get('req_tok_s', 0):.0f}",
            f"{mini_c64.get('req_tok_s', 0):.0f}",
            f"{delta_pct(mini_c64.get('req_tok_s', 0), base_c64.get('req_tok_s', 0)):+.1f}%",
        ],
        [
            "Batch c64 P99",
            f"{base_c64.get('p99_ms', 0):.0f}ms",
            f"{mini_c64.get('p99_ms', 0):.0f}ms",
            f"{delta_pct(mini_c64.get('p99_ms', 0), base_c64.get('p99_ms', 0)):+.1f}%",
        ],
        [
            "Overall Req/s",
            f"{base_overall['requests_per_sec']:.2f}",
            f"{mini_overall['requests_per_sec']:.2f}",
            f"{delta_pct(mini_overall['requests_per_sec'], base_overall['requests_per_sec']):+.1f}%",
        ],
        [
            "Overall Req Tok/s",
            f"{base_overall['request_tokens_per_sec_exact']:.0f}",
            f"{mini_overall['request_tokens_per_sec_exact']:.0f}",
            f"{delta_pct(mini_overall['request_tokens_per_sec_exact'], base_overall['request_tokens_per_sec_exact']):+.1f}%",
        ],
        [
            "Overall Text Tok/s",
            f"{base_overall['text_tokens_per_sec_exact']:.0f}",
            f"{mini_overall['text_tokens_per_sec_exact']:.0f}",
            f"{delta_pct(mini_overall['text_tokens_per_sec_exact'], base_overall['text_tokens_per_sec_exact']):+.1f}%",
        ],
    ]

    return go.Table(
        header=dict(
            values=["<b>Throughput</b>", "<b>Baseline</b>", "<b>Cached</b>", "<b>Δ</b>"],
            fill_color=HEADER_BG,
            font=dict(color="white", size=10),
            align=["left", "center", "center", "center"],
            height=24,
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=CARD,
            align=["left", "center", "center", "center"],
            height=20,
            font=dict(color=TEXT, size=10),
            line_color=GRID,
        ),
    )


def build_cost_table(baseline: Report, minimal: Report) -> go.Table:
    """Build cost comparison table - shows mixed+long @c64 (batch scenario)."""
    base_ml = get_mixed_long_by_concurrency(baseline)
    mini_ml = get_mixed_long_by_concurrency(minimal)

    # Get c64 mixed+long data (the batch scenario)
    base_c64 = base_ml.get(64, {})
    mini_c64 = mini_ml.get(64, {})

    base_overall = baseline.overall
    mini_overall = minimal.overall

    rows = [
        [
            "Batch c64 $/1M",
            f"${base_c64.get('cost_per_mtok', 0):.5f}",
            f"${mini_c64.get('cost_per_mtok', 0):.5f}",
            f"{delta_pct(mini_c64.get('cost_per_mtok', 0), base_c64.get('cost_per_mtok', 0)):+.1f}%",
        ],
        [
            "Batch c64 Latency",
            f"{base_c64.get('mean_ms', 0):.0f}ms",
            f"{mini_c64.get('mean_ms', 0):.0f}ms",
            f"{delta_pct(mini_c64.get('mean_ms', 0), base_c64.get('mean_ms', 0)):+.1f}%",
        ],
        [
            "Overall $/1M",
            f"${base_overall['cost_per_million_request_tokens_exact']:.5f}",
            f"${mini_overall['cost_per_million_request_tokens_exact']:.5f}",
            f"{delta_pct(mini_overall['cost_per_million_request_tokens_exact'], base_overall['cost_per_million_request_tokens_exact']):+.1f}%",
        ],
        [
            "Success",
            f"{int(base_overall['ok'])}",
            f"{int(mini_overall['ok'])}",
            f"{int(mini_overall['ok'] - base_overall['ok']):+d}",
        ],
        [
            "Errors",
            f"{int(base_overall['errors'])}",
            f"{int(mini_overall['errors'])}",
            f"{int(mini_overall['errors'] - base_overall['errors']):+d}",
        ],
    ]

    return go.Table(
        header=dict(
            values=["<b>Cost & Reliability</b>", "<b>Baseline</b>", "<b>Cached</b>", "<b>Δ</b>"],
            fill_color=HEADER_BG,
            font=dict(color="white", size=10),
            align=["left", "center", "center", "center"],
            height=24,
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=CARD,
            align=["left", "center", "center", "center"],
            height=20,
            font=dict(color=TEXT, size=10),
            line_color=GRID,
        ),
    )


def get_mixed_long_by_concurrency(report: Report) -> dict[int, dict[str, float]]:
    """Get mixed+long scenario data grouped by concurrency."""
    result = {}
    for scenario in report.scenarios:
        if scenario["schema_mode"] == "mixed" and scenario["text_mode"] == "long":
            conc = int(scenario["concurrency"])
            summary = scenario["summary"]
            result[conc] = {
                "mean_ms": float(summary["mean_ms"]),
                "p99_ms": float(summary["p99_ms"]),
                "req_tok_s": float(summary["request_tokens_per_sec_exact"]),
                "cost_per_mtok": float(summary["cost_per_million_request_tokens_exact"]),
            }
    return result


def get_batch64_data(
    report: Report, dimension: str, order: list[str]
) -> dict[str, dict[str, float]]:
    """Get batch 64 data grouped by schema_mode or text_mode."""
    result = {}
    for scenario in report.scenarios:
        if int(scenario["concurrency"]) == 64:
            key = scenario[dimension]
            summary = scenario["summary"]
            result[key] = {
                "mean_ms": float(summary["mean_ms"]),
                "p95_ms": float(summary["p95_ms"]),
                "p99_ms": float(summary["p99_ms"]),
                "req_tok_s": float(summary["request_tokens_per_sec_exact"]),
                "cost_per_mtok": float(summary["cost_per_million_request_tokens_exact"]),
                "success_rate": float(summary["ok"]) / float(summary["count"]) * 100.0
                if summary["count"]
                else 0.0,
            }
    return result


def build_batch64_schema_table(baseline: Report, minimal: Report) -> go.Table:
    """Build batch 64 cost table for all schema modes."""
    SCHEMA_ORDER = ["entities", "classifications", "relations", "structures", "mixed"]
    base_data = get_batch64_data(baseline, "schema_mode", SCHEMA_ORDER)
    mini_data = get_batch64_data(minimal, "schema_mode", SCHEMA_ORDER)

    rows = []
    for schema in SCHEMA_ORDER:
        if schema in base_data and schema in mini_data:
            b = base_data[schema]
            m = mini_data[schema]
            rows.append(
                [
                    schema,
                    f"${b['cost_per_mtok']:.5f}",
                    f"${m['cost_per_mtok']:.5f}",
                    f"{delta_pct(m['cost_per_mtok'], b['cost_per_mtok']):+.1f}%",
                    f"{b['mean_ms']:.0f}",
                    f"{m['mean_ms']:.0f}",
                ]
            )

    return go.Table(
        header=dict(
            values=[
                "<b>Schema @c64</b>",
                "<b>Base $/1M</b>",
                "<b>Cached $/1M</b>",
                "<b>Δ%</b>",
                "<b>Base ms</b>",
                "<b>Cached ms</b>",
            ],
            fill_color=HEADER_BG,
            font=dict(color="white", size=10),
            align=["left", "center", "center", "center", "center", "center"],
            height=24,
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=CARD,
            align=["left", "center", "center", "center", "center", "center"],
            height=20,
            font=dict(color=TEXT, size=10),
            line_color=GRID,
        ),
    )


def build_batch64_text_table(baseline: Report, minimal: Report) -> go.Table:
    """Build batch 64 cost table for all text modes."""
    TEXT_ORDER = ["short", "varied", "long"]
    base_data = get_batch64_data(baseline, "text_mode", TEXT_ORDER)
    mini_data = get_batch64_data(minimal, "text_mode", TEXT_ORDER)

    rows = []
    for text in TEXT_ORDER:
        if text in base_data and text in mini_data:
            b = base_data[text]
            m = mini_data[text]
            rows.append(
                [
                    text,
                    f"${b['cost_per_mtok']:.5f}",
                    f"${m['cost_per_mtok']:.5f}",
                    f"{delta_pct(m['cost_per_mtok'], b['cost_per_mtok']):+.1f}%",
                    f"{b['mean_ms']:.0f}",
                    f"{m['mean_ms']:.0f}",
                ]
            )

    return go.Table(
        header=dict(
            values=[
                "<b>Text @c64</b>",
                "<b>Base $/1M</b>",
                "<b>Cached $/1M</b>",
                "<b>Δ%</b>",
                "<b>Base ms</b>",
                "<b>Cached ms</b>",
            ],
            fill_color=HEADER_BG,
            font=dict(color="white", size=10),
            align=["left", "center", "center", "center", "center", "center"],
            height=24,
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=CARD,
            align=["left", "center", "center", "center", "center", "center"],
            height=20,
            font=dict(color=TEXT, size=10),
            line_color=GRID,
        ),
    )


def build_mixed_long_cost_table(baseline: Report, minimal: Report) -> go.Table:
    """Build cost breakdown table for mixed+long at all concurrency levels."""
    base_data = get_mixed_long_by_concurrency(baseline)
    mini_data = get_mixed_long_by_concurrency(minimal)

    rows = []
    for conc in CONCURRENCY_ORDER:
        if conc in base_data and conc in mini_data:
            b = base_data[conc]
            m = mini_data[conc]
            rows.append(
                [
                    f"c{conc}",
                    f"${b['cost_per_mtok']:.5f}",
                    f"${m['cost_per_mtok']:.5f}",
                    f"{delta_pct(m['cost_per_mtok'], b['cost_per_mtok']):+.1f}%",
                    f"{b['mean_ms']:.0f}",
                    f"{m['mean_ms']:.0f}",
                ]
            )

    return go.Table(
        header=dict(
            values=[
                "<b>Mixed+Long</b>",
                "<b>Base $/1M</b>",
                "<b>Cached $/1M</b>",
                "<b>Δ%</b>",
                "<b>Base ms</b>",
                "<b>Cached ms</b>",
            ],
            fill_color=HEADER_BG,
            font=dict(color="white", size=10),
            align=["left", "center", "center", "center", "center", "center"],
            height=24,
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=CARD,
            align=["left", "center", "center", "center", "center", "center"],
            height=20,
            font=dict(color=TEXT, size=10),
            line_color=GRID,
        ),
    )


def build_dashboard(baseline: Report, minimal: Report) -> go.Figure:
    # Get data for charts
    base_ml = get_mixed_long_by_concurrency(baseline)
    mini_ml = get_mixed_long_by_concurrency(minimal)
    concs = [str(c) for c in CONCURRENCY_ORDER]

    fig = make_subplots(
        rows=4,
        cols=3,
        specs=[
            [{"type": "table"}, {"type": "table"}, {"type": "table"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "table"}, {"type": "table"}, {"type": "table"}],
            [{"type": "table", "colspan": 3}, None, None],
        ],
        subplot_titles=[
            "",
            "",
            "",
            "Mean Latency by Concurrency (ms)",
            "Throughput by Concurrency (Req Tok/s)",
            "Cost by Concurrency ($/1M Tok)",
            "Batch 64 — Schema Modes",
            "Batch 64 — Text Modes",
            "Mixed+Long — All Concurrency",
            "",
        ],
        vertical_spacing=0.03,
        horizontal_spacing=0.04,
        row_heights=[0.25, 0.35, 0.25, 0.15],
    )

    # Row 1: Three compact tables
    fig.add_trace(build_latency_table(baseline, minimal), row=1, col=1)
    fig.add_trace(build_throughput_table(baseline, minimal), row=1, col=2)
    fig.add_trace(build_cost_table(baseline, minimal), row=1, col=3)

    # Row 2: Charts for mixed+long across concurrency with text annotations
    bar_cfg = dict(marker_line_width=0, width=0.32, opacity=0.9)

    # Latency chart
    base_lat = [base_ml.get(int(c), {}).get("mean_ms", 0) for c in concs]
    mini_lat = [mini_ml.get(int(c), {}).get("mean_ms", 0) for c in concs]
    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=concs,
            y=base_lat,
            marker_color=BASELINE,
            text=[f"{v:.0f}" for v in base_lat],
            textposition="outside",
            textfont=dict(size=9),
            **bar_cfg,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Cached",
            x=concs,
            y=mini_lat,
            marker_color=CURRENT,
            text=[f"{v:.0f}" for v in mini_lat],
            textposition="outside",
            textfont=dict(size=9),
            **bar_cfg,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Throughput chart
    base_tp = [base_ml.get(int(c), {}).get("req_tok_s", 0) for c in concs]
    mini_tp = [mini_ml.get(int(c), {}).get("req_tok_s", 0) for c in concs]
    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=concs,
            y=base_tp,
            marker_color=BASELINE,
            text=[f"{v:.0f}" for v in base_tp],
            textposition="outside",
            textfont=dict(size=9),
            **bar_cfg,
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Cached",
            x=concs,
            y=mini_tp,
            marker_color=CURRENT,
            text=[f"{v:.0f}" for v in mini_tp],
            textposition="outside",
            textfont=dict(size=9),
            **bar_cfg,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Cost chart
    base_cost = [base_ml.get(int(c), {}).get("cost_per_mtok", 0) for c in concs]
    mini_cost = [mini_ml.get(int(c), {}).get("cost_per_mtok", 0) for c in concs]
    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=concs,
            y=base_cost,
            marker_color=BASELINE,
            text=[f"${v:.4f}" for v in base_cost],
            textposition="outside",
            textfont=dict(size=9),
            **bar_cfg,
            showlegend=False,
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            name="Cached",
            x=concs,
            y=mini_cost,
            marker_color=CURRENT,
            text=[f"${v:.4f}" for v in mini_cost],
            textposition="outside",
            textfont=dict(size=9),
            **bar_cfg,
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # Row 3: Batch 64 breakdown tables
    fig.add_trace(build_batch64_schema_table(baseline, minimal), row=3, col=1)
    fig.add_trace(build_batch64_text_table(baseline, minimal), row=3, col=2)
    fig.add_trace(build_mixed_long_cost_table(baseline, minimal), row=3, col=3)

    # Row 4: Configuration info (compact)
    base = baseline.data
    config_rows = [
        [
            "Model: "
            + base.get("model", "N/A")
            + "  |  GPU: Modal L4 $"
            + f"{base.get('gpu_hourly_price', 0):.2f}/hr"
            + "  |  Tokenizer: "
            + base.get("token_encoding", "N/A")
        ],
        [
            "Req/Scen: "
            + str(base.get("requests", "N/A"))
            + "  |  Concurrency: "
            + ", ".join(str(c) for c in base.get("concurrency", []))
            + "  |  Schema: "
            + ", ".join(base.get("schema_modes", []))
            + "  |  Text: "
            + ", ".join(base.get("text_modes", []))
        ],
    ]
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Configuration</b>"],
                fill_color=HEADER_BG,
                font=dict(color="white", size=9),
                height=20,
            ),
            cells=dict(
                values=list(zip(*config_rows)),
                fill_color=CARD,
                align="left",
                height=18,
                font=dict(color=TEXT, size=9),
                line_color=GRID,
            ),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text="<b>GLiNER2 Cache Benchmarking</b><br><sup>feat: enhance GLiNER2 processing with tokenization caching and schema preprocessing</sup>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, color=TEXT),
        ),
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        height=1000,
        width=1600,
        margin=dict(t=80, r=15, b=10, l=15),
        barmode="group",
        bargap=0.2,
        showlegend=False,
        font=dict(color=TEXT, size=10),
    )

    # Style axes
    for col in [1, 2, 3]:
        fig.update_xaxes(
            showgrid=False, title_text="Concurrency", row=2, col=col, title_font=dict(size=9)
        )
        fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, row=2, col=col, tickfont=dict(size=9))

    return fig


def print_chat_tables(baseline: Report, minimal: Report) -> None:
    print("\nOverall")
    print(
        "| Report | Success | Fail | Status Counts | Mean ms | Median ms | P95 ms | P99 ms | Req/s | Text tok/s | Request tok/s | Avg Resp Bytes | Cost / 1M Request Tok |"
    )
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for report in [baseline, minimal]:
        overall = report.overall
        status_counts = ", ".join(f"{k}:{v}" for k, v in sorted(overall["status_counts"].items()))
        print(
            f"| {report.label} | {overall['ok']} | {overall['errors']} | {status_counts} | "
            f"{overall['mean_ms']:.2f} | {overall['median_ms']:.2f} | {overall['p95_ms']:.2f} | {overall['p99_ms']:.2f} | "
            f"{overall['requests_per_sec']:.2f} | {overall['text_tokens_per_sec_exact']:.2f} | "
            f"{overall['request_tokens_per_sec_exact']:.2f} | {overall['avg_response_bytes']:.2f} | "
            f"{overall['cost_per_million_request_tokens_exact']:.5f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare minimal vs baseline benchmark JSON reports and render Plotly HTML."
    )
    parser.add_argument(
        "--baseline",
        default="benchmark_modal_vllm_l4_smoke_patched_original_final_2026-04-12.json",
        help="Baseline/original JSON report path.",
    )
    parser.add_argument(
        "--minimal",
        default="benchmark_modal_vllm_l4_smoke_patched_minimal_final_2026-04-12.json",
        help="Minimal patch JSON report path.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_l4_comparison_dashboard.html",
        help="Output HTML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_report(Path(args.baseline))
    minimal = load_report(Path(args.minimal))
    print_chat_tables(baseline, minimal)
    fig = build_dashboard(baseline, minimal)
    output = Path(args.output)
    fig.write_html(output, include_plotlyjs=True, config={"displayModeBar": False})
    print(f"\nWrote dashboard to {output.resolve()}")


if __name__ == "__main__":
    main()
