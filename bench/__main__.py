"""CLI entry point: python -m bench run|chart|report"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_CHARTS_DIR = Path(__file__).parent / "charts"
DEFAULT_AUDIT_RESULTS_DIR = Path(__file__).parent / "results_v019_audit"


def _parse_csv_ints(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def _parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_results_path(path: Path):
    from .results import BenchResult

    if path.is_dir():
        return BenchResult.load_dir(path)
    return [BenchResult.from_json(path)]


def _latest_by_plugin(results):
    latest = {}
    for result in results:
        current = latest.get(result.plugin)
        if current is None or result.timestamp > current.timestamp:
            latest[result.plugin] = result
    return latest


def _sweep_index(result):
    return {(s.mode, s.concurrency): s for s in result.sweeps}


def cmd_compare(args):
    target_results = _latest_by_plugin(_parse_results_path(Path(args.targets)))
    current_results = _latest_by_plugin(_parse_results_path(Path(args.current)))

    if args.plugins:
        plugins = [p for p in args.plugins if p in target_results]
    else:
        plugins = sorted(set(target_results) & set(current_results))

    focus_levels = args.focus_levels
    report = []
    for plugin in plugins:
        target = target_results.get(plugin)
        current = current_results.get(plugin)
        if target is None or current is None:
            continue

        target_sweeps = _sweep_index(target)
        current_sweeps = _sweep_index(current)
        comparisons = []
        for mode in args.modes:
            for concurrency in focus_levels:
                target_point = target_sweeps.get((mode, concurrency))
                current_point = current_sweeps.get((mode, concurrency))
                if target_point is None or current_point is None:
                    continue
                comparisons.append(
                    {
                        "mode": mode,
                        "concurrency": concurrency,
                        "target_throughput_factor": round(target_point.throughput_factor, 2),
                        "current_throughput_factor": round(current_point.throughput_factor, 2),
                        "target_vllm_req_per_s": round(target_point.vllm_req_per_s, 1),
                        "current_vllm_req_per_s": round(current_point.vllm_req_per_s, 1),
                        "target_vanilla_req_per_s": round(target_point.vanilla_req_per_s, 1),
                        "current_vanilla_req_per_s": round(current_point.vanilla_req_per_s, 1),
                        "throughput_factor_delta": round(
                            current_point.throughput_factor - target_point.throughput_factor,
                            2,
                        ),
                        "vllm_req_per_s_delta": round(
                            current_point.vllm_req_per_s - target_point.vllm_req_per_s,
                            1,
                        ),
                        "vanilla_req_per_s_delta": round(
                            current_point.vanilla_req_per_s - target_point.vanilla_req_per_s,
                            1,
                        ),
                    }
                )

        report.append(
            {
                "plugin": plugin,
                "target_timestamp": target.timestamp,
                "current_timestamp": current.timestamp,
                "target_parity_score": target.parity_score,
                "current_parity_score": current.parity_score,
                "comparisons": comparisons,
            }
        )

    print(json.dumps({"plugins": report}, indent=2))


def cmd_run(args):
    from .registry import list_plugins
    from .runner import run_benchmark

    plugins = [args.plugin] if args.plugin else list_plugins()
    if args.all:
        plugins = list_plugins()

    results = []
    for plugin in plugins:
        try:
            result = run_benchmark(
                plugin_name=plugin,
                num_requests=args.num_requests,
                concurrency_levels=args.concurrency_levels,
                modes=args.modes,
                warmup=args.warmup,
                seq_len=args.seq_len,
                output_dir=args.output,
                staggered_load_fraction=args.staggered_load_fraction,
            )
            results.append(result)
        except Exception as e:
            print(f"\n  FAILED: {plugin} — {type(e).__name__}: {e!r}")
            if not args.all and not args.plugin:
                raise

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {len(results)}/{len(plugins)} benchmarks completed")
    print(f"{'='*70}")
    for r in results:
        parts = [f"  {r.plugin:25s}  parity={r.parity_score:.4f}"]
        for mode in r.modes:
            best = r.best_sweep(mode)
            if best is None:
                continue
            parts.append(
                f"{mode}=throughput {best.throughput_factor:.1f}x "
                f"latency {best.latency_factor:.1f}x @ {best.concurrency}"
            )
        print("  |  ".join(parts))


def cmd_chart(args):
    from .charts import generate_charts
    from .results import BenchResult

    results_path = Path(args.results)
    if results_path.is_dir():
        results = BenchResult.load_dir(results_path)
    else:
        results = [BenchResult.from_json(results_path)]

    if not results:
        print(f"No results found in {results_path}")
        sys.exit(1)

    generate_charts(results, args.output)


def cmd_report(args):
    from .results import BenchResult

    results_path = Path(args.results)
    if results_path.is_dir():
        results = BenchResult.load_dir(results_path)
    else:
        results = [BenchResult.from_json(results_path)]

    if not results:
        print(f"No results found in {results_path}")
        sys.exit(1)

    report = {
        "total": len(results),
        "results": [],
    }

    def _sort_key(result):
        return max((s.throughput_factor for s in result.sweeps), default=0.0)

    for r in sorted(results, key=_sort_key, reverse=True):
        mode_summary = {}
        for mode in r.modes:
            best = r.best_sweep(mode)
            if best is None:
                continue
            mode_summary[mode] = {
                "best_concurrency": best.concurrency,
                "throughput_factor": best.throughput_factor,
                "latency_factor": best.latency_factor,
                "vllm_req_per_s": best.vllm_req_per_s,
                "vanilla_req_per_s": best.vanilla_req_per_s,
            }
        report["results"].append({
            "plugin": r.plugin,
            "model_id": r.model_id,
            "served_model_id": r.served_model_id,
            "gpu": r.gpu,
            "parity_metric": r.parity_metric,
            "parity_score": r.parity_score,
            "modes": r.modes,
            "concurrency_levels": r.concurrency_levels,
            "summary": mode_summary,
            "sweeps": [s.__dict__ for s in r.sweeps],
        })

    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="bench",
        description="vLLM Factory Benchmark Suite",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Run benchmarks")
    p_run.add_argument("--plugin", type=str, help="Single plugin to benchmark")
    p_run.add_argument("--all", action="store_true", help="Run all registered plugins")
    p_run.add_argument("--num-requests", type=int, default=500)
    p_run.add_argument(
        "--concurrency-levels",
        type=_parse_csv_ints,
        default=_parse_csv_ints("1,4,8,16,32,64"),
        help="Comma-separated concurrency / batch-size sweep, e.g. 1,4,8,16,32,64",
    )
    p_run.add_argument(
        "--modes",
        type=_parse_csv_strings,
        default=_parse_csv_strings("saturate,staggered"),
        help="Comma-separated benchmark modes: saturate,staggered",
    )
    p_run.add_argument("--warmup", type=int, default=100)
    p_run.add_argument("--seq-len", type=int, default=None,
                       help="Override default sequence length")
    p_run.add_argument(
        "--staggered-load-fraction",
        type=float,
        default=0.85,
        help="Poisson staggered target rate as a fraction of measured saturate req/s",
    )
    p_run.add_argument("--output", type=str, default=str(DEFAULT_RESULTS_DIR),
                       help="Output directory for result JSON files")

    # --- chart ---
    p_chart = sub.add_parser("chart", help="Generate charts from results")
    p_chart.add_argument("--results", type=str, default=str(DEFAULT_RESULTS_DIR),
                         help="Path to results dir or single JSON file")
    p_chart.add_argument("--output", type=str, default=str(DEFAULT_CHARTS_DIR),
                         help="Output directory for chart files")

    # --- report ---
    p_report = sub.add_parser("report", help="Print JSON summary to stdout")
    p_report.add_argument("--results", type=str, default=str(DEFAULT_RESULTS_DIR),
                          help="Path to results dir or single JSON file")

    # --- compare ---
    p_compare = sub.add_parser("compare", help="Compare current results against historical targets")
    p_compare.add_argument("--targets", type=str, default=str(DEFAULT_RESULTS_DIR),
                           help="Target results dir or single JSON file")
    p_compare.add_argument("--current", type=str, default=str(DEFAULT_AUDIT_RESULTS_DIR),
                           help="Current results dir or single JSON file")
    p_compare.add_argument("--focus-levels", type=_parse_csv_ints, default=_parse_csv_ints("16,32,64"),
                           help="Comma-separated concurrency levels to compare")
    p_compare.add_argument("--modes", type=_parse_csv_strings, default=_parse_csv_strings("saturate,staggered"),
                           help="Comma-separated modes to compare")
    p_compare.add_argument("--plugins", type=_parse_csv_strings, default=None,
                           help="Optional comma-separated plugin list")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "chart":
        cmd_chart(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
