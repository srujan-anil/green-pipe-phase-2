import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import config

METRICS = ["co2_kg", "energy_kwh", "runtime_seconds", "accuracy"]


def load_raw_results(raw_dir):
    results = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, fname), "r") as f:
            data = json.load(f)
            data["_source_file"] = fname
            results.append(data)
    return results


def group_by_strategy(results):
    grouped = {}
    for r in results:
        grouped.setdefault(r["strategy"], []).append(r)
    return grouped


def compute_stats(values):
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    mean = statistics.mean(clean)
    std = statistics.stdev(clean) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n}


def pct_change(value, baseline_value):
    if value is None or baseline_value is None or baseline_value == 0:
        return None
    return ((value - baseline_value) / baseline_value) * 100.0


def build_summary(results, baseline_strategy="baseline"):
    grouped = group_by_strategy(results)
    summary = {}

    baseline_runs = grouped.get(baseline_strategy, [])
    baseline_complete = [r for r in baseline_runs if r.get("completion_status") == "complete"]
    baseline_stats = {}
    for metric in METRICS:
        baseline_stats[metric] = compute_stats([r.get(metric) for r in baseline_complete])

    for strategy, runs in grouped.items():
        complete = [r for r in runs if r.get("completion_status") == "complete"]
        incomplete = [r for r in runs if r.get("completion_status") != "complete"]

        strategy_summary = {
            "strategy": strategy,
            "total_runs": len(runs),
            "complete_runs": len(complete),
            "incomplete_runs": len(incomplete),
            "incomplete_run_files": [r["_source_file"] for r in incomplete],
            "incomplete_run_statuses": [r.get("completion_status") for r in incomplete],
            "seeds_run": sorted(set(r.get("seed") for r in runs)),
            "metrics": {},
        }

        for metric in METRICS:
            stats = compute_stats([r.get(metric) for r in complete])
            baseline_mean = baseline_stats.get(metric, {}).get("mean")
            stats["pct_change_vs_baseline"] = (
                pct_change(stats["mean"], baseline_mean) if strategy != baseline_strategy else 0.0
                if stats["mean"] is not None
                else None
            )
            strategy_summary["metrics"][metric] = stats

        summary[strategy] = strategy_summary

    return summary


def write_summary(summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "strategy_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(out_dir, "strategy_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["strategy", "total_runs", "complete_runs", "incomplete_runs"]
        for metric in METRICS:
            header += [f"{metric}_mean", f"{metric}_std", f"{metric}_n", f"{metric}_pct_change_vs_baseline"]
        writer.writerow(header)

        for strategy, s in summary.items():
            row = [strategy, s["total_runs"], s["complete_runs"], s["incomplete_runs"]]
            for metric in METRICS:
                m = s["metrics"][metric]
                row += [m["mean"], m["std"], m["n"], m["pct_change_vs_baseline"]]
            writer.writerow(row)

    return json_path, csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=os.path.join(config.RESULTS_DIR, "experiments", "raw"))
    parser.add_argument("--out-dir", default=os.path.join(config.RESULTS_DIR, "experiments", "summaries"))
    parser.add_argument("--baseline-strategy", default="baseline")
    args = parser.parse_args()

    results = load_raw_results(args.raw_dir)
    if not results:
        print(f"No raw result files found in {args.raw_dir}")
        sys.exit(1)

    summary = build_summary(results, baseline_strategy=args.baseline_strategy)
    json_path, csv_path = write_summary(summary, args.out_dir)

    for strategy, s in summary.items():
        print(f"\n{strategy}: {s['complete_runs']}/{s['total_runs']} complete, seeds={s['seeds_run']}")
        if s["incomplete_runs"] > 0:
            print(f"  INCOMPLETE: {list(zip(s['incomplete_run_files'], s['incomplete_run_statuses']))}")
        for metric in METRICS:
            m = s["metrics"][metric]
            if m["mean"] is not None:
                pct = f"{m['pct_change_vs_baseline']:+.1f}%" if m["pct_change_vs_baseline"] is not None else "n/a"
                print(f"  {metric}: mean={m['mean']:.6g} std={m['std']:.6g} n={m['n']} vs_baseline={pct}")
            else:
                print(f"  {metric}: no complete data")

    print(f"\nWritten to: {json_path}")
    print(f"Written to: {csv_path}")