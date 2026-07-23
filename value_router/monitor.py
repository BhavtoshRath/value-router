"""
Monitor for the value-weighted routing project.

Reads a decision log written by decision_logger.py and produces a lightweight
dashboard: spend by category (vs. volume share, to surface starvation of the
low-volume/high-value segment) and a calibration check (estimate vs. the
simulator's hidden ground truth, for both value and difficulty).

This is a dashboard sketch for demonstrating the monitoring concern, not a
production monitoring system.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Optional

from value_router.decision_logger import DecisionLogEntry, load_log
from value_router.router import SLOW
from value_router.stats_utils import correlation


@dataclass
class CategorySpend:
    category: str
    n: int
    volume_share: float
    slow_share: float
    cost: float
    spend_share: float


@dataclass
class CalibrationReport:
    n: int
    mean_abs_error: float
    mean_error: float
    mean_abs_pct_error: Optional[float]
    correlation: Optional[float]


def spend_by_category(entries: list[DecisionLogEntry]) -> list[CategorySpend]:
    total_cost = sum(e.cost for e in entries)
    total_n = len(entries)
    by_cat: dict[str, list[DecisionLogEntry]] = {}
    for e in entries:
        by_cat.setdefault(e.category, []).append(e)

    rows = []
    for cat, es in by_cat.items():
        n = len(es)
        slow_count = sum(1 for e in es if e.path == SLOW)
        cost = sum(e.cost for e in es)
        rows.append(
            CategorySpend(
                category=cat,
                n=n,
                volume_share=round(n / total_n, 4) if total_n else 0.0,
                slow_share=round(slow_count / n, 4) if n else 0.0,
                cost=cost,
                spend_share=round(cost / total_cost, 4) if total_cost else 0.0,
            )
        )
    return sorted(rows, key=lambda r: -r.volume_share)


def _calibration(true_values: list[float], estimates: list[float]) -> CalibrationReport:
    errors = [est - true for true, est in zip(true_values, estimates)]
    abs_errors = [abs(e) for e in errors]
    rel_errors = [abs(e) / t for e, t in zip(errors, true_values) if t > 0]
    corr = correlation(true_values, estimates) if len(true_values) > 1 else None
    return CalibrationReport(
        n=len(true_values),
        mean_abs_error=round(statistics.mean(abs_errors), 4) if abs_errors else 0.0,
        mean_error=round(statistics.mean(errors), 4) if errors else 0.0,
        mean_abs_pct_error=round(statistics.mean(rel_errors) * 100, 2) if rel_errors else None,
        correlation=round(corr, 4) if corr is not None else None,
    )


def calibration_check(entries: list[DecisionLogEntry]) -> dict[str, CalibrationReport]:
    return {
        "value": _calibration([e.value_true for e in entries], [e.value_estimate for e in entries]),
        "difficulty": _calibration([e.difficulty_true for e in entries], [e.difficulty_estimate for e in entries]),
    }


def calibration_by_category(entries: list[DecisionLogEntry]) -> dict[str, dict[str, CalibrationReport]]:
    by_cat: dict[str, list[DecisionLogEntry]] = {}
    for e in entries:
        by_cat.setdefault(e.category, []).append(e)
    return {cat: calibration_check(es) for cat, es in by_cat.items()}


def print_dashboard(entries: list[DecisionLogEntry]) -> None:
    strategy = entries[0].strategy if entries else "?"
    total_cost = sum(e.cost for e in entries)
    print(f"Decision log: {len(entries)} entries, strategy='{strategy}', total budget={total_cost:.0f}\n")

    print("spend by category (volume_share vs spend_share -- gap flags starvation/over-spend):")
    header = f"{'category':<12}{'n':>7}{'volume%':>10}{'slow%':>9}{'spend%':>9}"
    print(header)
    print("-" * len(header))
    for row in spend_by_category(entries):
        flag = ""
        if row.spend_share < row.volume_share - 0.05:
            flag = "  <- under-spending relative to volume"
        elif row.spend_share > row.volume_share + 0.05:
            flag = "  <- over-spending relative to volume"
        print(
            f"{row.category:<12}{row.n:>7}{row.volume_share * 100:>9.1f}%"
            f"{row.slow_share * 100:>8.1f}%{row.spend_share * 100:>8.1f}%{flag}"
        )

    print("\ncalibration check (estimate vs. ground truth, overall):")
    overall = calibration_check(entries)
    for signal, rep in overall.items():
        pct = "" if rep.mean_abs_pct_error is None else f"{rep.mean_abs_pct_error:.1f}%"
        corr = "" if rep.correlation is None else f"{rep.correlation:.3f}"
        print(
            f"  {signal:<12}mean_abs_error={rep.mean_abs_error:<10}"
            f"mean_abs_pct_error={pct:<8}correlation={corr}"
        )

    print("\ncalibration by category (correlation with ground truth):")
    by_cat = calibration_by_category(entries)
    for cat, reps in by_cat.items():
        v, d = reps["value"], reps["difficulty"]
        v_corr = "" if v.correlation is None else f"{v.correlation:.3f}"
        d_corr = "" if d.correlation is None else f"{d.correlation:.3f}"
        print(f"  {cat:<12}value_corr={v_corr:<8}difficulty_corr={d_corr}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate a decision log into a spend/calibration dashboard.")
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="path to a decision log JSONL written by decision_logger.py",
    )
    args = parser.parse_args()

    entries = load_log(args.log)
    if not entries:
        print(f"No entries found in {args.log}")
        return
    print_dashboard(entries)


if __name__ == "__main__":
    main()