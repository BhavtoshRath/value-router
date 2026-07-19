"""
Eval harness for the value-weighted routing project.

Runs the full pipeline (simulate -> estimate -> route) for each router
strategy against the same items, then grades every decision against the
simulator's hidden ground truth: how much true value ended up on the slow
path, how much budget that cost, and whether any category gets starved of
careful treatment relative to its share of item volume.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from value_router.simulator import Item, Simulator
from value_router.difficulty_scorer import DifficultyScorer
from value_router.value_estimator import ValueEstimator
from value_router.router import (
    SLOW,
    DifficultyOnlyRouter,
    RandomRouter,
    RoutingDecision,
    ValueWeightedRouter,
    route_items,
)
from value_router.stats_utils import correlation

# Arbitrary but fixed unit costs standing in for "cheap heuristic" vs
# "expensive LLM call" — only their ratio matters for comparing strategies.
FAST_COST = 1.0
SLOW_COST = 20.0


@dataclass
class StrategyReport:
    name: str
    n: int
    slow_count: int
    slow_share: float
    budget: float
    value_protected: float  # true value of items routed to the slow path
    value_starved: float  # true value of high-value items left on the fast path
    recall_high_value: Optional[float]  # of true high-value items, fraction sent slow
    precision_slow: Optional[float]  # of slow-path items, fraction that were true high-value
    efficiency: Optional[float]  # value_protected per unit of budget spent
    by_category: dict[str, dict]


def _segment_breakdown(items: list[Item], decisions: list[RoutingDecision], value_threshold: float) -> dict[str, dict]:
    slow_ids = {d.item_id for d in decisions if d.path == SLOW}
    by_cat: dict[str, list[Item]] = {}
    for it in items:
        by_cat.setdefault(it.category, []).append(it)

    breakdown = {}
    for cat, cat_items in by_cat.items():
        n = len(cat_items)
        slow_count = sum(1 for it in cat_items if it.id in slow_ids)
        high_value = [it for it in cat_items if it.value >= value_threshold]
        high_value_covered = sum(1 for it in high_value if it.id in slow_ids)
        breakdown[cat] = {
            "n": n,
            "slow_count": slow_count,
            "slow_share": round(slow_count / n, 4) if n else 0.0,
            "high_value_count": len(high_value),
            "high_value_recall": round(high_value_covered / len(high_value), 4) if high_value else None,
        }
    return breakdown


def evaluate_strategy(
    name: str,
    items: list[Item],
    decisions: list[RoutingDecision],
    value_threshold: float,
) -> StrategyReport:
    n = len(items)
    slow_ids = {d.item_id for d in decisions if d.path == SLOW}
    slow_count = len(slow_ids)
    fast_count = n - slow_count

    value_protected = sum(it.value for it in items if it.id in slow_ids)
    high_value_items = [it for it in items if it.value >= value_threshold]
    value_starved = sum(it.value for it in high_value_items if it.id not in slow_ids)

    budget = fast_count * FAST_COST + slow_count * SLOW_COST

    recall = (
        round(sum(1 for it in high_value_items if it.id in slow_ids) / len(high_value_items), 4)
        if high_value_items
        else None
    )
    precision = (
        round(sum(1 for it in items if it.id in slow_ids and it.value >= value_threshold) / slow_count, 4)
        if slow_count
        else None
    )
    efficiency = round(value_protected / budget, 4) if budget else None

    return StrategyReport(
        name=name,
        n=n,
        slow_count=slow_count,
        slow_share=round(slow_count / n, 4) if n else 0.0,
        budget=budget,
        value_protected=round(value_protected, 2),
        value_starved=round(value_starved, 2),
        recall_high_value=recall,
        precision_slow=precision,
        efficiency=efficiency,
        by_category=_segment_breakdown(items, decisions, value_threshold),
    )


def volume_value_starvation_check(report: StrategyReport, items: list[Item]) -> Optional[float]:
    """Correlation between a category's volume share and its slow-path
    coverage of that category's own high-value items. A strongly positive
    correlation means low-volume, high-value categories (e.g. luxury) get
    systematically worse coverage than big, common categories — i.e. the
    router is starving the rare-but-valuable segment."""
    total = len(items)
    by_cat_count: dict[str, int] = {}
    for it in items:
        by_cat_count[it.category] = by_cat_count.get(it.category, 0) + 1

    shares, recalls = [], []
    for cat, stats in report.by_category.items():
        if stats["high_value_recall"] is None:
            continue
        shares.append(by_cat_count[cat] / total)
        recalls.append(stats["high_value_recall"])
    return correlation(shares, recalls) if len(shares) > 1 else None


def compare_strategies(
    items: list[Item],
    difficulty_estimates: list[float],
    value_estimates: list[float],
    value_threshold: float,
    difficulty_threshold: float,
    random_seed: Optional[int],
) -> dict[str, StrategyReport]:
    strategies = {
        "value_weighted": ValueWeightedRouter(difficulty_threshold, value_threshold),
        "difficulty_only": DifficultyOnlyRouter(difficulty_threshold),
        "random": RandomRouter(seed=random_seed),
    }
    reports = {}
    for name, router in strategies.items():
        decisions = route_items(items, difficulty_estimates, value_estimates, router)
        reports[name] = evaluate_strategy(name, items, decisions, value_threshold)
    return reports


def print_report(reports: dict[str, StrategyReport], items: list[Item]) -> None:
    header = (
        f"{'strategy':<16}{'slow%':>8}{'budget':>10}{'value_protected':>17}"
        f"{'value_starved':>15}{'recall':>9}{'precision':>11}{'efficiency':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, r in reports.items():
        recall_str = "" if r.recall_high_value is None else f"{r.recall_high_value * 100:.1f}%"
        precision_str = "" if r.precision_slow is None else f"{r.precision_slow * 100:.1f}%"
        efficiency_str = "" if r.efficiency is None else f"{r.efficiency:.3f}"
        print(
            f"{name:<16}{r.slow_share * 100:>7.1f}%{r.budget:>10.0f}"
            f"{r.value_protected:>17.1f}{r.value_starved:>15.1f}"
            f"{recall_str:>9}{precision_str:>11}{efficiency_str:>12}"
        )

    print("\nvolume/value starvation check (category volume_share vs. that category's high-value recall):")
    for name, r in reports.items():
        corr = volume_value_starvation_check(r, items)
        corr_str = "" if corr is None else f"{corr:.3f}"
        flag = "  <- low-volume/high-value categories under-covered relative to big ones" if (corr or 0) > 0.3 else ""
        print(f"  {name:<16}{corr_str}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate router strategies against ground truth.")
    parser.add_argument("-n", type=int, default=2000, help="number of items to generate")
    parser.add_argument("--seed", type=int, default=42, help="simulator seed")
    parser.add_argument("--scorer-seed", type=int, default=7)
    parser.add_argument("--estimator-seed", type=int, default=11)
    parser.add_argument("--random-router-seed", type=int, default=99)
    parser.add_argument("--difficulty-threshold", type=float, default=0.5)
    parser.add_argument("--value-threshold", type=float, default=20.0)
    args = parser.parse_args()

    sim = Simulator(seed=args.seed)
    items = sim.generate_batch(args.n)

    difficulty_estimates = DifficultyScorer(seed=args.scorer_seed).score_batch(items)
    value_estimates = ValueEstimator(seed=args.estimator_seed).estimate_batch(items)

    reports = compare_strategies(
        items,
        difficulty_estimates,
        value_estimates,
        value_threshold=args.value_threshold,
        difficulty_threshold=args.difficulty_threshold,
        random_seed=args.random_router_seed,
    )

    print(f"Evaluated {args.n} items across {len(reports)} strategies\n")
    print_report(reports, items)


if __name__ == "__main__":
    main()
