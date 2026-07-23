"""
Decision logger for the value-weighted routing project.

Runs the simulate -> estimate -> route pipeline for a single strategy and
writes one structured log line per item: the estimates the router acted on,
the routing decision itself, its cost, and the simulator's hidden ground
truth (value/difficulty). A real deployment would call this after every live
routing decision; here it's a batch CLI since there's no live traffic to log,
so a downstream monitor can check spend and calibration without re-running
the pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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

# Same unit-cost model as the eval harness -- only their ratio matters.
FAST_COST = 1.0
SLOW_COST = 20.0


@dataclass
class DecisionLogEntry:
    item_id: int
    category: str
    price: float
    strategy: str
    path: str
    cost: float
    difficulty_estimate: float
    value_estimate: float
    difficulty_true: float
    value_true: float

    def as_dict(self) -> dict:
        return asdict(self)


def build_log(
    items: list[Item],
    decisions: list[RoutingDecision],
    strategy: str,
    fast_cost: float = FAST_COST,
    slow_cost: float = SLOW_COST,
) -> list[DecisionLogEntry]:
    entries = []
    for item, decision in zip(items, decisions):
        cost = slow_cost if decision.path == SLOW else fast_cost
        entries.append(
            DecisionLogEntry(
                item_id=item.id,
                category=item.category,
                price=item.price,
                strategy=strategy,
                path=decision.path,
                cost=cost,
                difficulty_estimate=decision.difficulty_estimate,
                value_estimate=decision.value_estimate,
                difficulty_true=item.difficulty,
                value_true=item.value,
            )
        )
    return entries


def write_log(entries: list[DecisionLogEntry], path: str) -> None:
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e.as_dict()) + "\n")


def load_log(path: str) -> list[DecisionLogEntry]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(DecisionLogEntry(**json.loads(line)))
    return entries


def _build_router(
    strategy: str,
    difficulty_threshold: float,
    value_threshold: float,
    random_seed: Optional[int],
):
    if strategy == "value_weighted":
        return ValueWeightedRouter(difficulty_threshold, value_threshold)
    if strategy == "difficulty_only":
        return DifficultyOnlyRouter(difficulty_threshold)
    return RandomRouter(seed=random_seed)


def main():
    parser = argparse.ArgumentParser(description="Log routing decisions for the value-weighted routing project.")
    parser.add_argument("-n", type=int, default=2000, help="number of items to generate")
    parser.add_argument("--seed", type=int, default=42, help="simulator seed")
    parser.add_argument("--scorer-seed", type=int, default=7)
    parser.add_argument("--estimator-seed", type=int, default=11)
    parser.add_argument("--random-router-seed", type=int, default=99)
    parser.add_argument("--difficulty-threshold", type=float, default=0.5)
    parser.add_argument("--value-threshold", type=float, default=20.0)
    parser.add_argument(
        "--strategy",
        choices=["value_weighted", "difficulty_only", "random"],
        default="value_weighted",
    )
    parser.add_argument("--out", type=str, default="decision_log.jsonl", help="path to write the decision log JSONL")
    args = parser.parse_args()

    sim = Simulator(seed=args.seed)
    items = sim.generate_batch(args.n)

    difficulty_estimates = DifficultyScorer(seed=args.scorer_seed).score_batch(items)
    value_estimates = ValueEstimator(seed=args.estimator_seed).estimate_batch(items)

    router = _build_router(args.strategy, args.difficulty_threshold, args.value_threshold, args.random_router_seed)
    decisions = route_items(items, difficulty_estimates, value_estimates, router)

    entries = build_log(items, decisions, args.strategy)
    write_log(entries, args.out)

    total_cost = sum(e.cost for e in entries)
    print(f"Logged {len(entries)} decisions for strategy '{args.strategy}' to {args.out}")
    print(f"total budget spent: {total_cost:.0f}")


if __name__ == "__main__":
    main()