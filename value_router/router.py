"""
Router for the value-weighted routing project.

Decides whether an item goes down the cheap "fast path" or the expensive
"slow path", using estimated difficulty and estimated value (never the
simulator's hidden ground truth) as a simple 2D threshold matrix rather than
a learned model.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Optional

from value_router.simulator import Item, Simulator
from value_router.difficulty_scorer import DifficultyScorer
from value_router.value_estimator import ValueEstimator

FAST = "fast"
SLOW = "slow"


@dataclass
class RoutingDecision:
    item_id: int
    difficulty_estimate: float
    value_estimate: float
    path: str


class ValueWeightedRouter:
    """Routes on a 2x2 threshold matrix over (difficulty, value).

    Only items that are both hard *and* valuable go to the slow path;
    hard-but-low-value items stay on the fast path so slow-path budget isn't
    spent on items that wouldn't move the needle even if handled carefully.
    """

    def __init__(self, difficulty_threshold: float = 0.5, value_threshold: float = 20.0):
        self.difficulty_threshold = difficulty_threshold
        self.value_threshold = value_threshold

    def route(self, difficulty_estimate: float, value_estimate: float) -> str:
        is_hard = difficulty_estimate >= self.difficulty_threshold
        is_valuable = value_estimate >= self.value_threshold
        return SLOW if (is_hard and is_valuable) else FAST


class DifficultyOnlyRouter:
    """Baseline: routes purely on difficulty, ignoring value entirely."""

    def __init__(self, difficulty_threshold: float = 0.5):
        self.difficulty_threshold = difficulty_threshold

    def route(self, difficulty_estimate: float, value_estimate: float) -> str:
        return SLOW if difficulty_estimate >= self.difficulty_threshold else FAST


class RandomRouter:
    """Baseline: routes to the slow path with a fixed probability, ignoring both signals."""

    def __init__(self, slow_path_rate: float = 0.2, seed: Optional[int] = None):
        self.slow_path_rate = slow_path_rate
        self._rng = random.Random(seed)

    def route(self, difficulty_estimate: float, value_estimate: float) -> str:
        return SLOW if self._rng.random() < self.slow_path_rate else FAST


def route_items(
    items: list[Item],
    difficulty_estimates: list[float],
    value_estimates: list[float],
    router,
) -> list[RoutingDecision]:
    decisions = []
    for item, d_est, v_est in zip(items, difficulty_estimates, value_estimates):
        path = router.route(d_est, v_est)
        decisions.append(RoutingDecision(item.id, d_est, v_est, path))
    return decisions


def summarize(decisions: list[RoutingDecision]) -> dict:
    slow = [d for d in decisions if d.path == SLOW]
    fast = [d for d in decisions if d.path == FAST]
    return {
        "n": len(decisions),
        "slow_count": len(slow),
        "slow_share": round(len(slow) / len(decisions), 4) if decisions else 0.0,
        "fast_count": len(fast),
    }


def main():
    parser = argparse.ArgumentParser(description="Route synthetic items to a fast or slow path.")
    parser.add_argument("-n", type=int, default=2000, help="number of items to generate")
    parser.add_argument("--seed", type=int, default=42, help="simulator seed")
    parser.add_argument("--scorer-seed", type=int, default=7, help="difficulty scorer noise seed")
    parser.add_argument("--estimator-seed", type=int, default=11, help="value estimator noise seed")
    parser.add_argument("--difficulty-threshold", type=float, default=0.5)
    parser.add_argument("--value-threshold", type=float, default=20.0)
    parser.add_argument(
        "--strategy",
        choices=["value_weighted", "difficulty_only", "random"],
        default="value_weighted",
    )
    args = parser.parse_args()

    sim = Simulator(seed=args.seed)
    items = sim.generate_batch(args.n)

    difficulty_estimates = DifficultyScorer(seed=args.scorer_seed).score_batch(items)
    value_estimates = ValueEstimator(seed=args.estimator_seed).estimate_batch(items)

    if args.strategy == "value_weighted":
        router = ValueWeightedRouter(args.difficulty_threshold, args.value_threshold)
    elif args.strategy == "difficulty_only":
        router = DifficultyOnlyRouter(args.difficulty_threshold)
    else:
        router = RandomRouter(seed=args.scorer_seed)

    decisions = route_items(items, difficulty_estimates, value_estimates, router)
    summary = summarize(decisions)

    print(f"Routed {summary['n']} items using '{args.strategy}' strategy")
    print(f"slow path: {summary['slow_count']} ({summary['slow_share'] * 100:.1f}%)")
    print(f"fast path: {summary['fast_count']}")

    # Sanity check: what fraction of *truly* high-value items (ground truth) landed
    # on the slow path? A value-blind router should do noticeably worse here.
    true_high_value = [it for it in items if it.value >= args.value_threshold]
    if true_high_value:
        slow_ids = {d.item_id for d in decisions if d.path == SLOW}
        covered = sum(1 for it in true_high_value if it.id in slow_ids)
        print(
            f"\ntrue high-value items ({len(true_high_value)}) routed to slow path: "
            f"{covered} ({covered / len(true_high_value) * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
