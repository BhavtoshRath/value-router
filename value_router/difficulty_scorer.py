"""
Difficulty scorer for the value-weighted routing project.

Estimates how hard an item is to route/decide on using only signals that
would be available before an item is processed (category, price) — not the
item's ground-truth `difficulty` field, which exists purely so the eval
harness can measure how well this scorer's estimate lines up with reality.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from typing import Optional

from value_router.simulator import Item, Simulator
from value_router.stats_utils import correlation as _correlation


@dataclass
class CategoryDifficultyPrior:
    baseline: float  # expected difficulty for this category, 0..1
    price_sensitivity: float  # how much difficulty rises with price within the category


DEFAULT_PRIORS: dict[str, CategoryDifficultyPrior] = {
    "commodity": CategoryDifficultyPrior(baseline=0.15, price_sensitivity=0.05),
    "accessory": CategoryDifficultyPrior(baseline=0.25, price_sensitivity=0.08),
    "mid_tier": CategoryDifficultyPrior(baseline=0.40, price_sensitivity=0.12),
    "premium": CategoryDifficultyPrior(baseline=0.55, price_sensitivity=0.18),
    "luxury": CategoryDifficultyPrior(baseline=0.65, price_sensitivity=0.25),
}

# Price is normalized against this scale so price_sensitivity stays a small,
# comparable coefficient across categories with very different price ranges.
PRICE_NORMALIZER = 500.0


class DifficultyScorer:
    def __init__(
        self,
        priors: Optional[dict[str, CategoryDifficultyPrior]] = None,
        noise_std: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        priors: per-category difficulty baseline + price sensitivity.
        noise_std: stddev of gaussian noise added to simulate imperfect
            estimation (a real scorer wouldn't nail difficulty exactly).
        seed: RNG seed for reproducibility.
        """
        self.priors = priors if priors is not None else DEFAULT_PRIORS
        self.noise_std = noise_std
        self._rng = random.Random(seed)

    def score(self, item: Item) -> float:
        prior = self.priors.get(item.category)
        if prior is None:
            # unknown category: fall back to the average baseline across known categories
            prior = CategoryDifficultyPrior(
                baseline=statistics.mean(p.baseline for p in self.priors.values()),
                price_sensitivity=statistics.mean(p.price_sensitivity for p in self.priors.values()),
            )
        price_signal = (item.price / PRICE_NORMALIZER) * prior.price_sensitivity
        noise = self._rng.gauss(0, self.noise_std)
        estimate = prior.baseline + price_signal + noise
        return round(min(1.0, max(0.0, estimate)), 4)

    def score_batch(self, items: list[Item]) -> list[float]:
        return [self.score(it) for it in items]


def evaluate(items: list[Item], estimates: list[float]) -> dict:
    """Compare scorer estimates against ground-truth difficulty."""
    errors = [est - it.difficulty for it, est in zip(items, estimates)]
    abs_errors = [abs(e) for e in errors]
    correlation = _correlation([it.difficulty for it in items], estimates) if len(items) > 1 else None
    return {
        "n": len(items),
        "mean_abs_error": round(statistics.mean(abs_errors), 4),
        "mean_error": round(statistics.mean(errors), 4),  # signed, shows over/under-estimation bias
        "correlation": round(correlation, 4) if correlation is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Score synthetic items for estimated difficulty.")
    parser.add_argument("-n", type=int, default=2000, help="number of items to generate")
    parser.add_argument("--seed", type=int, default=42, help="simulator seed")
    parser.add_argument("--scorer-seed", type=int, default=7, help="difficulty scorer noise seed")
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--out", type=str, default=None, help="optional path to write scored items as JSONL")
    args = parser.parse_args()

    sim = Simulator(seed=args.seed)
    items = sim.generate_batch(args.n)

    scorer = DifficultyScorer(noise_std=args.noise_std, seed=args.scorer_seed)
    estimates = scorer.score_batch(items)

    metrics = evaluate(items, estimates)
    print(f"Scored {metrics['n']} items")
    print(f"mean absolute error: {metrics['mean_abs_error']}")
    print(f"mean signed error:   {metrics['mean_error']}")
    print(f"correlation w/ true difficulty: {metrics['correlation']}")

    if args.out:
        with open(args.out, "w") as f:
            for it, est in zip(items, estimates):
                row = it.as_dict()
                row["estimated_difficulty"] = est
                f.write(json.dumps(row) + "\n")
        print(f"\nWrote {len(items)} scored items to {args.out}")


if __name__ == "__main__":
    main()
