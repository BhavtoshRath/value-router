"""
Value estimator for the value-weighted routing project.

Estimates the expected profit ("value") of an item using only signals that
would be available before an item is processed (category, price) — not the
item's true per-item margin, which drives the ground-truth `value` field
the eval harness uses to grade this estimator.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from typing import Optional

from value_router.simulator import Item, Simulator
from value_router.stats_utils import correlation


@dataclass
class CategoryMarginPrior:
    expected_margin: float  # typical margin for this category, as a fraction (e.g. 0.10 = 10%)


# Reflects the kind of category-level margin knowledge a business would
# actually have on hand (historical averages), not the exact per-item
# margin, which fluctuates with discounts/costs and isn't known upfront.
DEFAULT_MARGIN_PRIORS: dict[str, CategoryMarginPrior] = {
    "commodity": CategoryMarginPrior(expected_margin=0.09),
    "accessory": CategoryMarginPrior(expected_margin=0.15),
    "mid_tier": CategoryMarginPrior(expected_margin=0.22),
    "premium": CategoryMarginPrior(expected_margin=0.29),
    "luxury": CategoryMarginPrior(expected_margin=0.40),
}


class ValueEstimator:
    def __init__(
        self,
        margin_priors: Optional[dict[str, CategoryMarginPrior]] = None,
        noise_std: float = 0.02,
        seed: Optional[int] = None,
    ):
        """
        margin_priors: per-category expected margin (fraction).
        noise_std: stddev of gaussian noise added to the estimated margin,
            simulating imprecision in the category-level margin estimate.
        seed: RNG seed for reproducibility.
        """
        self.margin_priors = margin_priors if margin_priors is not None else DEFAULT_MARGIN_PRIORS
        self.noise_std = noise_std
        self._rng = random.Random(seed)

    def estimate(self, item: Item) -> float:
        prior = self.margin_priors.get(item.category)
        if prior is None:
            # unknown category: fall back to the average prior across known categories
            prior = CategoryMarginPrior(
                expected_margin=statistics.mean(p.expected_margin for p in self.margin_priors.values())
            )
        noise = self._rng.gauss(0, self.noise_std)
        estimated_margin = max(0.0, prior.expected_margin + noise)
        return round(item.price * estimated_margin, 4)

    def estimate_batch(self, items: list[Item]) -> list[float]:
        return [self.estimate(it) for it in items]


def evaluate(items: list[Item], estimates: list[float]) -> dict:
    """Compare estimator output against ground-truth value (price * true margin)."""
    errors = [est - it.value for it, est in zip(items, estimates)]
    abs_errors = [abs(e) for e in errors]
    rel_errors = [abs(e) / it.value for e, it in zip(errors, items) if it.value > 0]
    corr = correlation([it.value for it in items], estimates) if len(items) > 1 else None
    return {
        "n": len(items),
        "mean_abs_error": round(statistics.mean(abs_errors), 4),
        "mean_error": round(statistics.mean(errors), 4),  # signed, shows over/under-estimation bias
        "mean_abs_pct_error": round(statistics.mean(rel_errors) * 100, 2) if rel_errors else None,
        "correlation": round(corr, 4) if corr is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate value for synthetic items.")
    parser.add_argument("-n", type=int, default=2000, help="number of items to generate")
    parser.add_argument("--seed", type=int, default=42, help="simulator seed")
    parser.add_argument("--estimator-seed", type=int, default=11, help="value estimator noise seed")
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--out", type=str, default=None, help="optional path to write estimated items as JSONL")
    args = parser.parse_args()

    sim = Simulator(seed=args.seed)
    items = sim.generate_batch(args.n)

    estimator = ValueEstimator(noise_std=args.noise_std, seed=args.estimator_seed)
    estimates = estimator.estimate_batch(items)

    metrics = evaluate(items, estimates)
    print(f"Estimated value for {metrics['n']} items")
    print(f"mean absolute error:     {metrics['mean_abs_error']}")
    print(f"mean signed error:       {metrics['mean_error']}")
    print(f"mean absolute % error:   {metrics['mean_abs_pct_error']}%")
    print(f"correlation w/ true value: {metrics['correlation']}")

    if args.out:
        with open(args.out, "w") as f:
            for it, est in zip(items, estimates):
                row = it.as_dict()
                row["estimated_value"] = est
                f.write(json.dumps(row) + "\n")
        print(f"\nWrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
