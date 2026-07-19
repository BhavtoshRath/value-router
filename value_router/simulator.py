"""
Item simulator for the value-weighted routing project.

Generates synthetic items that carry a "value" dimension (price x margin) on
top of category and difficulty, so downstream components (difficulty scorer,
value estimator, router, eval harness) have something realistic to work with.

Design notes
------------
- Each item belongs to a category. Categories differ in typical price,
  margin, and difficulty, and in how often they occur (volume weight).
- `value = price * margin` is the deterministic ground-truth expected
  profit per item. This is what the value estimator will later try to
  approximate, and what the eval harness will use as the "realized value"
  to check routing decisions against.
- Difficulty is sampled independently of value within a category's range,
  representing how hard it is to make a good routing/content decision for
  that item (not how valuable it is).
- The default category set deliberately bakes in a volume/value inverse
  correlation (cheap, high-volume "commodity" categories vs. rare,
  high-value "premium" categories), matching the concern in the proposal
  that a naive router could silently starve the low-volume/high-value
  segment of budget. This can be turned off via `inverse_correlation=False`
  for a null-hypothesis / control run.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class CategoryConfig:
    name: str
    weight: float  # relative volume share (not required to sum to 1; normalized internally)
    price_range: tuple[float, float]
    margin_range: tuple[float, float]  # fraction, e.g. 0.05 = 5% margin
    difficulty_range: tuple[float, float]  # 0..1, higher = harder to route/decide on


# Deliberately inverse-correlated: high weight (volume) categories have low
# price/margin (low value); low weight categories have high price/margin
# (high value). This mirrors common e-commerce distributions.
DEFAULT_CATEGORIES: list[CategoryConfig] = [
    CategoryConfig("commodity", weight=40.0, price_range=(3, 15), margin_range=(0.05, 0.12), difficulty_range=(0.05, 0.35)),
    CategoryConfig("accessory", weight=30.0, price_range=(10, 40), margin_range=(0.10, 0.20), difficulty_range=(0.15, 0.45)),
    CategoryConfig("mid_tier", weight=18.0, price_range=(40, 150), margin_range=(0.15, 0.28), difficulty_range=(0.25, 0.60)),
    CategoryConfig("premium", weight=9.0, price_range=(150, 500), margin_range=(0.22, 0.35), difficulty_range=(0.40, 0.80)),
    CategoryConfig("luxury", weight=3.0, price_range=(500, 2500), margin_range=(0.30, 0.50), difficulty_range=(0.50, 0.95)),
]


@dataclass
class Item:
    id: int
    category: str
    price: float
    margin: float
    value: float  # ground-truth price * margin
    difficulty: float  # 0..1

    def as_dict(self) -> dict:
        return asdict(self)


class Simulator:
    def __init__(
        self,
        categories: Optional[list[CategoryConfig]] = None,
        seed: Optional[int] = None,
        inverse_correlation: bool = True,
    ):
        """
        categories: override the default category set.
        seed: RNG seed for reproducibility.
        inverse_correlation: if False, all categories are re-weighted equally
            (uniform volume), removing the built-in volume/value inverse
            correlation — useful as a control condition.
        """
        cats = categories if categories is not None else DEFAULT_CATEGORIES
        if not inverse_correlation:
            cats = [
                CategoryConfig(c.name, 1.0, c.price_range, c.margin_range, c.difficulty_range)
                for c in cats
            ]
        self.categories = cats
        self._rng = random.Random(seed)
        self._weights = [c.weight for c in self.categories]
        self._next_id = 0

    def _sample_category(self) -> CategoryConfig:
        return self._rng.choices(self.categories, weights=self._weights, k=1)[0]

    def generate_item(self) -> Item:
        cat = self._sample_category()
        price = self._rng.uniform(*cat.price_range)
        margin = self._rng.uniform(*cat.margin_range)
        difficulty = self._rng.uniform(*cat.difficulty_range)
        value = price * margin
        item = Item(
            id=self._next_id,
            category=cat.name,
            price=round(price, 2),
            margin=round(margin, 4),
            value=round(value, 4),
            difficulty=round(difficulty, 4),
        )
        self._next_id += 1
        return item

    def generate_batch(self, n: int) -> list[Item]:
        return [self.generate_item() for _ in range(n)]


def summarize(items: list[Item]) -> dict:
    """Per-category summary stats, useful for sanity-checking the
    volume/value inverse correlation the eval harness will later check."""
    by_cat: dict[str, list[Item]] = {}
    for it in items:
        by_cat.setdefault(it.category, []).append(it)

    summary = {}
    for cat, its in by_cat.items():
        values = [i.value for i in its]
        summary[cat] = {
            "count": len(its),
            "volume_share": round(len(its) / len(items), 4),
            "mean_value": round(statistics.mean(values), 2),
            "mean_price": round(statistics.mean(i.price for i in its), 2),
            "mean_difficulty": round(statistics.mean(i.difficulty for i in its), 3),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic items for the value router project.")
    parser.add_argument("-n", type=int, default=2000, help="number of items to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-inverse", action="store_true", help="disable built-in volume/value inverse correlation")
    parser.add_argument("--out", type=str, default=None, help="optional path to write items as JSONL")
    args = parser.parse_args()

    sim = Simulator(seed=args.seed, inverse_correlation=not args.no_inverse)
    items = sim.generate_batch(args.n)

    summary = summarize(items)
    print(f"Generated {len(items)} items across {len(summary)} categories\n")
    print(f"{'category':<12}{'volume%':>10}{'mean_value':>13}{'mean_price':>13}{'mean_diff':>11}")
    for cat, s in sorted(summary.items(), key=lambda kv: -kv[1]["volume_share"]):
        print(f"{cat:<12}{s['volume_share']*100:>9.1f}%{s['mean_value']:>13.2f}{s['mean_price']:>13.2f}{s['mean_difficulty']:>11.3f}")

    # quick correlation check: volume_share vs mean_value across categories
    shares = [s["volume_share"] for s in summary.values()]
    values = [s["mean_value"] for s in summary.values()]
    if len(shares) > 1:
        corr = statistics.correlation(shares, values) if hasattr(statistics, "correlation") else None
        if corr is not None:
            direction = "inverse (high volume -> low value)" if corr < -0.1 else (
                "positive" if corr > 0.1 else "no strong relationship"
            )
            print(f"\nvolume_share vs mean_value correlation: {corr:.3f} [{direction}]")

    if args.out:
        with open(args.out, "w") as f:
            for it in items:
                f.write(json.dumps(it.as_dict()) + "\n")
        print(f"\nWrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
