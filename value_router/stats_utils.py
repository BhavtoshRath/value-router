"""Small statistics helpers shared across value-router modules."""

from __future__ import annotations

import statistics
from typing import Optional


def correlation(xs: list[float], ys: list[float]) -> Optional[float]:
    """Pearson correlation, with a manual fallback for Python < 3.8
    (statistics.correlation was added in 3.8)."""
    if hasattr(statistics, "correlation"):
        return statistics.correlation(xs, ys)
    n = len(xs)
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    std_x = statistics.pstdev(xs)
    std_y = statistics.pstdev(ys)
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)
