# value-router

Simulating value-weighted routing decisions over a stream of items.

The core idea: items have an observable **category** and **price**, and a
latent, hard-to-observe **value** (`price * margin`) and **difficulty**. A
router has to make good decisions per item without knowing value/difficulty
directly — this project builds the pieces needed to study that, starting
with a synthetic data generator.

## Example

Running the simulator (`-n 2000 --seed 42`) produces this per-category
breakdown:

| category  | volume % | mean value | mean price |
|-----------|---------:|-----------:|-----------:|
| commodity |    39.5% |       0.77 |       8.95 |
| accessory |    32.4% |       3.83 |      25.50 |
| mid_tier  |    17.2% |      20.48 |      95.56 |
| premium   |     7.8% |      92.92 |     323.36 |
| luxury    |     3.1% |     552.44 |    1391.97 |

`luxury` items are only 3% of traffic, but each one is worth ~700x more
than a `commodity` item. A router only observes `category`/`price` per
item — `value` and `difficulty` are latent. If a router optimizes for
"handle the most items well" (a volume-weighted objective), it naturally
spends its effort on `commodity`/`accessory` (72% of traffic) and can
end up making cheap decisions on the rare `luxury` item that's worth
more than the other 700 items combined. A value-weighted router instead
has to weight decision quality by expected value, not by frequency, so
it doesn't silently starve that thin, high-value tail. `--no-inverse`
removes this effect (flat category weights) as a control.

## Status

**Tier 1: core routing MVP** — done. Item simulator, difficulty scorer,
value estimator, router (value-weighted + baselines), fast/slow path
agents, and an eval harness comparing all three routing strategies against
ground truth.

**Tier 2: decision logging & monitoring** — done. See below.

## Item simulator

`value_router/simulator.py` generates items across five categories
(`commodity`, `accessory`, `mid_tier`, `premium`, `luxury`) with a
deliberate **inverse correlation between volume and value**: high-volume
categories (e.g. commodity) are low price/margin, while rare categories
(e.g. luxury) are high price/margin. This mirrors common e-commerce
distributions and is the scenario that stresses a naive router — it can
silently starve the low-volume/high-value segment of budget.

Each item has:
- `category` — one of the five category buckets
- `price`, `margin` — sampled per category
- `value` — ground truth `price * margin`, the quantity a value estimator
  will later try to approximate
- `difficulty` — 0..1, sampled independently of value within a category's
  range; represents how hard it is to make a good routing decision for
  that item

### Usage

```bash
python -m value_router.simulator -n 2000 --seed 42
```

Options:
- `-n` — number of items to generate (default: 2000)
- `--seed` — RNG seed for reproducibility (default: 42)
- `--no-inverse` — disable the built-in volume/value inverse correlation
  (control condition: all categories re-weighted equally)
- `--out PATH` — write generated items to a JSONL file

The command prints a per-category summary (volume share, mean value,
mean price, mean difficulty) and a quick correlation check between
volume share and mean value across categories.

## Decision logging & monitoring

Tier 2 adds observability on top of the Tier 1 pipeline. It's a dashboard
sketch meant to demonstrate the "don't silently starve the low-value
segment" concern, not a production monitoring system.

`value_router/decision_logger.py` runs simulate -> estimate -> route for a
single strategy and writes one JSONL line per item: the estimates the
router acted on, the routing decision, its cost, and the simulator's hidden
ground truth (value/difficulty). Ground truth is only available because
this is a simulation — logging it alongside the decision is what lets the
monitor check calibration.

```bash
python -m value_router.decision_logger -n 2000 --seed 42 \
  --strategy value_weighted --out decision_log.jsonl
```

`value_router/monitor.py` reads that log and prints:
- **spend by category** — volume share vs. spend share, flagging categories
  that are under- or over-spent relative to their share of traffic
- **calibration check** — mean absolute error, mean absolute % error, and
  correlation between each estimate (value, difficulty) and its ground
  truth, overall and broken down by category

```bash
python -m value_router.monitor --log decision_log.jsonl
```

On the default synthetic data, `value_weighted` and `difficulty_only` both
route 100% of `premium`/`luxury` items (10.9% of volume) to the slow path,
so those two categories absorb ~67% of total spend — the intended
concentration of effort on the high-value tail. `random` routing, by
contrast, spends roughly proportional to volume, since it ignores value and
difficulty entirely.