"""
Fast/slow path agents for the value-weighted routing project.

Both paths turn a routed item into a merchandising recommendation — one of
a fixed set of commercial actions (feature, standard, discount, bundle,
deprioritize) — using only the difficulty/value estimates already produced
upstream, never the simulator's hidden ground truth.

The fast path is a flat category lookup. The slow path has two
implementations behind the same `run()` interface: `MockSlowPathAgent`
follows a small decision rubric over the estimates (no API calls, free,
deterministic), and `ClaudeSlowPathAgent` makes a real Claude API call with
structured outputs constraining the response to the same action set. `--live`
selects the real agent; the mock is the default so routing runs don't
silently start making billed API calls.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

from value_router.simulator import Item, Simulator
from value_router.difficulty_scorer import DifficultyScorer
from value_router.value_estimator import ValueEstimator
from value_router.router import SLOW, RoutingDecision, ValueWeightedRouter, route_items

ACTIONS = ("feature", "standard", "discount", "bundle", "deprioritize")

FAST = "fast"


@dataclass
class AgentResult:
    item_id: int
    path: str  # "fast" or "slow"
    action: str
    rationale: Optional[str]  # only the slow path logs a rationale


# Flat, category-only lookup — no reasoning about the specific item, which
# is exactly what makes this path cheap.
FAST_ACTION_BY_CATEGORY: dict[str, str] = {
    "commodity": "standard",
    "accessory": "standard",
    "mid_tier": "bundle",
    "premium": "feature",
    "luxury": "feature",
}


class FastPathAgent:
    def run(self, item: Item, difficulty_estimate: float, value_estimate: float) -> AgentResult:
        action = FAST_ACTION_BY_CATEGORY.get(item.category, "standard")
        return AgentResult(item.id, FAST, action, rationale=None)


# Decision rubric the mocked "LLM" follows — a stand-in for reasoning a real
# call would do over the same two numbers.
HIGH_VALUE_THRESHOLD = 60.0
LOW_VALUE_THRESHOLD = 5.0
HIGH_DIFFICULTY_THRESHOLD = 0.6
LOW_DIFFICULTY_THRESHOLD = 0.3

RATIONALE_TEMPLATES: dict[str, str] = {
    "feature": (
        "Estimated value (${value:.2f}) is well above typical for {category}; "
        "feature it prominently to capture that upside."
    ),
    "deprioritize": (
        "Estimated value (${value:.2f}) is too low to justify shelf space; "
        "deprioritize in favor of higher-value items."
    ),
    "bundle": (
        "High estimated difficulty ({difficulty:.2f}) suggests this item is hard to "
        "sell on its own; bundle it with an easier-selling {category} item."
    ),
    "discount": (
        "Low estimated difficulty ({difficulty:.2f}) with moderate value suggests price "
        "sensitivity is the main lever; discount to move volume."
    ),
    "standard": (
        "Estimated value (${value:.2f}) and difficulty ({difficulty:.2f}) are both "
        "mid-range for {category}; a standard listing is sufficient."
    ),
}


def _decide_action(value_estimate: float, difficulty_estimate: float) -> str:
    if value_estimate >= HIGH_VALUE_THRESHOLD:
        return "feature"
    if value_estimate < LOW_VALUE_THRESHOLD:
        return "deprioritize"
    if difficulty_estimate >= HIGH_DIFFICULTY_THRESHOLD:
        return "bundle"
    if difficulty_estimate <= LOW_DIFFICULTY_THRESHOLD:
        return "discount"
    return "standard"


class MockSlowPathAgent:
    """Stand-in for a real LLM reasoning call. Deterministic given the same
    estimates, so results stay reproducible until a live call replaces it."""

    def run(self, item: Item, difficulty_estimate: float, value_estimate: float) -> AgentResult:
        action = _decide_action(value_estimate, difficulty_estimate)
        rationale = RATIONALE_TEMPLATES[action].format(
            value=value_estimate, difficulty=difficulty_estimate, category=item.category
        )
        return AgentResult(item.id, SLOW, action, rationale)


def _load_dotenv(path: str = ".env") -> None:
    """Load ANTHROPIC_API_KEY (and any other vars) from a .env file into the
    environment, without pulling in a python-dotenv dependency for one line."""
    if os.environ.get("ANTHROPIC_API_KEY") or not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


class ClaudeSlowPathAgent:
    """Real LLM reasoning call for the slow path, via the Claude API.

    Same run() interface as MockSlowPathAgent, so it drops in without
    touching call sites. Uses structured outputs to constrain the response
    to the fixed action set instead of parsing free-form text.
    """

    def __init__(self, model: str = "claude-opus-4-8"):
        import anthropic  # imported lazily so mock-only runs don't need the package installed

        self._client = anthropic.Anthropic()
        self._model = model

    def run(self, item: Item, difficulty_estimate: float, value_estimate: float) -> AgentResult:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": list(ACTIONS)},
                            "rationale": {"type": "string"},
                        },
                        "required": ["action", "rationale"],
                        "additionalProperties": False,
                    },
                }
            },
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are deciding how to merchandise a retail item. Choose exactly "
                        f"one action from {list(ACTIONS)} and give a one-sentence rationale "
                        "grounded in the numbers below.\n\n"
                        f"category: {item.category}\n"
                        f"price: ${item.price:.2f}\n"
                        f"estimated value (expected profit): ${value_estimate:.2f}\n"
                        f"estimated difficulty (0-1, how hard this item is to route well): "
                        f"{difficulty_estimate:.2f}"
                    ),
                }
            ],
        )
        text = next(b.text for b in response.content if b.type == "text")
        parsed = json.loads(text)
        return AgentResult(item.id, SLOW, parsed["action"], parsed["rationale"])


def run_agents(
    items: list[Item],
    decisions: list[RoutingDecision],
    fast_agent: FastPathAgent,
    slow_agent: MockSlowPathAgent,
) -> list[AgentResult]:
    results = []
    for item, decision in zip(items, decisions):
        agent = slow_agent if decision.path == SLOW else fast_agent
        results.append(agent.run(item, decision.difficulty_estimate, decision.value_estimate))
    return results


def summarize(results: list[AgentResult]) -> dict:
    by_path: dict[str, dict[str, int]] = {FAST: {a: 0 for a in ACTIONS}, SLOW: {a: 0 for a in ACTIONS}}
    for r in results:
        by_path[r.path][r.action] += 1
    return by_path


def main():
    parser = argparse.ArgumentParser(description="Run fast/slow path agents over routed items.")
    parser.add_argument("-n", type=int, default=2000, help="number of items to generate")
    parser.add_argument("--seed", type=int, default=42, help="simulator seed")
    parser.add_argument("--scorer-seed", type=int, default=7)
    parser.add_argument("--estimator-seed", type=int, default=11)
    parser.add_argument("--difficulty-threshold", type=float, default=0.5)
    parser.add_argument("--value-threshold", type=float, default=20.0)
    parser.add_argument("--out", type=str, default=None, help="optional path to write agent results as JSONL")
    parser.add_argument(
        "--live",
        action="store_true",
        help="use a real Claude API call for the slow path instead of the mock (makes billed requests)",
    )
    args = parser.parse_args()

    sim = Simulator(seed=args.seed)
    items = sim.generate_batch(args.n)

    difficulty_estimates = DifficultyScorer(seed=args.scorer_seed).score_batch(items)
    value_estimates = ValueEstimator(seed=args.estimator_seed).estimate_batch(items)

    router = ValueWeightedRouter(args.difficulty_threshold, args.value_threshold)
    decisions = route_items(items, difficulty_estimates, value_estimates, router)

    if args.live:
        _load_dotenv()
        slow_agent = ClaudeSlowPathAgent()
    else:
        slow_agent = MockSlowPathAgent()

    results = run_agents(items, decisions, FastPathAgent(), slow_agent)

    by_path = summarize(results)
    print(f"Ran agents on {len(results)} items\n")
    for path in (FAST, SLOW):
        total = sum(by_path[path].values())
        print(f"{path} path ({total} items):")
        for action in ACTIONS:
            count = by_path[path][action]
            if count:
                print(f"  {action:<13}{count}")

    slow_examples = [r for r in results if r.path == SLOW][:3]
    if slow_examples:
        print("\nsample slow-path rationales:")
        for r in slow_examples:
            print(f"  [{r.item_id}] {r.action}: {r.rationale}")

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r.__dict__) + "\n")
        print(f"\nWrote {len(results)} agent results to {args.out}")


if __name__ == "__main__":
    main()
