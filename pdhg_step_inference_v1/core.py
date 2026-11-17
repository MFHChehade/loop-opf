"""Small dependency-free helpers for the PDHG step-pair inference experiment."""
from __future__ import annotations


def choose_depth(measurements: dict[int, float], budget_ms: float) -> int | None:
    """Largest measured depth whose median batch latency is within budget."""
    eligible = [depth for depth, latency in measurements.items() if latency <= budget_ms]
    return max(eligible) if eligible else None


def paired_budget_rows(
    learned: dict[int, float], fixed: dict[int, float], budgets: list[float]
) -> list[dict]:
    return [
        {
            "budget_ms": float(budget),
            "learned_max_k": choose_depth(learned, budget),
            "fixed_max_k": choose_depth(fixed, budget),
        }
        for budget in budgets
    ]
