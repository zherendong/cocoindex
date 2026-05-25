"""Reward-code variant that promotes useful tool traces without changing the judge."""

from __future__ import annotations


REWARD_LOGIC_VERSION = "code-v2-tool-trace-bonus"


def adjust_score(
    score: float,
    *,
    answer_exact: bool,
    answer_contains_ground_truth: bool,
    action_count: int,
    tool_action_count: int,
) -> tuple[float, str]:
    if answer_exact and tool_action_count > 0 and action_count > 4:
        return score + 0.22, "promote long exact tool trace"
    if answer_contains_ground_truth and tool_action_count > 0:
        return score + 0.25, "promote grounded partial-answer tool trace"
    return score, "tool-trace code path made no adjustment"
