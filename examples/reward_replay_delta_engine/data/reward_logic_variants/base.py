"""Base reward-code adjustment used by the Reward Replay Delta Engine demo."""

from __future__ import annotations


REWARD_LOGIC_VERSION = "code-v1-base"


def adjust_score(
    score: float,
    *,
    answer_exact: bool,  # noqa: ARG001
    answer_contains_ground_truth: bool,  # noqa: ARG001
    action_count: int,  # noqa: ARG001
    tool_action_count: int,  # noqa: ARG001
) -> tuple[float, str]:
    return score, "base reward code"
