from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import hashlib
import html
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import cocoindex as coco
import reward_logic as _reward_logic
from cocoindex.connectors import localfs, sqlite
from cocoindex.resources.file import FileLike, PatternFilePathMatcher


BASE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
PYTHON_SOURCE_DIR = REPO_ROOT / "python"
DATA_DIR = BASE_DIR / "data"
TRAJECTORY_DIR = DATA_DIR / "trajectories"
SEED_TRAJECTORY_DIR = DATA_DIR / "seed_trajectories"
SCENARIO_TRAJECTORY_DIR = DATA_DIR / "scenario_trajectories"
POLICY_DIR = DATA_DIR / "policies"
POLICY_PATH = BASE_DIR / "reward_policy.json"
REWARD_LOGIC_PATH = BASE_DIR / "reward_logic.py"
REWARD_LOGIC_VARIANTS_DIR = DATA_DIR / "reward_logic_variants"
OUTPUT_DIR = BASE_DIR / "output"
COCOINDEX_STATE_DIR = BASE_DIR / ".cocoindex"
COCOINDEX_DB_PATH = COCOINDEX_STATE_DIR / "reward_replay"
REWARD_DB_PATH = OUTPUT_DIR / "reward_catalog.sqlite"
VERIFIER_AUDIT_PATH = OUTPUT_DIR / "verifier_calls.jsonl"
STAGE_AUDIT_PATH = OUTPUT_DIR / "stage_calls.jsonl"
DEMO_RUNS_PATH = OUTPUT_DIR / "demo_runs.json"
DASHBOARD_PATH = OUTPUT_DIR / "dashboard.html"
JUDGE_COST_PER_CALL_USD = 0.05
REFERENCE_SCALE_TRAJECTORIES = 10_000

SQLITE_DB = coco.ContextKey[sqlite.ManagedConnection](
    "reward_replay_delta_engine/sqlite"
)

_TOOL_ACTION_PREFIXES = (
    "browser:",
    "calculator:",
    "lookup:",
    "open:",
    "search:",
    "sql:",
    "tool:",
)
_ACTIVE_DEMO_RUN_ID: str | None = os.environ.get("REWARD_REPLAY_DEMO_RUN_ID")


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    task_id: str
    prompt: str
    actions: list[str]
    observations: list[str]
    tool_calls: list[dict[str, object]]
    final_answer: str
    ground_truth: str
    model: str
    model_version: str
    environment: str
    sample_temperature: float
    train_step: int
    prompt_template_version: str
    token_count: int
    baseline_label: str
    quarantined: bool


@dataclass(frozen=True)
class RewardPolicy:
    reward_version: str
    baseline_reward_version: str
    verifier_prompt: str
    pass_threshold: float
    training_threshold: float
    verifier_pass_bonus: float
    tool_use_bonus: float
    missing_tool_penalty: float
    extra_action_penalty: float
    max_actions: int
    min_success_actions: int


@dataclass(frozen=True)
class RewardFeatures:
    trajectory_id: str
    action_count: int
    tool_action_count: int
    answer_exact: bool
    answer_contains_ground_truth: bool
    answer_overlap: float
    trace_summary: str


@dataclass(frozen=True)
class VerifierLabel:
    label: str
    confidence: float
    prompt_hash: str
    rationale: str


@dataclass(frozen=True)
class RewardDecision:
    label: str
    score: float
    include_in_training: bool
    training_reason: str


@dataclass
class RewardRow:
    trajectory_id: str
    task_id: str
    reward_version: str
    reward_logic_version: str
    baseline_reward_version: str
    baseline_label: str
    label: str
    score: float
    include_in_training: bool
    verifier_label: str
    verifier_confidence: float
    changed_from_baseline: bool
    model: str
    model_version: str
    environment: str
    train_step: int
    prompt_template_version: str
    action_count: int
    tool_action_count: int
    quarantined: bool
    summary: str


@dataclass
class RewardDiffRow:
    trajectory_id: str
    reward_version: str
    baseline_reward_version: str
    previous_label: str
    new_label: str
    score: float
    include_in_training: bool
    reason: str


@dataclass(frozen=True)
class ProcessedTrajectory:
    trajectory_id: str
    task_id: str
    baseline_label: str
    label: str
    score: float
    include_in_training: bool
    changed_from_baseline: bool
    quarantined: bool
    training_record_json: str | None


@dataclass(frozen=True)
class DemoTrajectoryStatus:
    trajectory_id: str
    source: str
    parser: str
    features: str
    verifier: str
    reward: str
    catalog: str
    training: str
    label: str
    score: float | None


@dataclass(frozen=True)
class DemoRun:
    run_id: str
    title: str
    description: str
    duration_ms: int
    verifier_calls_added: int
    total_verifier_calls: int
    reward_rows_total: int
    reward_rows_added: int
    reward_rows_changed: int
    reward_rows_removed: int
    label_flips: int
    training_added: int
    training_removed: int
    training_total: int
    changed_labels: int
    reward_version: str
    reward_logic_versions: list[str]
    prompt_template_versions: list[str]
    stage_calls: dict[str, int]
    prompt_hashes: list[str]
    statuses: list[DemoTrajectoryStatus]


@contextlib.contextmanager
def _reward_db_connection() -> Iterator[sqlite.ManagedConnection]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite.managed_connection(REWARD_DB_PATH, load_vec=False) as conn:
        yield conn


@coco.lifespan
def coco_lifespan(builder: coco.EnvironmentBuilder) -> Iterator[None]:
    COCOINDEX_STATE_DIR.mkdir(parents=True, exist_ok=True)
    builder.settings.db_path = COCOINDEX_DB_PATH
    builder.provide_with(SQLITE_DB, _reward_db_connection())
    yield


def _read_json(path: pathlib.Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return _json_object_from_text(f.read(), str(path))


def _json_object_from_text(text: str, label: str) -> Mapping[str, object]:
    data: object = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return cast(Mapping[str, object], data)


def _required_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object field {key!r}")
    return cast(Mapping[str, object], value)


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string field {key!r}")
    return value


def _required_str_list(data: Mapping[str, object], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected list[str] field {key!r}")
    return list(value)


def _optional_mapping_list(
    data: Mapping[str, object], key: str
) -> list[dict[str, object]]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected list[object] field {key!r}")
    return [dict(cast(Mapping[str, object], item)) for item in value]


def _required_float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"Expected numeric field {key!r}")
    return float(value)


def _required_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer field {key!r}")
    return value


def _optional_bool(data: Mapping[str, object], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean field {key!r}")
    return value


def _optional_str(data: Mapping[str, object], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"Expected string field {key!r}")
    return value


def _optional_float(data: Mapping[str, object], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"Expected numeric field {key!r}")
    return float(value)


def _optional_int(data: Mapping[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer field {key!r}")
    return value


def _load_policy(policy_path: pathlib.Path) -> RewardPolicy:
    data = _read_json(policy_path)
    return RewardPolicy(
        reward_version=_required_str(data, "reward_version"),
        baseline_reward_version=_required_str(data, "baseline_reward_version"),
        verifier_prompt=_required_str(data, "verifier_prompt"),
        pass_threshold=_required_float(data, "pass_threshold"),
        training_threshold=_required_float(data, "training_threshold"),
        verifier_pass_bonus=_required_float(data, "verifier_pass_bonus"),
        tool_use_bonus=_required_float(data, "tool_use_bonus"),
        missing_tool_penalty=_required_float(data, "missing_tool_penalty"),
        extra_action_penalty=_required_float(data, "extra_action_penalty"),
        max_actions=_required_int(data, "max_actions"),
        min_success_actions=_required_int(data, "min_success_actions"),
    )


def _reward_logic_source() -> str:
    return REWARD_LOGIC_PATH.read_text(encoding="utf-8")


def _load_trajectory(data: Mapping[str, object]) -> Trajectory:
    metadata = _required_mapping(data, "metadata")
    return Trajectory(
        trajectory_id=_required_str(data, "trajectory_id"),
        task_id=_required_str(data, "task_id"),
        prompt=_required_str(data, "prompt"),
        actions=_required_str_list(data, "actions"),
        observations=_required_str_list(data, "observations"),
        tool_calls=_optional_mapping_list(data, "tool_calls"),
        final_answer=_required_str(data, "final_answer"),
        ground_truth=_required_str(data, "ground_truth"),
        model=_required_str(metadata, "model"),
        model_version=_optional_str(metadata, "model_version", "unknown"),
        environment=_required_str(metadata, "environment"),
        sample_temperature=_optional_float(metadata, "sample_temperature", 0.0),
        train_step=_optional_int(metadata, "train_step", 0),
        prompt_template_version=_optional_str(
            metadata, "prompt_template_version", "unknown"
        ),
        token_count=_optional_int(metadata, "token_count", 0),
        baseline_label=_required_str(metadata, "baseline_label"),
        quarantined=_optional_bool(data, "quarantined", False),
    )


def _normalize_answer(text: str) -> str:
    chars = [ch.lower() for ch in text if ch.isalnum() or ch.isspace()]
    return " ".join("".join(chars).split())


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _normalize_answer(text).split()
        if token not in {"a", "an", "is", "of", "the", "to"}
    }


def _answer_overlap(final_answer: str, ground_truth: str) -> float:
    truth_tokens = _tokens(ground_truth)
    if not truth_tokens:
        return 0.0
    answer_tokens = _tokens(final_answer)
    return len(answer_tokens & truth_tokens) / len(truth_tokens)


def _count_tool_actions(actions: Sequence[str]) -> int:
    return sum(
        1
        for action in actions
        if action.strip().lower().startswith(_TOOL_ACTION_PREFIXES)
    )


def _append_jsonl(path: pathlib.Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True))
        f.write("\n")


def _append_stage_audit(stage: str, trajectory_id: str) -> None:
    if _ACTIVE_DEMO_RUN_ID is None:
        return
    _append_jsonl(
        STAGE_AUDIT_PATH,
        {
            "at": datetime.datetime.now(datetime.UTC).isoformat(),
            "run_id": _ACTIVE_DEMO_RUN_ID,
            "stage": stage,
            "trajectory_id": trajectory_id,
        },
    )


@coco.fn(memo=True)
async def parse_trajectory(file: FileLike[pathlib.Path]) -> Trajectory:
    trajectory = _load_trajectory(
        _json_object_from_text(
            await file.read_text(),
            file.file_path.path.as_posix(),
        )
    )
    _append_stage_audit("parse", trajectory.trajectory_id)
    return trajectory


@coco.fn(memo=True)
def extract_features(trajectory: Trajectory) -> RewardFeatures:
    _append_stage_audit("features", trajectory.trajectory_id)
    normalized_answer = _normalize_answer(trajectory.final_answer)
    normalized_truth = _normalize_answer(trajectory.ground_truth)
    answer_contains_ground_truth = bool(normalized_truth) and (
        normalized_truth in normalized_answer
    )
    action_count = len(trajectory.actions)
    tool_action_count = (
        len(trajectory.tool_calls)
        if trajectory.tool_calls
        else _count_tool_actions(trajectory.actions)
    )
    return RewardFeatures(
        trajectory_id=trajectory.trajectory_id,
        action_count=action_count,
        tool_action_count=tool_action_count,
        answer_exact=normalized_answer == normalized_truth,
        answer_contains_ground_truth=answer_contains_ground_truth,
        answer_overlap=_answer_overlap(
            trajectory.final_answer, trajectory.ground_truth
        ),
        trace_summary=(
            f"{trajectory.task_id}: {action_count} actions, "
            f"{tool_action_count} tool calls, model={trajectory.model_version}, "
            f"step={trajectory.train_step}, final={trajectory.final_answer!r}"
        ),
    )


@coco.fn(memo=True, memo_key={"audit_path": None})
async def run_verifier(
    features: RewardFeatures,
    verifier_prompt: str,
    audit_path: pathlib.Path,
) -> VerifierLabel:
    _append_stage_audit("verifier", features.trajectory_id)
    await asyncio.sleep(0.25)
    prompt_hash = hashlib.sha256(verifier_prompt.encode()).hexdigest()[:10]
    strict = "strict" in verifier_prompt.lower()
    if strict:
        passed = (
            features.answer_exact
            and features.tool_action_count > 0
            and features.action_count <= 4
        )
    else:
        passed = features.answer_exact or features.answer_contains_ground_truth

    label = "pass" if passed else "fail"
    confidence = (
        0.96 if features.answer_exact else round(0.45 + features.answer_overlap, 2)
    )
    rationale = (
        "exact answer accepted"
        if features.answer_exact
        else f"answer overlap={features.answer_overlap:.2f}"
    )
    await asyncio.to_thread(
        _append_jsonl,
        audit_path,
        {
            "at": datetime.datetime.now(datetime.UTC).isoformat(),
            "run_id": _ACTIVE_DEMO_RUN_ID,
            "trajectory_id": features.trajectory_id,
            "prompt_hash": prompt_hash,
            "label": label,
        },
    )
    return VerifierLabel(
        label=label,
        confidence=confidence,
        prompt_hash=prompt_hash,
        rationale=rationale,
    )


@coco.fn(memo=True, deps={"reward_logic.py": _reward_logic_source()})
def score_reward(
    trajectory: Trajectory,
    features: RewardFeatures,
    verifier: VerifierLabel,
    policy: RewardPolicy,
) -> RewardDecision:
    _append_stage_audit("reward", trajectory.trajectory_id)
    if trajectory.quarantined:
        return RewardDecision(
            label="fail",
            score=0.0,
            include_in_training=False,
            training_reason="trajectory is quarantined",
        )

    score = 0.0
    if features.answer_exact:
        score += 0.65
    elif features.answer_contains_ground_truth:
        score += 0.55
    else:
        score += features.answer_overlap * 0.2

    if verifier.label == "pass":
        score += policy.verifier_pass_bonus

    if features.tool_action_count > 0:
        score += policy.tool_use_bonus
    else:
        score -= policy.missing_tool_penalty

    if features.action_count < policy.min_success_actions:
        score -= policy.missing_tool_penalty

    extra_actions = max(0, features.action_count - policy.max_actions)
    score -= extra_actions * policy.extra_action_penalty
    score, code_reason = _reward_logic.adjust_score(
        score,
        answer_exact=features.answer_exact,
        answer_contains_ground_truth=features.answer_contains_ground_truth,
        action_count=features.action_count,
        tool_action_count=features.tool_action_count,
    )
    score = round(min(1.0, max(0.0, score)), 3)

    label = "pass" if score >= policy.pass_threshold else "fail"
    include_in_training = label == "pass" and score >= policy.training_threshold
    base_training_reason = (
        "eligible for training export"
        if include_in_training
        else f"below training threshold {policy.training_threshold:.2f}"
    )
    training_reason = f"{base_training_reason}; reward logic: {code_reason}"
    return RewardDecision(
        label=label,
        score=score,
        include_in_training=include_in_training,
        training_reason=training_reason,
    )


def _training_record_json(
    trajectory: Trajectory,
    decision: RewardDecision,
    policy: RewardPolicy,
) -> str:
    return json.dumps(
        {
            "trajectory_id": trajectory.trajectory_id,
            "task_id": trajectory.task_id,
            "reward_version": policy.reward_version,
            "reward_logic_version": _reward_logic.REWARD_LOGIC_VERSION,
            "messages": [
                {"role": "user", "content": trajectory.prompt},
                {"role": "assistant", "content": trajectory.final_answer},
            ],
            "tool_calls": trajectory.tool_calls,
            "metadata": {
                "model": trajectory.model,
                "model_version": trajectory.model_version,
                "environment": trajectory.environment,
                "sample_temperature": trajectory.sample_temperature,
                "train_step": trajectory.train_step,
                "prompt_template_version": trajectory.prompt_template_version,
                "token_count": trajectory.token_count,
            },
            "reward": decision.score,
            "label": decision.label,
        },
        sort_keys=True,
    )


@coco.fn(memo=True)
async def process_trajectory(
    file: FileLike[pathlib.Path],
    policy: RewardPolicy,
    reward_table: sqlite.TableTarget[RewardRow],
    diff_table: sqlite.TableTarget[RewardDiffRow],
    output_target: localfs.DirTarget,
    verifier_audit_path: pathlib.Path,
) -> ProcessedTrajectory:
    trajectory = await parse_trajectory(file)
    features = extract_features(trajectory)
    verifier = await run_verifier(
        features,
        policy.verifier_prompt,
        verifier_audit_path,
    )
    decision = score_reward(trajectory, features, verifier, policy)
    changed_from_baseline = decision.label != trajectory.baseline_label
    reward_table.declare_row(
        row=RewardRow(
            trajectory_id=trajectory.trajectory_id,
            task_id=trajectory.task_id,
            reward_version=policy.reward_version,
            reward_logic_version=_reward_logic.REWARD_LOGIC_VERSION,
            baseline_reward_version=policy.baseline_reward_version,
            baseline_label=trajectory.baseline_label,
            label=decision.label,
            score=decision.score,
            include_in_training=decision.include_in_training,
            verifier_label=verifier.label,
            verifier_confidence=verifier.confidence,
            changed_from_baseline=changed_from_baseline,
            model=trajectory.model,
            model_version=trajectory.model_version,
            environment=trajectory.environment,
            train_step=trajectory.train_step,
            prompt_template_version=trajectory.prompt_template_version,
            action_count=features.action_count,
            tool_action_count=features.tool_action_count,
            quarantined=trajectory.quarantined,
            summary=features.trace_summary,
        )
    )

    if changed_from_baseline:
        diff_table.declare_row(
            row=RewardDiffRow(
                trajectory_id=trajectory.trajectory_id,
                reward_version=policy.reward_version,
                baseline_reward_version=policy.baseline_reward_version,
                previous_label=trajectory.baseline_label,
                new_label=decision.label,
                score=decision.score,
                include_in_training=decision.include_in_training,
                reason=decision.training_reason,
            )
        )

    training_record_json = None
    if decision.include_in_training:
        training_record_json = _training_record_json(trajectory, decision, policy)
        output_target.declare_file(
            pathlib.PurePosixPath("training_rows")
            / f"{trajectory.trajectory_id}.jsonl",
            training_record_json + "\n",
            create_parent_dirs=True,
        )

    return ProcessedTrajectory(
        trajectory_id=trajectory.trajectory_id,
        task_id=trajectory.task_id,
        baseline_label=trajectory.baseline_label,
        label=decision.label,
        score=decision.score,
        include_in_training=decision.include_in_training,
        changed_from_baseline=changed_from_baseline,
        quarantined=trajectory.quarantined,
        training_record_json=training_record_json,
    )


def _component_subpath_for_file_key(file_key: str) -> coco.ComponentSubpath:
    parts = pathlib.PurePosixPath(file_key).with_suffix("").parts
    return coco.component_subpath("trajectory", *parts)


def _render_report(
    processed: Sequence[ProcessedTrajectory], policy: RewardPolicy
) -> str:
    total = len(processed)
    training_count = sum(1 for item in processed if item.include_in_training)
    changed = [item for item in processed if item.changed_from_baseline]
    promoted = [
        item
        for item in changed
        if item.baseline_label == "fail" and item.label == "pass"
    ]
    removed = [
        item
        for item in changed
        if item.baseline_label == "pass" and item.label == "fail"
    ]
    quarantined = sum(1 for item in processed if item.quarantined)

    lines = [
        "# Reward Replay Delta Report",
        "",
        f"Reward {policy.baseline_reward_version} -> {policy.reward_version}",
        "",
        "Changed pass/fail labels are compared with the baseline label stored "
        "in each trajectory's metadata.",
        "",
        f"- Trajectories scanned: {total}",
        f"- Reward rows maintained: {total}",
        f"- Training rows exported: {training_count}",
        f"- Changed pass/fail labels: {len(changed)}",
        f"- Newly promoted into training set: {len(promoted)}",
        f"- Removed from training set: {len(removed)}",
        f"- Quarantined trajectories: {quarantined}",
        "",
        "## Changed Labels",
        "",
    ]
    if changed:
        lines.extend(
            [
                "| Trajectory | Previous | New | Score | Training |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for item in sorted(changed, key=lambda value: value.trajectory_id):
            training = "yes" if item.include_in_training else "no"
            lines.append(
                f"| {item.trajectory_id} | {item.baseline_label} | "
                f"{item.label} | {item.score:.3f} | {training} |"
            )
    else:
        lines.append("No pass/fail labels changed from the baseline.")
    lines.append("")
    return "\n".join(lines)


@coco.fn
async def app_main(
    source_dir: pathlib.Path = TRAJECTORY_DIR,
    output_dir: pathlib.Path = OUTPUT_DIR,
    policy_path: pathlib.Path = POLICY_PATH,
) -> None:
    policy = _load_policy(policy_path)
    output_target = await localfs.mount_dir_target(output_dir)
    reward_table = await sqlite.mount_table_target(
        SQLITE_DB,
        table_name="reward_rows",
        table_schema=await sqlite.TableSchema.from_class(
            RewardRow,
            primary_key=["trajectory_id"],
        ),
    )
    diff_table = await sqlite.mount_table_target(
        SQLITE_DB,
        table_name="reward_diffs",
        table_schema=await sqlite.TableSchema.from_class(
            RewardDiffRow,
            primary_key=["trajectory_id", "reward_version"],
        ),
    )

    files = localfs.walk_dir(
        source_dir,
        recursive=True,
        path_matcher=PatternFilePathMatcher(included_patterns=["**/*.json"]),
    )
    processed: list[ProcessedTrajectory] = []
    async for file_key, file in files.items():
        processed.append(
            await coco.use_mount(
                _component_subpath_for_file_key(file_key),
                process_trajectory,
                file,
                policy,
                reward_table,
                diff_table,
                output_target,
                VERIFIER_AUDIT_PATH,
            )
        )

    processed.sort(key=lambda item: item.trajectory_id)
    training_records = [
        item.training_record_json
        for item in processed
        if item.training_record_json is not None
    ]
    output_target.declare_file(
        "training.jsonl",
        "".join(f"{record}\n" for record in training_records),
    )
    output_target.declare_file("reward_diff.md", _render_report(processed, policy))
    output_target.declare_file(
        "manifest.json",
        json.dumps(
            {
                "reward_version": policy.reward_version,
                "baseline_reward_version": policy.baseline_reward_version,
                "trajectories": len(processed),
                "training_rows": len(training_records),
                "changed_labels": sum(
                    1 for item in processed if item.changed_from_baseline
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


app = coco.App(
    coco.AppConfig(name="RewardReplayDeltaEngine"),
    app_main,
)


def _copytree_contents(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in destination.glob("*.json"):
        path.unlink()
    for path in sorted(source.glob("*.json")):
        shutil.copy2(path, destination / path.name)


def _restore_seed_inputs() -> None:
    _copytree_contents(SEED_TRAJECTORY_DIR, TRAJECTORY_DIR)
    shutil.copy2(POLICY_DIR / "balanced.json", POLICY_PATH)
    shutil.copy2(REWARD_LOGIC_VARIANTS_DIR / "base.py", REWARD_LOGIC_PATH)


def _reset_demo(*, announce: bool = True) -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if COCOINDEX_STATE_DIR.exists():
        shutil.rmtree(COCOINDEX_STATE_DIR)
    _restore_seed_inputs()
    if announce:
        print("Reset demo trajectories, reward policy, and generated output.")


def _add_new_trajectory(*, announce: bool = True) -> None:
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    source = SCENARIO_TRAJECTORY_DIR / "traj_refund_005.json"
    destination = TRAJECTORY_DIR / source.name
    shutil.copy2(source, destination)
    if announce:
        print(f"Added {destination.relative_to(BASE_DIR)}.")


def _set_policy(name: str, *, announce: bool = True) -> None:
    policy_path = POLICY_DIR / f"{name}.json"
    if not policy_path.exists():
        raise ValueError(f"Unknown policy {name!r}; expected one of: balanced, strict")
    shutil.copy2(policy_path, POLICY_PATH)
    if announce:
        print(f"Set reward policy to {name}.")


def _set_reward_logic(name: str, *, announce: bool = True) -> None:
    logic_path = REWARD_LOGIC_VARIANTS_DIR / f"{name}.py"
    if not logic_path.exists():
        raise ValueError(
            f"Unknown reward logic {name!r}; expected one of: base, tool_trace_bonus"
        )
    shutil.copy2(logic_path, REWARD_LOGIC_PATH)
    if announce:
        print(f"Set reward logic to {name}.")


def _set_quarantine(
    trajectory_id: str, quarantined: bool, *, announce: bool = True
) -> None:
    path = TRAJECTORY_DIR / f"{trajectory_id}.json"
    if not path.exists():
        raise ValueError(f"No trajectory file found for {trajectory_id!r}")
    data = dict(_read_json(path))
    data["quarantined"] = quarantined
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
    status = "quarantined" if quarantined else "unquarantined"
    if announce:
        print(f"{status.capitalize()} {trajectory_id}.")


def _read_verifier_call_count() -> int:
    if not VERIFIER_AUDIT_PATH.exists():
        return 0
    with VERIFIER_AUDIT_PATH.open("r", encoding="utf-8") as f:
        return sum(1 for _line in f)


def _print_sqlite_summary() -> None:
    if not REWARD_DB_PATH.exists():
        print("No reward catalog yet. Run: cocoindex update main.py")
        return
    with sqlite3.connect(REWARD_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT trajectory_id, label, score, include_in_training, changed_from_baseline
            FROM reward_rows
            ORDER BY trajectory_id
            """
        ).fetchall()
        print("Reward rows:")
        for row in rows:
            training = "train" if row["include_in_training"] else "skip"
            changed = "changed" if row["changed_from_baseline"] else "same"
            print(
                f"  {row['trajectory_id']}: {row['label']} "
                f"{row['score']:.3f} ({training}, {changed})"
            )


def _show_outputs() -> None:
    report_path = OUTPUT_DIR / "reward_diff.md"
    if report_path.exists():
        print(report_path.read_text(encoding="utf-8").rstrip())
        print()
    _print_sqlite_summary()
    print(f"Judge calls recorded: {_read_verifier_call_count()}")
    training_path = OUTPUT_DIR / "training.jsonl"
    if training_path.exists():
        print(f"Training export: {training_path.relative_to(BASE_DIR)}")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _reward_row_snapshot() -> dict[str, dict[str, object]]:
    if not REWARD_DB_PATH.exists():
        return {}
    with sqlite3.connect(REWARD_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if not _table_exists(conn, "reward_rows"):
            return {}
        rows = conn.execute(
            """
            SELECT trajectory_id, label, score, include_in_training,
                   changed_from_baseline, reward_version, reward_logic_version,
                   prompt_template_version
            FROM reward_rows
            ORDER BY trajectory_id
            """
        ).fetchall()
    return {
        str(row["trajectory_id"]): {key: row[key] for key in row.keys()} for row in rows
    }


def _training_snapshot() -> set[str]:
    training_dir = OUTPUT_DIR / "training_rows"
    if not training_dir.exists():
        return set()
    return {path.stem for path in training_dir.glob("*.jsonl")}


def _read_verifier_audit_records() -> list[Mapping[str, object]]:
    if not VERIFIER_AUDIT_PATH.exists():
        return []
    records: list[Mapping[str, object]] = []
    with VERIFIER_AUDIT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(cast(Mapping[str, object], record))
    return records


def _read_stage_audit_records() -> list[Mapping[str, object]]:
    if not STAGE_AUDIT_PATH.exists():
        return []
    records: list[Mapping[str, object]] = []
    with STAGE_AUDIT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(cast(Mapping[str, object], record))
    return records


def _stage_trajectory_ids(
    records: Sequence[Mapping[str, object]], stage: str
) -> set[str]:
    return {
        str(record["trajectory_id"])
        for record in records
        if record.get("stage") == stage and isinstance(record.get("trajectory_id"), str)
    }


def _stage_call_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {"parse": 0, "features": 0, "verifier": 0, "reward": 0}
    for record in records:
        stage = record.get("stage")
        if isinstance(stage, str):
            counts[stage] = counts.get(stage, 0) + 1
    return counts


def _prompt_hashes(records: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(
        {
            str(record["prompt_hash"])
            for record in records
            if isinstance(record.get("prompt_hash"), str)
        }
    )


def _row_values(rows: Mapping[str, Mapping[str, object]], key: str) -> list[str]:
    return sorted(
        {
            str(row[key])
            for row in rows.values()
            if isinstance(row.get(key), str) and row.get(key)
        }
    )


def _run_cocoindex_update(run_id: str) -> None:
    env = os.environ.copy()
    env["REWARD_REPLAY_DEMO_RUN_ID"] = run_id
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_parts = [str(PYTHON_SOURCE_DIR)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cocoindex.cli",
            "update",
            "-q",
            "main.py",
        ],
        cwd=BASE_DIR,
        env=env,
        check=True,
    )


def _active_reward_version(rows: Mapping[str, Mapping[str, object]]) -> str:
    for row in rows.values():
        value = row.get("reward_version")
        if isinstance(value, str):
            return value
    return _load_policy(POLICY_PATH).reward_version


def _row_label(row: Mapping[str, object] | None) -> str:
    if row is None:
        return "-"
    value = row.get("label")
    return str(value) if value is not None else "-"


def _row_score(row: Mapping[str, object] | None) -> float | None:
    if row is None:
        return None
    value = row.get("score")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _training_status(trajectory_id: str, before: set[str], after: set[str]) -> str:
    if trajectory_id in after and trajectory_id not in before:
        return "created"
    if trajectory_id not in after and trajectory_id in before:
        return "removed"
    if trajectory_id in after:
        return "kept"
    return "none"


def _source_status(trajectory_id: str, created: bool, parse_ids: set[str]) -> str:
    if created:
        return "new"
    if trajectory_id in parse_ids:
        return "edited"
    return "same"


def _stage_status(trajectory_id: str, stage_ids: set[str]) -> str:
    if trajectory_id in stage_ids:
        return "ran"
    return "cached"


def _reward_status(trajectory_id: str, created: bool, reward_ids: set[str]) -> str:
    if trajectory_id not in reward_ids:
        return "cached"
    if created:
        return "computed"
    return "recomputed"


def _verifier_status(trajectory_id: str, created: bool, verifier_ids: set[str]) -> str:
    if trajectory_id not in verifier_ids:
        return "cached"
    if created:
        return "computed"
    return "reran"


def _catalog_status(created: bool, removed: bool, row_changed: bool) -> str:
    if created:
        return "inserted"
    if removed:
        return "deleted"
    if row_changed:
        return "updated"
    return "same"


def _build_trajectory_statuses(
    before_rows: Mapping[str, Mapping[str, object]],
    after_rows: Mapping[str, Mapping[str, object]],
    before_training: set[str],
    after_training: set[str],
    stage_records: Sequence[Mapping[str, object]],
) -> list[DemoTrajectoryStatus]:
    parse_ids = _stage_trajectory_ids(stage_records, "parse")
    features_ids = _stage_trajectory_ids(stage_records, "features")
    verifier_ids = _stage_trajectory_ids(stage_records, "verifier")
    reward_ids = _stage_trajectory_ids(stage_records, "reward")
    all_ids = sorted(
        set(before_rows)
        | set(after_rows)
        | before_training
        | after_training
        | parse_ids
        | features_ids
        | verifier_ids
        | reward_ids
    )
    statuses: list[DemoTrajectoryStatus] = []
    for trajectory_id in all_ids:
        before_row = before_rows.get(trajectory_id)
        after_row = after_rows.get(trajectory_id)
        created = before_row is None and after_row is not None
        removed = before_row is not None and after_row is None
        row_changed = before_row != after_row
        statuses.append(
            DemoTrajectoryStatus(
                trajectory_id=trajectory_id,
                source=_source_status(trajectory_id, created, parse_ids),
                parser=_stage_status(trajectory_id, parse_ids),
                features=_stage_status(trajectory_id, features_ids),
                verifier=_verifier_status(trajectory_id, created, verifier_ids),
                reward=_reward_status(trajectory_id, created, reward_ids),
                catalog=_catalog_status(created, removed, row_changed),
                training=_training_status(
                    trajectory_id, before_training, after_training
                ),
                label=_row_label(after_row),
                score=_row_score(after_row),
            )
        )
    return statuses


def _run_update_step(run_id: str, title: str, description: str) -> DemoRun:
    before_rows = _reward_row_snapshot()
    before_training = _training_snapshot()
    before_audit = _read_verifier_audit_records()
    before_stage = _read_stage_audit_records()

    start = time.perf_counter()
    _run_cocoindex_update(run_id)
    duration_ms = int((time.perf_counter() - start) * 1000)

    after_rows = _reward_row_snapshot()
    after_training = _training_snapshot()
    after_audit = _read_verifier_audit_records()
    after_stage = _read_stage_audit_records()
    new_audit_records = after_audit[len(before_audit) :]
    new_stage_records = after_stage[len(before_stage) :]

    before_ids = set(before_rows)
    after_ids = set(after_rows)
    shared_ids = before_ids & after_ids
    reward_rows_added = len(after_ids - before_ids)
    reward_rows_removed = len(before_ids - after_ids)
    reward_rows_changed = sum(
        1 for key in shared_ids if before_rows[key] != after_rows[key]
    )
    label_flips = sum(
        1
        for key in shared_ids
        if before_rows[key].get("label") != after_rows[key].get("label")
    )
    changed_labels = sum(
        1 for row in after_rows.values() if bool(row.get("changed_from_baseline"))
    )

    return DemoRun(
        run_id=run_id,
        title=title,
        description=description,
        duration_ms=duration_ms,
        verifier_calls_added=len(new_audit_records),
        total_verifier_calls=len(after_audit),
        reward_rows_total=len(after_rows),
        reward_rows_added=reward_rows_added,
        reward_rows_changed=reward_rows_changed,
        reward_rows_removed=reward_rows_removed,
        label_flips=label_flips,
        training_added=len(after_training - before_training),
        training_removed=len(before_training - after_training),
        training_total=len(after_training),
        changed_labels=changed_labels,
        reward_version=_active_reward_version(after_rows),
        reward_logic_versions=_row_values(after_rows, "reward_logic_version"),
        prompt_template_versions=_row_values(after_rows, "prompt_template_version"),
        stage_calls=_stage_call_counts(new_stage_records),
        prompt_hashes=_prompt_hashes(new_audit_records),
        statuses=_build_trajectory_statuses(
            before_rows,
            after_rows,
            before_training,
            after_training,
            new_stage_records,
        ),
    )


def _demo_status_to_dict(status: DemoTrajectoryStatus) -> dict[str, object]:
    return {
        "trajectory_id": status.trajectory_id,
        "source": status.source,
        "parser": status.parser,
        "features": status.features,
        "verifier": status.verifier,
        "reward": status.reward,
        "catalog": status.catalog,
        "training": status.training,
        "label": status.label,
        "score": status.score,
    }


def _demo_run_to_dict(run: DemoRun) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "title": run.title,
        "description": run.description,
        "duration_ms": run.duration_ms,
        "verifier_calls_added": run.verifier_calls_added,
        "total_verifier_calls": run.total_verifier_calls,
        "reward_rows_total": run.reward_rows_total,
        "reward_rows_added": run.reward_rows_added,
        "reward_rows_changed": run.reward_rows_changed,
        "reward_rows_removed": run.reward_rows_removed,
        "label_flips": run.label_flips,
        "training_added": run.training_added,
        "training_removed": run.training_removed,
        "training_total": run.training_total,
        "changed_labels": run.changed_labels,
        "reward_version": run.reward_version,
        "reward_logic_versions": run.reward_logic_versions,
        "prompt_template_versions": run.prompt_template_versions,
        "stage_calls": run.stage_calls,
        "prompt_hashes": run.prompt_hashes,
        "statuses": [_demo_status_to_dict(status) for status in run.statuses],
    }


def _h(value: object) -> str:
    return html.escape(str(value), quote=True)


def _status_class(value: str) -> str:
    if value in {"cached", "same", "kept", "none"}:
        return "quiet"
    if value in {"new", "created", "computed", "inserted", "ran"}:
        return "created"
    if value in {"reran", "recomputed", "updated", "edited"}:
        return "updated"
    if value in {"removed", "deleted"}:
        return "removed"
    return "neutral"


def _render_status_pill(value: str) -> str:
    return f'<span class="pill {_status_class(value)}">{_h(value)}</span>'


def _render_run_card(run: DemoRun, index: int, max_duration_ms: int) -> str:
    duration_width = max(4, int(run.duration_ms / max(max_duration_ms, 1) * 100))
    changed_rows = (
        run.reward_rows_changed + run.reward_rows_added + run.reward_rows_removed
    )
    return f"""
      <div class="run-card">
        <span class="run-index">{index + 1:02d}</span>
        <span class="run-copy">
          <strong>{_h(run.title)}</strong>
          <small>{_h(run.description)}</small>
          <span class="duration-mini"><span style="width:{duration_width}%"></span></span>
        </span>
        <span class="run-kpis">
          <span><b>{run.duration_ms}</b> ms</span>
          <span><b>{run.verifier_calls_added}</b> judge calls</span>
          <span><b>{run.stage_calls.get("reward", 0)}</b> reward calls</span>
          <span><b>{changed_rows}</b> row updates</span>
        </span>
      </div>
    """


def _render_status_table(run: DemoRun) -> str:
    rows = []
    for status in run.statuses:
        score = "-" if status.score is None else f"{status.score:.3f}"
        rows.append(
            f"""
            <tr>
              <td class="mono">{_h(status.trajectory_id)}</td>
              <td>{_render_status_pill(status.parser)}</td>
              <td>{_render_status_pill(status.features)}</td>
              <td>{_render_status_pill(status.verifier)}</td>
              <td>{_render_status_pill(status.reward)}</td>
              <td>{_render_status_pill(status.catalog)}</td>
              <td>{_render_status_pill(status.training)}</td>
              <td><span class="label-badge {status.label}">{_h(status.label)}</span></td>
              <td class="numeric">{score}</td>
            </tr>
            """
        )
    return f"""
      <div class="run-panel">
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Trajectory</th>
                <th>Parser</th>
                <th>Features</th>
                <th>Judge</th>
                <th>Reward</th>
                <th>Catalog</th>
                <th>Training</th>
                <th>Label</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </div>
      </div>
    """


def _render_run_section(run: DemoRun, index: int, max_duration_ms: int) -> str:
    return f"""
      <section class="run-stack" id="{_h(run.run_id)}">
        {_render_run_card(run, index, max_duration_ms)}
        {_render_status_table(run)}
      </section>
    """


def _short_trajectory_id(trajectory_id: str) -> str:
    return trajectory_id.removeprefix("traj_")


def _heatmap_state(status: DemoTrajectoryStatus | None) -> str:
    if status is None:
        return "none"
    compute_stage_active = [
        status.parser == "ran",
        status.features == "ran",
        status.verifier in {"computed", "reran"},
        status.reward in {"computed", "recomputed"},
    ]
    if all(compute_stage_active):
        return "full"
    if (
        any(compute_stage_active)
        or status.source in {"new", "edited"}
        or status.catalog in {"inserted", "updated", "deleted"}
        or status.training in {"created", "removed"}
    ):
        return "partial"
    return "none"


def _render_delta_heatmap(runs: Sequence[DemoRun]) -> str:
    trajectory_ids = sorted(
        {status.trajectory_id for run in runs for status in run.statuses}
    )
    header_cells = "".join(
        f'<div class="heatmap-label mono" title="{_h(trajectory_id)}">'
        f"{_h(_short_trajectory_id(trajectory_id))}</div>"
        for trajectory_id in trajectory_ids
    )
    rows = []
    for run in runs:
        statuses = {status.trajectory_id: status for status in run.statuses}
        cells = []
        for trajectory_id in trajectory_ids:
            status = statuses.get(trajectory_id)
            state = _heatmap_state(status)
            title = (
                f"{run.title} / {trajectory_id}: {state}"
                if status is None
                else (
                    f"{run.title} / {trajectory_id}: {state}; "
                    f"parser={status.parser}, features={status.features}, "
                    f"judge={status.verifier}, reward={status.reward}, "
                    f"catalog={status.catalog}, training={status.training}"
                )
            )
            cells.append(
                f'<div class="heatmap-cell {state}" title="{_h(title)}">'
                f'<span class="sr-only">{_h(title)}</span></div>'
            )
        rows.append(
            f"""
            <div class="heatmap-run">{_h(run.title)}</div>
            {"".join(cells)}
            """
        )
    return f"""
      <div class="heatmap-wrap" aria-label="Run-by-trajectory delta heatmap">
        <div class="heatmap-grid" style="--cols: {len(trajectory_ids)}">
          <div class="heatmap-corner">Run</div>
          {header_cells}
          {"".join(rows)}
        </div>
        <div class="heatmap-legend">
          <span><i class="none"></i>No work</span>
          <span><i class="partial"></i>Partial replay</span>
          <span><i class="full"></i>Full compute path</span>
        </div>
      </div>
    """


def _render_training_bars(runs: Sequence[DemoRun]) -> str:
    max_training = max((run.training_total for run in runs), default=1)
    max_training = max(max_training, 1)
    bars = []
    for run in runs:
        width = max(6, int(run.training_total / max_training * 100))
        bars.append(
            f"""
            <div class="bar-row">
              <span>{_h(run.title)}</span>
              <div class="bar-track"><div class="bar-fill" style="width:{width}%"></div></div>
              <b>{run.training_total}</b>
            </div>
            """
        )
    return "".join(bars)


def _render_hashes(values: Sequence[str], *, empty: str = "cached") -> str:
    if not values:
        return f'<span class="hash-list"><span class="muted">{_h(empty)}</span></span>'
    hashes = "".join(f'<span class="hash">{_h(value)}</span>' for value in values)
    return f'<span class="hash-list">{hashes}</span>'


def _prompt_template_badge(runs: Sequence[DemoRun]) -> str:
    prompt_templates = sorted(
        {
            prompt_template
            for run in runs
            for prompt_template in run.prompt_template_versions
        }
    )
    label = (
        "1 prompt template"
        if len(prompt_templates) == 1
        else f"{len(prompt_templates)} prompt templates"
    )
    template_list = ", ".join(prompt_templates) if prompt_templates else "none"
    return f"""
      <div class="dataset-badge">
        <b>{label} in this dataset</b>
        <span>{_h(template_list)}</span>
      </div>
    """


def _render_prompt_code_provenance(runs: Sequence[DemoRun]) -> str:
    rows = [
        """
        <div class="audit-row audit-heading">
          <span>Run</span>
          <span>Judge</span>
          <span>Prompt hash</span>
          <span>Reward code</span>
        </div>
        """
    ]
    for run in runs:
        rows.append(
            f"""
            <div class="audit-row">
              <span>{_h(run.title)}</span>
              <b>{run.verifier_calls_added}</b>
              <span>{_render_hashes(run.prompt_hashes)}</span>
              <span>{_render_hashes(run.reward_logic_versions, empty="none")}</span>
            </div>
            """
        )
    return "".join(rows)


def _training_row_sample_json() -> str:
    training_path = OUTPUT_DIR / "training.jsonl"
    if not training_path.exists():
        return "{}"
    with training_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            sample = {
                "messages": row.get("messages"),
                "reward": row.get("reward"),
                "label": row.get("label"),
                "reward_version": row.get("reward_version"),
                "reward_logic_version": row.get("reward_logic_version"),
                "metadata": {
                    "model_version": metadata.get("model_version"),
                    "train_step": metadata.get("train_step"),
                    "prompt_template_version": metadata.get("prompt_template_version"),
                },
            }
            return json.dumps(sample, indent=2, sort_keys=True)
    return "{}"


def _render_artifact_map() -> str:
    return """
      <svg class="artifact-map" viewBox="0 0 960 340" role="img" aria-label="Reward replay artifact map">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z"></path>
          </marker>
        </defs>
        <g class="node source"><rect x="24" y="128" width="154" height="84" rx="12"></rect><text x="101" y="162">Trajectory</text><text x="101" y="186">JSON</text></g>
        <g class="node memo"><rect x="244" y="54" width="164" height="72" rx="12"></rect><text x="326" y="85">Memoized</text><text x="326" y="107">parse/features</text></g>
        <g class="node memo"><rect x="244" y="192" width="164" height="72" rx="12"></rect><text x="326" y="223">Memoized</text><text x="326" y="245">LLM judge</text></g>
        <g class="node reward"><rect x="474" y="128" width="160" height="84" rx="12"></rect><text x="554" y="162">Reward</text><text x="554" y="186">decision</text></g>
        <g class="node target"><rect x="720" y="18" width="194" height="50" rx="12"></rect><text x="817" y="48">SQLite reward_rows</text></g>
        <g class="node target"><rect x="720" y="80" width="194" height="50" rx="12"></rect><text x="817" y="110">SQLite reward_diffs</text></g>
        <g class="node target"><rect x="720" y="142" width="194" height="50" rx="12"></rect><text x="817" y="172">Training JSONL</text></g>
        <g class="node target"><rect x="720" y="204" width="194" height="50" rx="12"></rect><text x="817" y="234">Owned row files</text></g>
        <g class="node target"><rect x="720" y="266" width="194" height="50" rx="12"></rect><text x="817" y="296">Judge audit JSONL</text></g>
        <path class="edge" d="M178 170 C216 170 210 90 244 90"></path>
        <path class="edge" d="M178 170 C216 170 210 228 244 228"></path>
        <path class="edge" d="M408 90 C450 90 432 170 474 170"></path>
        <path class="edge" d="M408 228 C454 228 432 170 474 170"></path>
        <path class="edge" d="M634 170 C684 170 674 43 720 43"></path>
        <path class="edge" d="M634 170 C684 170 674 105 720 105"></path>
        <path class="edge" d="M634 170 C680 170 674 167 720 167"></path>
        <path class="edge" d="M634 170 C684 170 674 229 720 229"></path>
        <path class="edge" d="M408 228 C566 228 570 291 720 291"></path>
      </svg>
    """


def _find_run(runs: Sequence[DemoRun], run_id: str) -> DemoRun | None:
    return next((run for run in runs if run.run_id == run_id), None)


def _render_dashboard(runs: Sequence[DemoRun], training_sample_json: str) -> str:
    total_calls = runs[-1].total_verifier_calls if runs else 0
    max_duration = max((run.duration_ms for run in runs), default=1)
    naive_verifier_calls = sum(run.reward_rows_total for run in runs)
    saved_verifier_calls = max(0, naive_verifier_calls - total_calls)
    estimated_saved_cost = saved_verifier_calls * JUDGE_COST_PER_CALL_USD
    projected_saved_calls = round(
        saved_verifier_calls
        / max(runs[-1].reward_rows_total if runs else 1, 1)
        * REFERENCE_SCALE_TRAJECTORIES
    )
    projected_saved_cost = projected_saved_calls * JUDGE_COST_PER_CALL_USD
    speedup = 0.0
    if len(runs) >= 2 and runs[1].duration_ms > 0:
        speedup = runs[0].duration_ms / runs[1].duration_ms
    noop_run = _find_run(runs, "noop")
    code_change_run = _find_run(runs, "code-change")
    noop_stage_calls = sum(noop_run.stage_calls.values()) if noop_run is not None else 0
    code_change_judge_calls = (
        code_change_run.stage_calls.get("verifier", 0)
        if code_change_run is not None
        else 0
    )
    code_change_reward_calls = (
        code_change_run.stage_calls.get("reward", 0)
        if code_change_run is not None
        else 0
    )
    actual_stage_calls = {
        stage: sum(run.stage_calls.get(stage, 0) for run in runs)
        for stage in ("parse", "features", "verifier", "reward")
    }
    run_sections = "".join(
        _render_run_section(run, i, max_duration) for i, run in enumerate(runs)
    )
    heatmap = _render_delta_heatmap(runs)
    generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reward Replay Delta Engine</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1d1c18;
      --muted: #6e6a60;
      --line: #ded8cc;
      --paper: #fffdf8;
      --wash: #f6f1e8;
      --coral: #e45f45;
      --teal: #168c86;
      --green: #4d8f57;
      --amber: #c28a21;
      --violet: #7258a8;
      --shadow: 0 18px 50px rgba(48, 42, 32, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(228, 95, 69, 0.16), transparent 34rem),
        linear-gradient(140deg, #fffdf8 0%, #f6f1e8 48%, #eef6f4 100%);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{ width: min(1180px, calc(100vw - 36px)); margin: 0 auto; padding: 34px 0 56px; }}
    header {{ display: grid; grid-template-columns: 1.3fr 0.7fr; gap: 24px; align-items: stretch; margin-bottom: 22px; }}
    .hero, .metric-card, .section, .run-card {{
      background: rgba(255, 253, 248, 0.82);
      border: 1px solid rgba(90, 76, 54, 0.16);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }}
    .hero {{ padding: 30px; border-radius: 18px; min-height: 260px; display: flex; flex-direction: column; justify-content: space-between; }}
    .eyebrow {{ margin: 0 0 10px; color: var(--teal); font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0; }}
    h1 {{ margin: 0; max-width: 760px; font-size: clamp(38px, 6vw, 76px); line-height: 0.96; letter-spacing: 0; }}
    h1 span {{ color: var(--coral); }}
    h2 {{ margin: 0; font-size: 24px; letter-spacing: 0; }}
    .lede {{ max-width: 720px; margin: 18px 0 0; color: #4b473f; font-size: 18px; }}
    .claim {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 26px; }}
    .claim span {{ border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; background: white; font-size: 13px; font-weight: 700; }}
    .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .metric-card {{ min-height: 122px; border-radius: 16px; padding: 18px; display: flex; flex-direction: column; justify-content: space-between; }}
    .metric-card b {{ font-size: 34px; line-height: 1; }}
    .metric-card span {{ color: var(--muted); font-size: 13px; font-weight: 700; }}
    .metric-card small {{ color: var(--muted); font-size: 11px; line-height: 1.3; }}
    .section {{ border-radius: 18px; padding: 22px; margin-top: 18px; }}
    .section-head {{ display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 18px; }}
    .section-head p {{ max-width: 660px; margin: 6px 0 0; color: var(--muted); }}
    .run-stack-list {{ display: grid; gap: 18px; }}
    .run-stack {{ display: grid; gap: 10px; }}
    .run-card {{
      width: 100%;
      border-radius: 14px;
      padding: 16px;
      display: grid;
      grid-template-columns: 54px 1fr auto;
      gap: 16px;
      align-items: center;
      color: inherit;
      text-align: left;
    }}
    .run-index {{ width: 42px; height: 42px; display: grid; place-items: center; border-radius: 12px; background: #1d1c18; color: white; font-weight: 900; }}
    .run-copy strong {{ display: block; font-size: 16px; }}
    .run-copy small {{ display: block; color: var(--muted); margin-top: 3px; font-size: 13px; }}
    .duration-mini {{ display: block; width: min(360px, 100%); height: 8px; margin-top: 10px; border-radius: 999px; background: #e8e0d2; overflow: hidden; }}
    .duration-mini span {{ display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--teal), var(--coral)); }}
    .run-kpis {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: end; }}
    .run-kpis span {{ background: var(--wash); border: 1px solid var(--line); border-radius: 999px; padding: 7px 10px; font-size: 12px; color: var(--muted); }}
    .run-kpis b {{ color: var(--ink); }}
    .run-panel {{ display: block; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; background: white; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ padding: 8px 11px; border-bottom: 1px solid #eee8dc; text-align: left; font-size: 12px; white-space: nowrap; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0; background: #fbf6ed; }}
    tr:last-child td {{ border-bottom: 0; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .numeric {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 5px 9px; font-weight: 800; font-size: 12px; border: 1px solid transparent; }}
    .pill.quiet {{ color: #59564f; background: #eeebe4; border-color: #ded8cc; }}
    .pill.created {{ color: #23632e; background: #e5f3e5; border-color: #bddfbe; }}
    .pill.updated {{ color: #805812; background: #fff0c7; border-color: #efd38b; }}
    .pill.removed {{ color: #9c3422; background: #ffe0d8; border-color: #f2b7a8; }}
    .pill.neutral {{ color: #4f3f87; background: #ece7ff; border-color: #d3c8fb; }}
    .label-badge {{ font-weight: 900; }}
    .label-badge.pass {{ color: var(--green); }}
    .label-badge.fail {{ color: var(--coral); }}
    .heatmap-wrap {{ border: 1px solid var(--line); border-radius: 14px; background: white; padding: 14px; margin-bottom: 18px; overflow-x: auto; }}
    .heatmap-grid {{ display: grid; grid-template-columns: minmax(130px, 1fr) repeat(var(--cols), minmax(58px, 0.7fr)); gap: 6px; align-items: center; min-width: 760px; }}
    .heatmap-corner, .heatmap-label, .heatmap-run {{ color: var(--muted); font-size: 11px; font-weight: 900; }}
    .heatmap-run {{ color: var(--ink); }}
    .heatmap-label {{ text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .heatmap-cell {{ min-height: 22px; border-radius: 7px; border: 1px solid transparent; }}
    .heatmap-cell.none {{ background: #eeebe4; border-color: #ded8cc; }}
    .heatmap-cell.partial {{ background: #fff0c7; border-color: #efd38b; }}
    .heatmap-cell.full {{ background: #dff2e2; border-color: #bddfbe; }}
    .heatmap-legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; color: var(--muted); font-size: 12px; font-weight: 800; }}
    .heatmap-legend span {{ display: inline-flex; gap: 6px; align-items: center; }}
    .heatmap-legend i {{ width: 14px; height: 14px; border-radius: 5px; border: 1px solid var(--line); }}
    .heatmap-legend i.none {{ background: #eeebe4; }}
    .heatmap-legend i.partial {{ background: #fff0c7; border-color: #efd38b; }}
    .heatmap-legend i.full {{ background: #dff2e2; border-color: #bddfbe; }}
    .sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }}
    .insight-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .bars {{ display: grid; gap: 12px; margin-top: 14px; }}
    .bar-row {{ display: grid; grid-template-columns: 150px 1fr 30px; gap: 12px; align-items: center; font-size: 13px; }}
    .bar-track {{ height: 12px; border-radius: 999px; background: #e8e0d2; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--teal), var(--green)); }}
    .audit-list {{ display: grid; gap: 10px; margin-top: 14px; }}
    .audit-row {{ display: grid; grid-template-columns: 1fr 34px 1.2fr 1.2fr; gap: 10px; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee8dc; font-size: 13px; }}
    .audit-row:last-child {{ border-bottom: 0; }}
    .audit-heading {{ color: var(--muted); font-size: 11px; font-weight: 900; text-transform: uppercase; }}
    .audit-row b {{ color: var(--coral); font-variant-numeric: tabular-nums; }}
    .hash-list {{ display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }}
    .hash {{ display: inline-flex; padding: 4px 7px; border-radius: 999px; background: #ece7ff; color: #4f3f87; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .muted {{ color: var(--muted); }}
    .dataset-badge {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; border: 1px solid var(--line); border-radius: 999px; background: white; padding: 8px 12px; margin-top: 12px; font-size: 12px; }}
    .dataset-badge span {{ color: var(--muted); }}
    .sample-card {{ border: 1px solid var(--line); border-radius: 14px; background: #1d1c18; color: #fffdf8; padding: 16px; overflow: auto; }}
    .sample-card pre {{ margin: 0; min-width: 520px; font-size: 12px; line-height: 1.45; white-space: pre; }}
    .contrast-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .contrast-panel {{ border: 1px solid var(--line); border-radius: 14px; background: white; padding: 16px; }}
    .contrast-panel strong {{ display: block; margin-bottom: 10px; }}
    .contrast-panel p {{ margin: 0 0 10px; color: var(--muted); font-size: 14px; }}
    .contrast-panel p:last-child {{ margin-bottom: 0; }}
    .impact-badge {{ display: inline-flex; align-items: center; border-radius: 999px; background: #e5f3e5; border: 1px solid #bddfbe; color: #23632e !important; padding: 8px 12px; font-weight: 900; }}
    .artifact-map {{ width: 100%; height: auto; margin-top: 8px; }}
    .artifact-map .node rect {{ fill: white; stroke-width: 2; }}
    .artifact-map .source rect {{ stroke: var(--coral); }}
    .artifact-map .memo rect {{ stroke: var(--teal); }}
    .artifact-map .reward rect {{ stroke: var(--amber); }}
    .artifact-map .target rect {{ stroke: var(--violet); }}
    .artifact-map text {{ text-anchor: middle; font-size: 16px; font-weight: 850; fill: var(--ink); }}
    .artifact-map .edge {{ fill: none; stroke: #81786b; stroke-width: 2.5; marker-end: url(#arrow); }}
    .artifact-map marker path {{ fill: #81786b; }}
    footer {{ margin-top: 22px; color: var(--muted); font-size: 13px; text-align: center; }}
    @media (max-width: 860px) {{
      main {{ width: min(100vw - 24px, 1180px); padding-top: 18px; }}
      header, .insight-grid {{ grid-template-columns: 1fr; }}
      .metric-grid {{ grid-template-columns: 1fr 1fr; }}
      .contrast-grid {{ grid-template-columns: 1fr; }}
      .run-card {{ grid-template-columns: 44px 1fr; }}
      .run-kpis {{ grid-column: 1 / -1; justify-content: start; }}
      .section-head {{ align-items: start; flex-direction: column; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 6px; }}
      .audit-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <section class="hero">
        <div>
          <p class="eyebrow">CocoIndex RL Control Plane</p>
          <h1>Reward replay becomes a <span>delta update</span>.</h1>
          <p class="lede">This dashboard was generated from a real demo run with a simulated LLM judge. Stage calls are measured from memoized function executions, not inferred from labels.</p>
        </div>
        <div class="claim">
          <span>Code-aware invalidation</span>
          <span>Self-cleaning targets</span>
          <span>Training JSONL stays fresh</span>
        </div>
      </section>
      <section class="metric-grid" aria-label="Demo totals">
        <div class="metric-card"><span>No-op vs backfill</span><b>{speedup:.1f}x</b><small>Includes CocoIndex CLI startup; no-op stage calls = {noop_stage_calls}.</small></div>
        <div class="metric-card"><span>Judge calls saved vs naive replay</span><b>{saved_verifier_calls}</b><small>Naive = every active trajectory reruns on every update.</small></div>
        <div class="metric-card"><span>Illustrative judge cost avoided</span><b>${estimated_saved_cost:.2f}</b><small>${JUDGE_COST_PER_CALL_USD:.2f}/judge call; 10K-row replay mix ≈ ${projected_saved_cost:,.0f} avoided.</small></div>
        <div class="metric-card"><span>Measured reward calls</span><b>{actual_stage_calls["reward"]}</b><small>Code-change run: {code_change_judge_calls} judge / {code_change_reward_calls} reward calls.</small></div>
      </section>
    </header>

    <section class="section">
      <div class="section-head">
        <div>
          <p class="eyebrow">Replay Progression</p>
          <h2>Six updates, each matrix in reading order</h2>
          <p>The heatmap summarizes which trajectories did work in each run. The stacked matrices below show the exact parser, feature, judge, reward, and target deltas without a click.</p>
        </div>
      </div>
      {heatmap}
      <div class="run-stack-list">{run_sections}</div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <p class="eyebrow">Control Plane Shape</p>
          <h2>One declaration keeps every artifact aligned</h2>
          <p>Reward rows, diff rows, training exports, owned row files, and judge audit logs stay in step with the same trajectory-level declaration.</p>
        </div>
      </div>
      {_render_artifact_map()}
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <p class="eyebrow">Without / With</p>
          <h2>The delta engine replaces broad replay scripts</h2>
        </div>
      </div>
      <div class="contrast-grid">
        <div class="contrast-panel">
          <strong>Without maintained reward views</strong>
          <p>Reward or judge edits trigger coarse replay jobs: {naive_verifier_calls} judge calls in this six-update sequence.</p>
          <p>Training JSONL can drift from catalog and diff state.</p>
          <p>Quarantined trajectories can leave stale exported files.</p>
        </div>
        <div class="contrast-panel">
          <strong>With CocoIndex</strong>
          <p class="impact-badge">10K-row projection: ≈ {projected_saved_calls:,} judge calls / ${projected_saved_cost:,.0f} avoided.</p>
          <p>Actual run: {total_calls} judge calls; no-op: {noop_stage_calls} stage calls.</p>
          <p>Python reward-code change: {code_change_judge_calls} judge calls, {code_change_reward_calls} reward calls.</p>
          <p>SQLite, JSONL, row files, and reports are declared together.</p>
          <p>Owned target state removes stale training output automatically.</p>
        </div>
      </div>
    </section>

    <section class="section insight-grid">
      <div>
        <p class="eyebrow">Training Set</p>
        <h2>Export size across runs</h2>
        <div class="bars">{_render_training_bars(runs)}</div>
      </div>
      <div>
        <p class="eyebrow">Prompt & Code Provenance</p>
        <h2>What changed across runs</h2>
        {_prompt_template_badge(runs)}
        <div class="audit-list">{_render_prompt_code_provenance(runs)}</div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <p class="eyebrow">Training Row Sample</p>
          <h2>Reward metadata travels with the trainer-ready JSONL</h2>
          <p>The export keeps the OpenAI-style messages field next to reward, label, reward-code version, model version, prompt template, and train-step provenance.</p>
        </div>
      </div>
      <div class="sample-card"><pre>{_h(training_sample_json)}</pre></div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <p class="eyebrow">What To Notice</p>
          <h2>The hard parts are measured</h2>
          <p>The no-op run records zero stage calls. Adding one trajectory records one parser/features/judge/reward path. Changing the judge prompt records judge+reward calls but no parser work. Changing Python reward code records reward calls only. Quarantine cleans owned training output.</p>
        </div>
      </div>
    </section>

    <footer>Generated {generated_at}. Source data and generated artifacts live under <span class="mono">examples/reward_replay_delta_engine</span>.</footer>
  </main>
</body>
</html>
"""


def _write_demo_artifacts(runs: Sequence[DemoRun]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_RUNS_PATH.write_text(
        json.dumps([_demo_run_to_dict(run) for run in runs], indent=2) + "\n",
        encoding="utf-8",
    )
    DASHBOARD_PATH.write_text(
        _render_dashboard(runs, _training_row_sample_json()),
        encoding="utf-8",
    )


def _run_demo() -> None:
    _reset_demo(announce=False)
    runs: list[DemoRun] = []
    scenarios = [
        (
            "backfill",
            "Full backfill",
            "Build every maintained reward artifact from seed trajectories.",
            None,
        ),
        (
            "noop",
            "No-op rerun",
            "Run again with unchanged sources and policy to show cache reuse.",
            None,
        ),
        (
            "add-trajectory",
            "Add trajectory",
            "Add one support trajectory and replay only that new item.",
            lambda: _add_new_trajectory(announce=False),
        ),
        (
            "strict-policy",
            "Switch reward policy",
            "Tighten reward thresholds and judge prompt, invalidating reward layers.",
            lambda: _set_policy("strict", announce=False),
        ),
        (
            "code-change",
            "Change reward code",
            "Edit Python scoring logic while parser and judge results stay cached.",
            lambda: _set_reward_logic("tool_trace_bonus", announce=False),
        ),
        (
            "quarantine",
            "Quarantine row",
            "Quarantine one previously exported trajectory and clean stale output.",
            lambda: _set_quarantine("traj_weather_002", True, announce=False),
        ),
    ]
    try:
        for run_id, title, description, prepare in scenarios:
            if prepare is not None:
                prepare()
            print(f"Running: {title}")
            runs.append(_run_update_step(run_id, title, description))
    finally:
        _restore_seed_inputs()
    _reset_demo(announce=False)
    app.update_blocking()
    _write_demo_artifacts(runs)
    print(f"Wrote {DASHBOARD_PATH.relative_to(BASE_DIR)}")
    print(f"Wrote {DEMO_RUNS_PATH.relative_to(BASE_DIR)}")
    print("Restored seed trajectories, balanced policy, and baseline outputs.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utilities for the Reward Replay Delta Engine example."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("show", help="Print the latest reward report and table rows.")
    subparsers.add_parser(
        "reset-demo", help="Restore seed trajectories and clear generated output."
    )
    subparsers.add_parser(
        "add-new-trajectory", help="Add one new trajectory to demonstrate a delta run."
    )
    subparsers.add_parser(
        "run-demo",
        help="Run the full scenario sequence and generate output/dashboard.html.",
    )
    set_policy = subparsers.add_parser(
        "set-policy", help="Switch reward_policy.json between demo policies."
    )
    set_policy.add_argument("name", choices=["balanced", "strict"])
    set_reward_logic = subparsers.add_parser(
        "set-reward-logic", help="Switch reward_logic.py between demo code variants."
    )
    set_reward_logic.add_argument("name", choices=["base", "tool_trace_bonus"])
    quarantine = subparsers.add_parser(
        "quarantine", help="Mark a trajectory as quarantined."
    )
    quarantine.add_argument("trajectory_id")
    unquarantine = subparsers.add_parser(
        "unquarantine", help="Clear a trajectory quarantine flag."
    )
    unquarantine.add_argument("trajectory_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command or "show"
    try:
        if command == "show":
            _show_outputs()
        elif command == "reset-demo":
            _reset_demo()
        elif command == "add-new-trajectory":
            _add_new_trajectory()
        elif command == "run-demo":
            _run_demo()
        elif command == "set-policy":
            _set_policy(cast(str, args.name))
        elif command == "set-reward-logic":
            _set_reward_logic(cast(str, args.name))
        elif command == "quarantine":
            _set_quarantine(cast(str, args.trajectory_id), True)
        elif command == "unquarantine":
            _set_quarantine(cast(str, args.trajectory_id), False)
        else:
            parser.error(f"Unknown command {command!r}")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
