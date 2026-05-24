from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import hashlib
import json
import pathlib
import shutil
import sqlite3
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import cocoindex as coco
from cocoindex.connectors import localfs, sqlite
from cocoindex.resources.file import FileLike, PatternFilePathMatcher


BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAJECTORY_DIR = DATA_DIR / "trajectories"
SEED_TRAJECTORY_DIR = DATA_DIR / "seed_trajectories"
SCENARIO_TRAJECTORY_DIR = DATA_DIR / "scenario_trajectories"
POLICY_DIR = DATA_DIR / "policies"
POLICY_PATH = BASE_DIR / "reward_policy.json"
OUTPUT_DIR = BASE_DIR / "output"
COCOINDEX_STATE_DIR = BASE_DIR / ".cocoindex"
COCOINDEX_DB_PATH = COCOINDEX_STATE_DIR / "reward_replay"
REWARD_DB_PATH = OUTPUT_DIR / "reward_catalog.sqlite"
VERIFIER_AUDIT_PATH = OUTPUT_DIR / "verifier_calls.jsonl"

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


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    task_id: str
    prompt: str
    actions: list[str]
    observations: list[str]
    final_answer: str
    ground_truth: str
    model: str
    environment: str
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
    baseline_reward_version: str
    baseline_label: str
    label: str
    score: float
    include_in_training: bool
    verifier_label: str
    verifier_confidence: float
    changed_from_baseline: bool
    model: str
    environment: str
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


def _load_trajectory(data: Mapping[str, object]) -> Trajectory:
    metadata = _required_mapping(data, "metadata")
    return Trajectory(
        trajectory_id=_required_str(data, "trajectory_id"),
        task_id=_required_str(data, "task_id"),
        prompt=_required_str(data, "prompt"),
        actions=_required_str_list(data, "actions"),
        observations=_required_str_list(data, "observations"),
        final_answer=_required_str(data, "final_answer"),
        ground_truth=_required_str(data, "ground_truth"),
        model=_required_str(metadata, "model"),
        environment=_required_str(metadata, "environment"),
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


@coco.fn(memo=True)
async def parse_trajectory(file: FileLike[pathlib.Path]) -> Trajectory:
    return _load_trajectory(
        _json_object_from_text(
            await file.read_text(),
            file.file_path.path.as_posix(),
        )
    )


@coco.fn(memo=True)
def extract_features(trajectory: Trajectory) -> RewardFeatures:
    normalized_answer = _normalize_answer(trajectory.final_answer)
    normalized_truth = _normalize_answer(trajectory.ground_truth)
    answer_contains_ground_truth = bool(normalized_truth) and (
        normalized_truth in normalized_answer
    )
    action_count = len(trajectory.actions)
    tool_action_count = _count_tool_actions(trajectory.actions)
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
            f"{tool_action_count} tool actions, final={trajectory.final_answer!r}"
        ),
    )


@coco.fn(memo=True, memo_key={"audit_path": None})
async def run_verifier(
    features: RewardFeatures,
    verifier_prompt: str,
    audit_path: pathlib.Path,
) -> VerifierLabel:
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


@coco.fn(memo=True)
def score_reward(
    trajectory: Trajectory,
    features: RewardFeatures,
    verifier: VerifierLabel,
    policy: RewardPolicy,
) -> RewardDecision:
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
    score = round(min(1.0, max(0.0, score)), 3)

    label = "pass" if score >= policy.pass_threshold else "fail"
    include_in_training = label == "pass" and score >= policy.training_threshold
    training_reason = (
        "eligible for training export"
        if include_in_training
        else f"below training threshold {policy.training_threshold:.2f}"
    )
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
            "messages": [
                {"role": "user", "content": trajectory.prompt},
                {"role": "assistant", "content": trajectory.final_answer},
            ],
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
            baseline_reward_version=policy.baseline_reward_version,
            baseline_label=trajectory.baseline_label,
            label=decision.label,
            score=decision.score,
            include_in_training=decision.include_in_training,
            verifier_label=verifier.label,
            verifier_confidence=verifier.confidence,
            changed_from_baseline=changed_from_baseline,
            model=trajectory.model,
            environment=trajectory.environment,
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


def _reset_demo() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if COCOINDEX_STATE_DIR.exists():
        shutil.rmtree(COCOINDEX_STATE_DIR)
    _copytree_contents(SEED_TRAJECTORY_DIR, TRAJECTORY_DIR)
    shutil.copy2(POLICY_DIR / "balanced.json", POLICY_PATH)
    print("Reset demo trajectories, reward policy, and generated output.")


def _add_new_trajectory() -> None:
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    source = SCENARIO_TRAJECTORY_DIR / "traj_refund_005.json"
    destination = TRAJECTORY_DIR / source.name
    shutil.copy2(source, destination)
    print(f"Added {destination.relative_to(BASE_DIR)}.")


def _set_policy(name: str) -> None:
    policy_path = POLICY_DIR / f"{name}.json"
    if not policy_path.exists():
        raise ValueError(f"Unknown policy {name!r}; expected one of: balanced, strict")
    shutil.copy2(policy_path, POLICY_PATH)
    print(f"Set reward policy to {name}.")


def _set_quarantine(trajectory_id: str, quarantined: bool) -> None:
    path = TRAJECTORY_DIR / f"{trajectory_id}.json"
    if not path.exists():
        raise ValueError(f"No trajectory file found for {trajectory_id!r}")
    data = dict(_read_json(path))
    data["quarantined"] = quarantined
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
    status = "quarantined" if quarantined else "unquarantined"
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
    print(f"Verifier calls recorded: {_read_verifier_call_count()}")
    training_path = OUTPUT_DIR / "training.jsonl"
    if training_path.exists():
        print(f"Training export: {training_path.relative_to(BASE_DIR)}")


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
    set_policy = subparsers.add_parser(
        "set-policy", help="Switch reward_policy.json between demo policies."
    )
    set_policy.add_argument("name", choices=["balanced", "strict"])
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
        elif command == "set-policy":
            _set_policy(cast(str, args.name))
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
