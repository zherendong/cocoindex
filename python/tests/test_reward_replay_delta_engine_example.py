from __future__ import annotations

import collections
import datetime
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
from typing import cast


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_EXAMPLE_SOURCE = _REPO_ROOT / "examples" / "reward_replay_delta_engine"


def _copy_example(tmp_path: pathlib.Path) -> pathlib.Path:
    repo_dir = tmp_path / "fresh-clone"
    example_dir = repo_dir / "examples" / "reward_replay_delta_engine"
    shutil.copytree(
        _EXAMPLE_SOURCE,
        example_dir,
        ignore=shutil.ignore_patterns(
            ".cocoindex",
            ".venv",
            "__pycache__",
            "output",
            "uv.lock",
        ),
    )

    poison_package = repo_dir / "python" / "cocoindex"
    poison_package.mkdir(parents=True)
    (poison_package / "__init__.py").write_text(
        'raise RuntimeError("run-demo imported the unbuilt source tree")\n',
        encoding="utf-8",
    )
    return example_dir


def _read_json_objects(path: pathlib.Path) -> list[dict[str, object]]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, list)
    objects: list[dict[str, object]] = []
    for item in value:
        assert isinstance(item, dict)
        objects.append(cast(dict[str, object], item))
    return objects


def _read_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value: object = json.loads(line)
        assert isinstance(value, dict)
        objects.append(cast(dict[str, object], value))
    return objects


def _stage_calls(run: dict[str, object]) -> dict[str, int]:
    value = run["stage_calls"]
    assert isinstance(value, dict)
    return {str(key): int(cast(int, count)) for key, count in value.items()}


def test_reward_replay_demo_runs_cold_and_preserves_evidence(
    tmp_path: pathlib.Path,
) -> None:
    example_dir = _copy_example(tmp_path)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "main.py", "run-demo"],
        cwd=example_dir,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    output_dir = example_dir / "output"
    runs = _read_json_objects(output_dir / "demo_runs.json")
    assert [run["run_id"] for run in runs] == [
        "backfill",
        "noop",
        "add-trajectory",
        "strict-policy",
        "code-change",
        "quarantine",
    ]
    assert [_stage_calls(run) for run in runs] == [
        {"parse": 6, "features": 6, "verifier": 6, "reward": 6},
        {"parse": 0, "features": 0, "verifier": 0, "reward": 0},
        {"parse": 1, "features": 1, "verifier": 1, "reward": 1},
        {"parse": 0, "features": 0, "verifier": 7, "reward": 7},
        {"parse": 0, "features": 0, "verifier": 0, "reward": 7},
        {"parse": 1, "features": 1, "verifier": 0, "reward": 1},
    ]
    assert sum(cast(int, run["verifier_calls_added"]) for run in runs) == 14
    assert sum(cast(int, run["reward_rows_total"]) for run in runs) == 40

    verifier_records = _read_jsonl(output_dir / "verifier_calls.jsonl")
    stage_records = _read_jsonl(output_dir / "stage_calls.jsonl")
    assert len(verifier_records) == 14
    assert len(stage_records) == 52
    assert all(record.get("run_id") is not None for record in verifier_records)
    assert all(record.get("run_id") is not None for record in stage_records)

    backfill_verifier_times = [
        datetime.datetime.fromisoformat(cast(str, record["at"]))
        for record in stage_records
        if record.get("run_id") == "backfill" and record.get("stage") == "verifier"
    ]
    assert len(backfill_verifier_times) == 6
    assert (
        max(backfill_verifier_times) - min(backfill_verifier_times)
    ).total_seconds() < 0.75

    with sqlite3.connect(output_dir / "reward_catalog.sqlite") as conn:
        confidence, prompt_hashes = cast(
            tuple[float, int],
            conn.execute(
                """
                SELECT MAX(verifier_confidence), COUNT(DISTINCT judge_prompt_hash)
                FROM reward_rows
                """
            ).fetchone(),
        )
        assert 0.0 <= confidence <= 1.0
        assert prompt_hashes == 1

        diff_columns = {
            cast(str, row[1])
            for row in conn.execute("PRAGMA table_info(reward_diffs)").fetchall()
        }
        assert {"reward_logic_version", "judge_prompt_hash"} <= diff_columns

    training_rows = _read_jsonl(output_dir / "training.jsonl")
    assert len(training_rows) == 5
    assert all(
        row.get("baseline_reward_version") == "v17-prod" for row in training_rows
    )
    assert all(isinstance(row.get("judge_prompt_hash"), str) for row in training_rows)

    stage_counts = collections.Counter(
        cast(str, record["stage"]) for record in stage_records
    )
    assert stage_counts == {
        "parse": 8,
        "features": 8,
        "verifier": 14,
        "reward": 22,
    }

    mismatched_policy = json.loads(
        (example_dir / "reward_policy.json").read_text(encoding="utf-8")
    )
    assert isinstance(mismatched_policy, dict)
    mismatched_policy["baseline_reward_version"] = "v18-balanced"
    (example_dir / "reward_policy.json").write_text(
        json.dumps(mismatched_policy, indent=2) + "\n",
        encoding="utf-8",
    )
    mismatch_result = subprocess.run(
        [sys.executable, "-m", "cocoindex.cli", "update", "-q", "main.py"],
        cwd=example_dir,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    assert mismatch_result.returncode != 0
    assert "baseline reward version" in (
        mismatch_result.stdout + mismatch_result.stderr
    )
