"""Launch and finalize a pinned model-issue benchmark after adapter review."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from model_handoff import read_handoff, validate_handoff

REPO = "Hanno-Labs/decision-bench"
RESULTS_REPO = "Hanno-Labs/decision-bench-results"
NAMESPACE = "Hanno-Labs"
BUCKET = "hf://buckets/Hanno-Labs/training/decision-bench/runs"
DATASET_REVISION = "b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443"
ALLOWED_ASSOCIATIONS = frozenset({"CONTRIBUTOR", "MEMBER", "OWNER", "COLLABORATOR"})
BRANCH = re.compile(r"codex/model-issue-([1-9][0-9]*)\Z")
REVISION = re.compile(r"[a-f0-9]{40}\Z")
BOOTSTRAP = """
set -eu
apt-get update
apt-get install -y --no-install-recommends git ca-certificates
python -m pip install --disable-pip-version-check -U uv 'huggingface_hub>=1.32' hf_xet
git init /source
git -C /source remote add origin https://github.com/Hanno-Labs/decision-bench.git
git -C /source fetch --depth=1 origin "$SOURCE_SHA"
git -C /source checkout --detach FETCH_HEAD
test "$(git -C /source rev-parse HEAD)" = "$SOURCE_SHA"
cd /source
bash jobs/run_public_hf_job.sh
"""


def command(
    *args: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> str:
    result = subprocess.run(
        args, cwd=cwd, env=env, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def source_env() -> dict[str, str]:
    return {**os.environ, "GH_TOKEN": os.environ["SOURCE_GH_TOKEN"]}


def results_env() -> dict[str, str]:
    return {**os.environ, "GH_TOKEN": os.environ["RESULTS_GH_TOKEN"]}


def issue_is_eligible(number: int) -> bool:
    issue = json.loads(
        command("gh", "api", f"repos/{REPO}/issues/{number}", env=source_env())
    )
    return issue.get("author_association") in ALLOWED_ASSOCIATIONS and any(
        isinstance(label, dict) and label.get("name") == "model"
        for label in issue.get("labels", [])
    )


def merged_pr() -> tuple[int, str]:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    pr = event.get("pull_request")
    if not isinstance(pr, dict) or not pr.get("merged"):
        raise ValueError("Expected a merged adapter PR")
    if pr.get("base", {}).get("ref") != "main":
        raise ValueError("Expected a PR merged into main")
    head = pr.get("head", {})
    if head.get("repo", {}).get("full_name") != REPO:
        raise ValueError("Expected a branch in the DecisionBench repository")
    match = BRANCH.fullmatch(str(head.get("ref", "")))
    if not match:
        raise ValueError("Expected a model-issue branch")
    merge_sha = pr.get("merge_commit_sha")
    if not isinstance(merge_sha, str) or not REVISION.fullmatch(merge_sha):
        raise ValueError("Merged PR lacks a full immutable commit SHA")
    issue_number = int(match.group(1))
    if not issue_is_eligible(issue_number):
        raise ValueError("Source issue no longer passes the contributor and label gate")
    return issue_number, merge_sha


def job_name(issue_number: int, merge_sha: str) -> str:
    return f"db-model-issue-{issue_number}-{merge_sha[:12]}"


def results_uri(issue_number: int, merge_sha: str) -> str:
    return f"{BUCKET}/model-issue-{issue_number}-{merge_sha}"


def launch() -> None:
    issue_number, merge_sha = merged_pr()
    repo = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    if command("git", "-C", str(repo), "rev-parse", "HEAD") != merge_sha:
        raise RuntimeError("Checkout does not match the merged adapter commit")
    handoff: dict[str, Any] = read_handoff(repo, issue_number)
    name = job_name(issue_number, merge_sha)
    existing = command(
        "hf", "jobs", "list", "--namespace", NAMESPACE, "--all", "--name", name,
        "--format", "quiet",
    )
    if existing:
        print(f"Benchmark job already exists for issue #{issue_number}: {existing}")
        return
    env = {
        "SOURCE_SHA": merge_sha,
        "SOURCE_DIR": "/source",
        "MODEL_TYPE": handoff["runner_model_type"],
        "MODEL_REPO": handoff["model_repo"],
        "MODEL_REVISION": handoff["model_revision"],
        "OUTPUT_DIR": "/workflow/out",
        "CHECKPOINT_DIR": "/workflow/out/checkpoints",
        "CHECKPOINT_INTERVAL_SECONDS": "120",
        "BATCH_SIZE": "8",
        "MAX_PROMPT_CHARACTERS_PER_BATCH": "262144",
        "RESULTS_URI": results_uri(issue_number, merge_sha),
        "RESULTS_SYNC_INTERVAL_SECONDS": "120",
        "EXPECTED_ROWS": "23900",
        "HF_XET_HIGH_PERFORMANCE": "1",
    }
    args = [
        "hf", "jobs", "run", "--detach", "--namespace", NAMESPACE,
        "--name", name, "--flavor", "l40sx1", "--timeout", "24h",
        "--secrets", "HF_TOKEN",
    ]
    for key, value in env.items():
        args.extend(("--env", f"{key}={value}"))
    args.extend(("python:3.12-bookworm", "--", "bash", "-lc", BOOTSTRAP))
    job_id = command(*args)
    print(f"Launched issue #{issue_number} benchmark job {job_id} at {name}")


def merged_adapter_prs() -> list[tuple[int, int, str]]:
    value = json.loads(
        command(
            "gh", "pr", "list", "--repo", REPO, "--state", "merged",
            "--limit", "100", "--json", "number,headRefName,mergeCommit,mergedAt",
            env=source_env(),
        )
    )
    candidates: list[tuple[int, int, str]] = []
    for pr in value:
        match = BRANCH.fullmatch(str(pr.get("headRefName", "")))
        merge_commit = pr.get("mergeCommit") or {}
        merge_sha = merge_commit.get("oid")
        if not match or not isinstance(merge_sha, str) or not REVISION.fullmatch(
            merge_sha
        ):
            continue
        issue_number = int(match.group(1))
        if issue_is_eligible(issue_number):
            candidates.append((int(pr["number"]), issue_number, merge_sha))
    return candidates


def merged_handoff(issue_number: int, merge_sha: str) -> dict[str, Any]:
    path = f"jobs/model_requests/{issue_number}.json"
    value = json.loads(
        command(
            "gh", "api", f"repos/{REPO}/contents/{path}?ref={merge_sha}",
            env=source_env(),
        )
    )
    content = base64.b64decode(value["content"])
    return validate_handoff(json.loads(content), issue_number)


def completed_job(issue_number: int, merge_sha: str) -> bool:
    ids = command(
        "hf", "jobs", "list", "--namespace", NAMESPACE, "--all",
        "--name", job_name(issue_number, merge_sha), "--format", "quiet",
    ).splitlines()
    if not ids:
        return False
    if len(ids) != 1:
        raise RuntimeError(f"Multiple HF Jobs exist for issue #{issue_number}")
    inspected = json.loads(
        command(
            "hf", "jobs", "inspect", "--namespace", NAMESPACE,
            ids[0], "--format", "json",
        )
    )
    if len(inspected) != 1:
        raise RuntimeError(f"Could not identify the HF Job for issue #{issue_number}")
    job = inspected[0]
    if job.get("name") != job_name(issue_number, merge_sha):
        raise RuntimeError("HF Job name does not match the reviewed adapter")
    stage = str((job.get("status") or {}).get("stage", "")).upper()
    if stage == "COMPLETED":
        return True
    if stage in {"ERROR", "FAILED", "CANCELED", "CANCELLED", "KILLED"}:
        raise RuntimeError(f"HF Job {ids[0]} for issue #{issue_number} ended {stage}")
    print(f"HF Job {ids[0]} for issue #{issue_number}: {stage or 'pending'}")
    return False


def existing_result_pr(branch: str, *, read_only: bool = False) -> bool:
    prs = json.loads(
        command(
            "gh", "pr", "list", "--repo", RESULTS_REPO, "--state", "all",
            "--head", branch, "--json", "number,url,state",
            env=source_env() if read_only else results_env(),
        )
    )
    if prs:
        print(f"Results PR already exists: {prs[0]['url']}")
    return bool(prs)


def result_branch(issue_number: int, merge_sha: str) -> str:
    return f"codex/model-issue-{issue_number}-{merge_sha[:12]}-result"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(run_dir: Path, handoff: dict[str, Any]) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    if manifest.get("schema_version") != "decision-bench-run-v1":
        raise ValueError("Unexpected run manifest schema")
    for name in ("raw.jsonl", "summary.json"):
        if sha256(run_dir / name) != (manifest.get("files") or {}).get(name):
            raise ValueError(f"Run artifact hash mismatch: {name}")
    for value in (manifest, summary):
        if value.get("requested_rows") != 23900:
            raise ValueError("Run did not request all 23,900 English rows")
        if value.get("successful_rows", 0) + value.get("error_rows", 0) != 23900:
            raise ValueError("Run row accounting is incomplete")
    if manifest["successful_rows"] != summary["successful_rows"] or manifest[
        "error_rows"
    ] != summary["error_rows"]:
        raise ValueError("Manifest and summary row counts differ")
    if summary.get("model") != handoff["model_repo"] or summary.get(
        "model_revision"
    ) != handoff["model_revision"]:
        raise ValueError("Run model identity differs from the reviewed handoff")
    if summary.get("model_type") != handoff["adapter"] or summary.get(
        "probability_source"
    ) != handoff["probability_source"]:
        raise ValueError("Run scoring contract differs from the reviewed handoff")
    if not any(
        dataset.get("path") == "Hanno-Labs/decision-bench"
        and dataset.get("revision") == DATASET_REVISION
        and dataset.get("split") == "eval"
        for dataset in summary.get("datasets", [])
    ):
        raise ValueError("Run dataset differs from the frozen English eval pin")
    row_ids: set[str] = set()
    successful = 0
    unsupported = 0
    with (run_dir / "raw.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or not row_id or row_id in row_ids:
                raise ValueError("Run has a missing or duplicate row ID")
            row_ids.add(row_id)
            if row.get("status") == "ok":
                successful += 1
            elif str(row.get("error_type", "")).startswith("Unsupported"):
                unsupported += 1
            else:
                raise ValueError(f"Execution error on benchmark row {row_id}")
    if len(row_ids) != 23900 or successful != manifest["successful_rows"] or (
        unsupported != manifest["error_rows"]
    ):
        raise ValueError("Run raw rows do not match the manifest")
    print(f"Verified 23900 rows: {successful} successful, {unsupported} unsupported")


def stage_result(
    *, pr_number: int, issue_number: int, merge_sha: str, handoff: dict[str, Any]
) -> None:
    branch = result_branch(issue_number, merge_sha)
    if existing_result_pr(branch):
        return
    with tempfile.TemporaryDirectory(prefix="decisionbench-model-result-") as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        results_repo = root / "results"
        command("hf", "sync", results_uri(issue_number, merge_sha), str(run_dir))
        verify_artifacts(run_dir, handoff)
        command("gh", "auth", "setup-git", env=results_env())
        command(
            "gh", "repo", "clone", RESULTS_REPO, str(results_repo), "--", "--branch", "main",
            env=results_env(),
        )
        command("git", "switch", "-c", branch, cwd=results_repo)
        source_repo = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
        command(
            "uv", "run", "--python", "3.12", "--frozen", "decision-bench",
            "stage-result", str(run_dir), str(results_repo),
            "--model-id", handoff["model_repo"],
            "--model-revision", handoff["model_revision"],
            "--dataset-revision", DATASET_REVISION,
            "--model-type", handoff["result_model_type"],
            "--artifact-uri", results_uri(issue_number, merge_sha),
            "--adapter", handoff["adapter"],
            "--probability-source", handoff["probability_source"],
            cwd=source_repo,
        )
        result_check_env = {**results_env(), "UV_PYTHON": "3.12"}
        command(
            "uv", "sync", "--locked", "--group", "dev",
            cwd=results_repo, env=result_check_env,
        )
        command("make", "check", cwd=results_repo, env=result_check_env)
        command("git", "add", "-A", cwd=results_repo)
        paths = command("git", "diff", "--cached", "--name-only", cwd=results_repo)
        if not paths:
            print(f"Result for issue #{issue_number} already exists on main")
            return
        if any(not path.startswith("results/") for path in paths.splitlines()):
            raise ValueError("Result staging changed a path outside results/")
        command(
            "git", "config", "user.name", "github-actions[bot]", cwd=results_repo
        )
        command(
            "git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com",
            cwd=results_repo,
        )
        command(
            "git", "commit", "-m", f"Add model result from issue #{issue_number}",
            cwd=results_repo,
        )
        command("git", "push", "-u", "origin", branch, cwd=results_repo, env=results_env())
        body = root / "pr-body.md"
        body.write_text(
            f"Full pinned DecisionBench result for {handoff['model_repo']} from "
            f"Hanno-Labs/decision-bench#{pr_number}.\n\n"
            f"Model revision: `{handoff['model_revision']}`.\n"
            f"Adapter merge commit: `{merge_sha}`.\n"
            f"Dataset revision: `{DATASET_REVISION}`.\n"
            "The 23,900 raw rows and artifact hashes were verified before staging.\n"
        )
        url = command(
            "gh", "pr", "create", "--repo", RESULTS_REPO, "--base", "main",
            "--head", branch, "--title",
            f"Add {handoff['model_repo']} DecisionBench result", "--body-file", str(body),
            cwd=results_repo, env=results_env(),
        )
        print(f"Opened verified results PR: {url}")


def finalize() -> None:
    for pr_number, issue_number, merge_sha in merged_adapter_prs():
        branch = result_branch(issue_number, merge_sha)
        if existing_result_pr(branch) or not completed_job(issue_number, merge_sha):
            continue
        handoff = merged_handoff(issue_number, merge_sha)
        stage_result(
            pr_number=pr_number,
            issue_number=issue_number,
            merge_sha=merge_sha,
            handoff=handoff,
        )


def ready() -> None:
    candidates = merged_adapter_prs()
    if candidates and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required to inspect model issue HF Jobs")
    has_results = any(
        not existing_result_pr(result_branch(issue_number, merge_sha), read_only=True)
        and completed_job(issue_number, merge_sha)
        for _pr_number, issue_number, merge_sha in candidates
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
        handle.write(f"has_results={str(has_results).lower()}\n")
    print(f"Completed results awaiting finalization: {has_results}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"launch", "ready", "finalize"}:
        raise SystemExit("Usage: model_results.py launch|ready|finalize")
    if sys.argv[1] == "launch":
        launch()
    elif sys.argv[1] == "ready":
        ready()
    else:
        finalize()
