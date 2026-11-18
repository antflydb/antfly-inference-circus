#!/usr/bin/env python3
"""Fetch pinned upstream snapshots and verify files against Antfly pull metadata."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def download_manifests(models_dir: Path) -> list[Path]:
    legacy = models_dir.glob("*/*--antfly-*/.antfly-download-complete.json")
    current = models_dir.glob("*/*/.antfly-download-complete.json")
    return sorted(set(legacy) | set(current))


def fetch(models_dir: Path, output: Path, repos: set[str]) -> list[dict]:
    manifests = download_manifests(models_dir)
    if not manifests:
        raise RuntimeError(f"no Antfly download manifests found below {C.public_path(models_dir)}")
    rows = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source"]
        repo = f"{source['owner']}/{source['name']}"
        if repo not in repos:
            continue
        revision = source["resolved_revision"]
        artifacts = [
            artifact
            for artifact in manifest.get("artifacts", [])
            if artifact.get("sha256") and artifact["path"] != "model_manifest.json"
        ]
        weight_artifacts = [artifact for artifact in artifacts if artifact["path"].endswith(".safetensors")]
        if not weight_artifacts:
            continue
        allow_patterns = sorted(
            {
                *(artifact["path"] for artifact in artifacts),
                "*.json",
                "*.model",
                "*.txt",
                "*.py",
                "*.jinja",
                "encoder_config/*",
                "1_Pooling/*",
                "2_Dense/*",
                "3_Dense/*",
            }
        )
        snapshot = Path(
            snapshot_download(
                repo_id=repo,
                revision=revision,
                allow_patterns=allow_patterns,
                cache_dir=str(C.HF_HOME),
            )
        )
        verified = []
        for artifact in artifacts:
            remote_file = snapshot / artifact["path"]
            if not remote_file.is_file():
                if artifact in weight_artifacts:
                    raise RuntimeError(f"{repo}@{revision} is missing required {artifact['path']}")
                continue
            actual = C.sha256_file(remote_file)
            if actual != artifact["sha256"]:
                raise RuntimeError(
                    f"{repo}@{revision} {artifact['path']} sha256 {actual} does not match Antfly metadata {artifact['sha256']}"
                )
            verified.append({"name": artifact["path"], "sha256": actual, "size_bytes": remote_file.stat().st_size})
        if not all(any(item["name"] == weight["path"] for item in verified) for weight in weight_artifacts):
            raise RuntimeError(f"{repo}@{revision} weight verification is incomplete")
        row = {"repo": repo, "revision": revision, "path": str(snapshot.resolve()), "files": verified, "match": True}
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")), flush=True)
    missing = repos - {row["repo"] for row in rows}
    if missing:
        raise RuntimeError(f"missing selected safetensors models: {', '.join(sorted(missing))}")
    previous = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else []
    retained = [row for row in previous if row["repo"] not in repos]
    C.write_json(output, retained + rows)
    return rows


def verify_cached(models_dir: Path, repos: set[str]) -> list[dict]:
    """Verify pinned cached reference weights before any timed workload starts."""
    pulled: dict[str, tuple[dict, Path]] = {}
    for manifest_path in download_manifests(models_dir):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source"]
        pulled[f"{source['owner']}/{source['name']}"] = (manifest, manifest_path.parent)
    reference_rows = {
        row["repo"]: row for row in json.loads(C.REFERENCES.read_text(encoding="utf-8"))
    }
    reference_paths = C.reference_paths()
    verified = []
    for repo in sorted(repos):
        if repo not in pulled:
            raise RuntimeError(f"missing pulled Antfly model {repo}")
        if repo not in reference_rows or repo not in reference_paths:
            raise RuntimeError(f"missing pinned reference snapshot {repo}")
        manifest, model_dir = pulled[repo]
        source = manifest["source"]
        row = reference_rows[repo]
        if str(row.get("revision")) != str(source["resolved_revision"]):
            raise RuntimeError(f"{repo} reference revision does not match the pulled Antfly model")
        weights = [
            artifact
            for artifact in manifest.get("artifacts", [])
            if artifact.get("sha256") and artifact["path"].endswith(".safetensors")
        ]
        if not weights:
            raise RuntimeError(f"{repo} pull metadata has no safetensors weights")
        snapshot = Path(reference_paths[repo])
        files = []
        for artifact in weights:
            reference_file = snapshot / artifact["path"]
            antfly_file = model_dir / artifact["path"]
            if not reference_file.is_file():
                raise RuntimeError(f"{repo} reference snapshot is missing {artifact['path']}")
            if not antfly_file.is_file():
                raise RuntimeError(f"{repo} pulled Antfly model is missing {artifact['path']}")
            reference_hash = C.sha256_file(reference_file)
            antfly_hash = C.sha256_file(antfly_file)
            if reference_hash != artifact["sha256"] or antfly_hash != artifact["sha256"]:
                raise RuntimeError(f"{repo} reference weight mismatch for {artifact['path']}")
            files.append({"name": artifact["path"], "sha256": reference_hash})
        verified.append({"repo": repo, "revision": row["revision"], "files": files})
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=C.MODELS_DIR)
    parser.add_argument("--output", type=Path, default=C.REFERENCES)
    parser.add_argument("--model", action="append", required=True, help="Fetch only these model repositories")
    args = parser.parse_args()
    fetch(args.models_dir.expanduser(), args.output.expanduser(), set(args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
