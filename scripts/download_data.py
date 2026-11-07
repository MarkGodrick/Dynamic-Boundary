"""Download the ALO release and restore the paths used by the evaluation code."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

RELEASE_VERSION = "1.1.0-alo"
DATASET_ID = "AItherapist/DBE-Bench"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, relative: str, *, allow_external_symlink: bool = False) -> Path:
    """Resolve a manifest path without allowing traversal or escaping symlinks."""
    part = PurePosixPath(relative)
    if part.is_absolute() or not part.parts or ".." in part.parts or "\\" in relative:
        raise ValueError(f"Unsafe manifest path: {relative!r}")
    target = root.joinpath(*part.parts)
    if not allow_external_symlink and not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes its root: {relative!r}")
    return target


def read_manifest(source: Path) -> dict:
    path = source / "release_manifest.json"
    if not path.is_file():
        raise ValueError("Missing release_manifest.json; the submission-era AVG package is unsupported.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != RELEASE_VERSION:
        raise ValueError(f"Expected {RELEASE_VERSION}, got {manifest.get('version')!r}")
    if manifest.get("filter_spec") != "ALO[0.20, 0.80]":
        raise ValueError("The release manifest does not identify the ALO filter.")
    return manifest


def install_data(
    source: Path,
    project: Path,
    *,
    with_responses: bool = False,
    overwrite: bool = False,
) -> int:
    """Verify all selected files before installing any into the project."""
    manifest = read_manifest(source)
    pending: dict[Path, Path] = {}
    for entry in manifest["files"]:
        if entry.get("kind") == "response_counts" and not with_responses:
            continue
        destinations = entry.get("install_paths", [])
        if not destinations:
            continue
        original = safe_path(source, entry["path"], allow_external_symlink=True)
        if sha256(original) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {entry['path']}")
        for relative in destinations:
            target = safe_path(project, relative)
            if target in pending:
                raise ValueError(f"Duplicate installation destination: {relative}")
            if target.exists():
                if sha256(target) == entry["sha256"]:
                    continue
                if not overwrite:
                    raise FileExistsError(f"Refusing to overwrite {relative}; use --overwrite explicitly.")
            pending[target] = original
    for target, original in pending.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    return len(pending)


def fetch_source(repo_id: str, revision: str | None, with_responses: bool) -> Path:
    """Resolve the manifest first, then pin all downloads to that same commit."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=revision,
        allow_patterns=["release_manifest.json"],
    ))
    manifest = read_manifest(snapshot)
    selected = [entry["path"] for entry in manifest["files"]
                if entry.get("install_paths")
                and (with_responses or entry.get("kind") != "response_counts")]
    for relative in selected:
        safe_path(snapshot, relative, allow_external_symlink=True)
    return Path(snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=snapshot.name,
        allow_patterns=["release_manifest.json", *selected],
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Use a local release without network access.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repo-id", default=DATASET_ID)
    parser.add_argument("--revision", help="HF dataset tag or commit; all downloaded files use one commit.")
    parser.add_argument("--with-responses", action="store_true", help="Install aggregate response counts for offline analyses.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.source_dir or fetch_source(args.repo_id, args.revision, args.with_responses)
    count = install_data(source, args.project_root, with_responses=args.with_responses, overwrite=args.overwrite)
    print(f"Installed {count} files from {RELEASE_VERSION} into {args.project_root.resolve()}")


if __name__ == "__main__":
    main()
