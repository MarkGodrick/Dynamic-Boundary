"""Release installation must preserve version, integrity, and user-owned files."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.download_data import RELEASE_VERSION, install_data, safe_path


def make_package(path: Path, *, version: str = RELEASE_VERSION) -> Path:
    path.mkdir()
    payload = b'{"filter_spec": "ALO[0.20, 0.80]"}\n'
    (path / "anchor.json").write_bytes(payload)
    manifest = {
        "version": version,
        "filter_spec": "ALO[0.20, 0.80]",
        "files": [{"path": "anchor.json", "sha256": hashlib.sha256(payload).hexdigest(),
                   "install_paths": ["benchmarks/example/anchor_alo_bundle_M9.json"]}],
    }
    (path / "release_manifest.json").write_text(json.dumps(manifest))
    return path


def test_rejects_avg_release(tmp_path):
    package = make_package(tmp_path / "source", version="1.0.0")
    with pytest.raises(ValueError, match="Expected"):
        install_data(package, tmp_path / "project")
    assert not (tmp_path / "project").exists()


def test_checks_integrity_before_installation(tmp_path):
    package = make_package(tmp_path / "source")
    (package / "anchor.json").write_text("corrupted")
    with pytest.raises(ValueError, match="Checksum"):
        install_data(package, tmp_path / "project")
    assert not (tmp_path / "project").exists()


def test_installs_idempotently_and_preserves_existing_files(tmp_path):
    package = make_package(tmp_path / "source")
    project = tmp_path / "project"
    target = project / "benchmarks/example/anchor_alo_bundle_M9.json"
    assert install_data(package, project) == 1
    assert install_data(package, project) == 0
    target.write_text("user modification")
    with pytest.raises(FileExistsError):
        install_data(package, project)
    assert target.read_text() == "user modification"


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "x/../../escape", "x\\escape"])
def test_manifest_cannot_escape_project(tmp_path, relative):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(tmp_path, relative)


def test_manifest_cannot_write_through_external_symlink(tmp_path):
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        safe_path(root, "linked/file")


def test_reads_huggingface_style_cached_blob_symlink(tmp_path):
    package = make_package(tmp_path / "source")
    blob = tmp_path / "cached-blob"
    (package / "anchor.json").rename(blob)
    (package / "anchor.json").symlink_to(blob)
    assert install_data(package, tmp_path / "project") == 1
