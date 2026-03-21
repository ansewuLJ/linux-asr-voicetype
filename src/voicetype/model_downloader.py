from __future__ import annotations

import os
import shutil
from pathlib import Path


def download_hf_repo_snapshot(
    repo_id: str,
    local_dir: Path,
    *,
    endpoint: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    force_download: bool = False,
    max_workers: int = 8,
    cache_dir: Path | None = None,
    strip_hf_metadata_dir: bool = True,
) -> Path:
    """
    Download a full HF repo snapshot into local_dir with real files only.

    local_dir keeps the same tree layout as the repository root.
    """
    from huggingface_hub import snapshot_download

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        force_download=force_download,
        token=token,
        max_workers=max_workers,
    )
    if strip_hf_metadata_dir:
        metadata_dir = local_dir / ".cache" / "huggingface"
        if metadata_dir.exists():
            shutil.rmtree(metadata_dir)
        cache_root = local_dir / ".cache"
        if cache_root.exists() and not any(cache_root.iterdir()):
            cache_root.rmdir()
    _replace_symlinks_with_real_files(local_dir)
    _assert_no_symlinks(local_dir)
    return local_dir


def _replace_symlinks_with_real_files(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_symlink():
            continue
        target = path.resolve(strict=True)
        if target.is_file():
            path.unlink()
            shutil.copy2(target, path)
            continue
        if target.is_dir():
            tmp_dir = path.with_name(f".{path.name}.tmp_copy")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            shutil.copytree(target, tmp_dir)
            path.unlink()
            tmp_dir.rename(path)
            continue
        raise RuntimeError(f"Unsupported symlink target: {path} -> {target}")


def _assert_no_symlinks(root: Path) -> None:
    links = [p for p in root.rglob("*") if p.is_symlink()]
    if links:
        joined = "\n".join(str(p) for p in links[:10])
        raise RuntimeError(
            "Symlinks are still present after download.\n"
            f"Examples:\n{joined}"
        )
