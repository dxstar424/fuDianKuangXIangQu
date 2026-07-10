#!/usr/bin/env python3
"""
评测平台构建入口 — 满足 /coursegrader/submit/setup.py 检查。

流程：
  1. 使用已提交的 vllm_cscc/ 源码（推荐，离线可编译）
  2. 否则从官方仓库浅克隆 v0.18.1
  3. 应用 scripts/apply_vllm_patches.sh（合入 fdu_vllm）
  4. 委托 vllm_cscc/setup.py 执行 bdist_wheel / install
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VLLM_REPO = os.environ.get(
    "VLLM_REPO",
    "http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git",
)
VLLM_BRANCH = os.environ.get("VLLM_BRANCH", "v0.18.1")
VLLM_SUBDIR = ROOT / "vllm_cscc"


def _is_vllm_tree(path: Path) -> bool:
    return (path / "setup.py").is_file() and (path / "vllm").is_dir()


def _find_vllm_tree() -> Path | None:
    env_dir = os.environ.get("VLLM_DIR", "").strip()
    candidates = [VLLM_SUBDIR]
    if env_dir:
        candidates.append(Path(env_dir))
    for cand in candidates:
        if _is_vllm_tree(cand):
            return cand.resolve()
    return None


def _clone_vllm_tree() -> Path:
    if VLLM_SUBDIR.exists():
        shutil.rmtree(VLLM_SUBDIR)
    subprocess.check_call(
        [
            "git",
            "clone",
            "-b",
            VLLM_BRANCH,
            "--depth",
            "1",
            VLLM_REPO,
            str(VLLM_SUBDIR),
        ]
    )
    if not _is_vllm_tree(VLLM_SUBDIR):
        raise RuntimeError(f"cloned tree is not vLLM source: {VLLM_SUBDIR}")
    return VLLM_SUBDIR.resolve()


def _apply_patches(vllm_dir: Path) -> None:
    patch_sh = ROOT / "scripts" / "apply_vllm_patches.sh"
    if not patch_sh.is_file():
        return
    subprocess.check_call(["bash", str(patch_sh), str(vllm_dir)])


def main() -> None:
    vllm_dir = _find_vllm_tree() or _clone_vllm_tree()
    _apply_patches(vllm_dir)
    os.chdir(vllm_dir)
    subprocess.check_call([sys.executable, "setup.py", *sys.argv[1:]])


if __name__ == "__main__":
    main()
