#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_FLAGS = ("-O3", "-std=c++17", "-shared", "-fPIC")


class BuildError(RuntimeError):
    pass


def build_command(compiler: Path, source: Path, output: Path, arch: str) -> list[str]:
    if arch != "gfx936":
        raise BuildError(f"unsupported architecture: {arch}")
    return [str(compiler), *DEFAULT_FLAGS, "--offload-arch=gfx936", "-o", str(output), str(source)]


def compiler_identity(compiler: Path) -> str:
    try:
        completed = subprocess.run(
            [str(compiler), "--version"], check=True, capture_output=True,
            text=True, timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BuildError(f"cannot query hipcc: {error}") from error
    return (completed.stdout + completed.stderr).strip()


def cache_key(source: bytes, identity: str, arch: str) -> str:
    digest = hashlib.sha256()
    digest.update(source)
    digest.update(identity.encode())
    digest.update(arch.encode())
    digest.update("\0".join(DEFAULT_FLAGS).encode())
    return digest.hexdigest()[:24]


def compile_kernel(
    source: Path,
    cache_root: Path,
    compiler: Path,
    arch: str = "gfx936",
    timeout_s: float = 45.0,
) -> Path:
    source = source.resolve(strict=True)
    compiler = compiler.resolve(strict=True)
    identity = compiler_identity(compiler)
    key = cache_key(source.read_bytes(), identity, arch)
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / f"gfx936_quant_{key}.so"
    if output.is_file() and output.stat().st_size > 0:
        return output
    temporary = cache_root / f".{output.name}.tmp.{os.getpid()}"
    temporary.unlink(missing_ok=True)
    command = build_command(compiler, source, temporary, arch)
    try:
        subprocess.run(
            command,
            check=True,
            timeout=timeout_s,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise BuildError("hipcc completed without producing a shared library")
        os.replace(temporary, output)
    except subprocess.TimeoutExpired as error:
        raise BuildError(f"hipcc exceeded {timeout_s:.1f}s") from error
    except (OSError, subprocess.CalledProcessError) as error:
        raise BuildError(f"hipcc failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return output


def find_hipcc(explicit: str | None) -> Path:
    candidates = [
        explicit, os.getenv("HIPCC"), shutil.which("hipcc"),
        "/opt/rocm/bin/hipcc", "/opt/dtk/bin/hipcc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate)
    raise BuildError("hipcc was not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile the gfx936 quant HIP library")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp/fdu_gfx936_quant"))
    parser.add_argument("--hipcc")
    parser.add_argument("--arch", default="gfx936")
    parser.add_argument("--timeout", type=float, default=45.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = compile_kernel(
            args.source, args.cache_root, find_hipcc(args.hipcc), args.arch, args.timeout
        )
    except BuildError as error:
        print(f"[gfx936-jit] {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
