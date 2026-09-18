"""Machine-readable provenance helpers for simulation training artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch


PROVENANCE_SOURCE_FILES = (
    "artifacts.py",
    "d2nn.py",
    "model_variants.py",
    "perturbations.py",
    "tasks.py",
    "train.py",
    "train_core.py",
    "training_provenance.py",
    "training_runtime.py",
)


def _index_sha256(indices) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def data_split_manifest(*, dataset_key, dataset_name, input_shape, source_train_set, train_set, val_set, test_set, seed):
    return {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "input_shape": list(input_shape),
        "source_train_samples": len(source_train_set),
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "test_samples": len(test_set),
        "split_strategy": "torch.utils.data.random_split",
        "split_seed": int(seed),
        "train_indices_sha256": _index_sha256(train_set.indices),
        "validation_indices_sha256": _index_sha256(val_set.indices),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Expected checkpoint file was not created: {path}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for package_name in ("torch", "torchvision", "numpy", "matplotlib", "Pillow"):
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    result = {}
    for key, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--porcelain"]),
    ):
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            result[key] = {"available": False, "error": type(exc).__name__}
            continue
        if completed.returncode != 0:
            result[key] = {"available": False, "returncode": completed.returncode}
        else:
            result[key] = {"available": True, "value": completed.stdout.strip()}
    return {
        "commit": result["commit"].get("value"),
        "dirty": bool(result["status"].get("value")) if result["status"].get("available") else None,
        "diagnostics": result,
    }


def environment_manifest(device, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    source_hashes = {name: file_sha256(repo_root / name) for name in PROVENANCE_SOURCE_FILES}
    cuda_device = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "packages": _package_versions(),
        "torch_runtime": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device_name": cuda_device,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        },
        "repository": _git_snapshot(repo_root),
        "source_files_sha256": source_hashes,
    }


def cli_configuration(args) -> dict[str, Any]:
    return {key: value for key, value in sorted(vars(args).items())}
