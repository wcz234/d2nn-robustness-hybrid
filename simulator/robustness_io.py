"""Strict JSON and staged artifact I/O for robustness workflows."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile

from robustness_plan import canonical_json_sha256
from training_provenance import file_sha256


def strict_json_dumps(payload, *, pretty: bool = False) -> str:
    options = {"allow_nan": False, "sort_keys": True}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(payload, **options)


def _reject_non_finite_json(value: str):
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def strict_json_loads(text: str):
    return json.loads(text, parse_constant=_reject_non_finite_json)


def read_json(path) -> dict:
    payload = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def write_json(path, payload) -> None:
    Path(path).write_text(f"{strict_json_dumps(payload, pretty=True)}\n", encoding="utf-8")


def write_json_exclusive(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{strict_json_dumps(payload, pretty=True)}\n")


class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self._handle = None
        self.row_count = 0

    def __enter__(self):
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.close()
        self._handle = None

    def write(self, payload) -> None:
        self._handle.write(f"{strict_json_dumps(payload)}\n")
        self.row_count += 1


def artifact_record(path: Path, *, row_count: int | None = None) -> dict:
    payload = {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if row_count is not None:
        payload["row_count"] = row_count
    return payload


def manifest_integrity_payload(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}


def attach_manifest_integrity(manifest: dict) -> dict:
    payload = dict(manifest)
    payload["manifest_payload_sha256"] = canonical_json_sha256(manifest_integrity_payload(payload))
    return payload


def verify_manifest_integrity(manifest: dict) -> None:
    expected = manifest.get("manifest_payload_sha256")
    actual = canonical_json_sha256(manifest_integrity_payload(manifest))
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("manifest_payload_sha256 does not match the manifest contents")


def ensure_output_available(output_dir) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("output path exists and is not a directory")
        if any(output_dir.iterdir()):
            raise ValueError("output directory must be empty or absent")
    return output_dir


@contextmanager
def staged_output_directory(output_dir):
    output_dir = ensure_output_available(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        yield stage
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def finalize_staged_output(stage: Path, output_dir) -> None:
    output_dir = ensure_output_available(output_dir)
    if output_dir.exists():
        output_dir.rmdir()
    stage.replace(output_dir)
