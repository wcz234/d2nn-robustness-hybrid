"""Build a machine-readable hash manifest for the Chinese paper evidence package."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "paper" / "EVIDENCE_PACKAGE_MANIFEST.json"

EVIDENCE_FILES = (
    "FORMAL_EXPERIMENT_PROTOCOL.json",
    "results/formal_mnist_v1/clean_summary_all_methods/summary_manifest.json",
    "results/formal_mnist_v1/clean_summary_all_methods/method_summary.jsonl",
    "results/formal_mnist_v1/clean_summary_all_methods/paired_comparisons.jsonl",
    "results/formal_mnist_v1/robustness_summary/summary_manifest.json",
    "results/formal_mnist_v1/robustness_summary/method_summary.jsonl",
    "results/formal_mnist_v1/robustness_summary/paired_comparisons.jsonl",
    "paper/figures/source_data_figure_2a.csv",
    "paper/figures/source_data_figure_2b.csv",
    "paper/figures/source_data_figure_3a.csv",
    "paper/figures/source_data_figure_3b.csv",
    "paper/figures/source_data_figure_4.csv",
    "references.bib",
    "paper/MANUSCRIPT_ZH.md",
    "paper/manuscript_zh.typ",
    "paper/MANUSCRIPT_ZH.pdf",
    "paper/STATISTICS_AUDIT_ZH.md",
    "paper/NATURE_REVIEW_ZH.md",
    "paper/MANUSCRIPT_ZH_AUDIT.md",
    "paper/SUBMISSION_READINESS_ZH.md",
    "results/citation_audit/nature_citation_zh_v2/DIRECT_CITATIONS.md",
    "results/formal_mnist_v4/reference_robustness_summary/method_summary.jsonl",
    "results/formal_mnist_v4/reference_robustness_summary/summary_manifest.json",
    "results/formal_fashion_mnist_v3/robustness_summary/method_summary.jsonl",
    "results/formal_fashion_mnist_v3/robustness_summary/summary_manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int | None:
    if path.suffix.lower() in {".pdf", ".png", ".tiff", ".pth"}:
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    missing = [relative for relative in EVIDENCE_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing evidence files: {missing}")

    files = []
    for relative in EVIDENCE_FILES:
        path = ROOT / relative
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "lines": line_count(path),
                "sha256": sha256(path),
            }
        )

    manifest = {
        "schema_version": "d2nn-chinese-paper-evidence-package/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_boundary": "Public MNIST numerical simulation only; no physical optical or edge-device measurements.",
        "independent_unit": "training_seed",
        "training_seeds": [42, 43, 44],
        "formal_condition_count": 14,
        "formal_deployment_draw_count": 34,
        "files": files,
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
