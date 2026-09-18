"""Run deterministic integrity checks for the Chinese manuscript package."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
MANUSCRIPT = PAPER_DIR / "MANUSCRIPT_ZH.md"
BIBLIOGRAPHY = ROOT / "references.bib"
PROTOCOL = ROOT / "FORMAL_EXPERIMENT_PROTOCOL.json"
PDF = PAPER_DIR / "MANUSCRIPT_ZH.pdf"

EXPECTED_METRICS = {
    # clean accuracy and dispersion (recomputed from the frozen summaries)
    "clean baseline accuracy": "89.64 ± 0.33",
    "clean robust accuracy": "88.31 ± 0.16",
    "clean hybrid accuracy": "95.70 ± 0.39",
    "clean electronic accuracy": "94.67 ± 0.25",
    "hybrid clean gain": "6.06",
    "robust clean change": "1.33",
    "electronic clean difference": "1.03",
    # robustness paired differences (robust D2NN minus baseline D2NN)
    "robust lateral 0.25px": "+0.17",
    "robust lateral 0.50px": "+2.20",
    "robust sixteen level": "-0.68",
    "robust eight level": "+3.29",
    "robust four level": "+17.91",
    "robust mixed stress": "+2.76",
    # robustness paired differences (hybrid minus baseline D2NN)
    "hybrid lateral gain": "9.48",
    "hybrid eight-level gain": "9.76",
    "hybrid mixed accuracy": "93.08",
    "hybrid mixed gain": "9.72",
    # interval endpoints that must not be silently narrowed
    "eight-level interval lower": "-3.97",
    "eight-level interval upper": "10.54",
    "four-level interval width": "73.44",
    # descriptors
    "baseline parameters": "12 288",
    "hybrid parameters": "14 698",
    "electronic parameters": "14 320",
    "electronic cpu seconds": "0.696",
    # v2 expansion cohort (protocol FORMAL_EXPERIMENT_PROTOCOL_V2.json, seeds 42-46, n=5)
    "v2 four-level QAT accuracy": "94.10",
    "v2 QAT minus hybrid at four levels": "26.41",
    "v2 QAT minus robust at four levels": "16.54",
    "v2 four-level hybrid accuracy": "67.69",
    "v2 four-level baseline accuracy": "59.51",
    "v2 four-level robust accuracy": "77.56",
    "v2 lenet5 clean accuracy": "98.25",
    "v2 hybrid clean accuracy": "95.68",
    "v2 baseline clean accuracy": "89.56",
    "v2 robust clean accuracy": "88.41",
    "v2 QAT clean accuracy": "61.74",
    "v2 lenet5 parameters": "34 622",
    "v2 hybrid-vs-lenet5 clean gap": "2.57",
    "v2 hybrid four-level seed spread low": "48.54",
    "v2 hybrid four-level seed spread high": "84.28",
    # v3 Fashion-MNIST replication (protocol FORMAL_EXPERIMENT_PROTOCOL_V3.json, n=3)
    "v3 baseline clean": "78.09",
    "v3 robust clean": "76.37",
    "v3 hybrid clean": "81.88",
    "v3 baseline four-level": "45.93",
    "v3 robust four-level": "67.59",
    "v3 hybrid four-level": "51.10",
    "v3 robust clean paired": "-1.72",
    "v3 robust clean interval low": "-3.33",
    "v3 robust clean interval high": "-0.10",
    "v3 four-level paired": "+21.67",
    "v3 four-level interval low": "-5.76",
    "v3 four-level interval high": "+49.10",
    "v3 baseline four-level seed spread": "24.20",
    # v4 ablations, phase-filter control and same-seed references
    "v4 linear-head clean": "96.60",
    "v4 pool4 clean": "96.04",
    "v4 phase-filter clean": "78.01",
    "v4 linear-head four-level": "91.48",
    "v4 pool4 four-level": "82.87",
    "v4 phase-filter four-level": "60.20",
    "v4 linear-head minus hybrid clean": "+0.91",
    "v4 linear-head minus hybrid interval low": "-0.21",
    "v4 linear-head minus hybrid interval high": "2.02",
    "v4 linear-head minus baseline clean": "+6.96",
    "v4 pool4 minus hybrid clean": "+0.35",
    "v4 phase-filter cost": "-11.63",
    "v4 phase-filter four-level interval low": "-9.22",
    "v4 phase-filter four-level interval high": "6.79",
    "v4 table9 hybrid four-level reference": "74.79",
    "v4 table9 baseline four-level reference": "61.41",
}

# Figure order follows the reviewed manuscript: 1 method, 2 confusion, 3 robustness, 4 complexity.
EXPECTED_FIGURE_ORDER = (
    "figures/figure_1_method_overview.png",
    "figures/figure_4_confusion.png",
    "figures/figure_2_robustness.png",
    "figures/figure_3_complexity.png",
)

EXPECTED_TABLE_COUNT = 9

FIGURES = (
    "figures/figure_1_method_overview.svg",
    "figures/figure_2_robustness.svg",
    "figures/figure_3_complexity.svg",
    "figures/figure_4_confusion.svg",
)

INTERNAL_PLACEHOLDERS = ("AUTHOR_INPUT_NEEDED", "TODO", "TBD")

FIGURE_SOURCE_SHA256 = {
    "figures/source_data_figure_2a.csv": "93ff9c165b16f5b7520ec734dd668fdb8c59f7a64436cbe0fe32c0e0f990a479",
    "figures/source_data_figure_2b.csv": "7f5dd354dde9dacbfa7991e01d7bfa11153cedeee9c7b6035519a7339263f25b",
    "figures/source_data_figure_3a.csv": "e5a7cccf496c9cbf8779c53b17352698c6d113ba9416966db4f2416e05168302",
    "figures/source_data_figure_3b.csv": "3aa9dc38403ce82d736cbbbfd791c52f1c717aeefac5bb75240cfea80ab2304f",
    "figures/source_data_figure_4.csv": "fd7de62551e58238cb39671805f1998d77425cf1cb150ca6a0d2f7fea83885a6",
}

REQUIRED_REPRODUCIBILITY_TEXT = (
    "Python 3.11.7",
    "PyTorch 2.10.0",
    "torchvision 0.25.0",
    "retry1",
    "不代表经过多重比较校正的逐条件显著性结论",
)

FORBIDDEN_HARDWARE_CLAIMS = (
    "本文实测光学时延",
    "本文实测边缘设备时延",
    "本文测得功耗",
    "本文测得能效",
    "真实边缘设备实验表明",
    "物理光学实验表明",
)


def citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", text):
        keys.update(key.strip() for key in match.group(1).split(","))
    return keys


def bibliography_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", text))


def manuscript_image_order(text: str) -> list[str]:
    """Image paths in the order they appear in the Markdown source."""
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)


def table_caption_count(text: str) -> int:
    """Number of `表 N |` caption lines that are followed by a pipe table."""
    lines = text.splitlines()
    total = 0
    for index, line in enumerate(lines):
        if not re.match(r"^表\s+\d+\s*\|", line.strip()):
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor < len(lines) and lines[cursor].strip().startswith("|"):
            total += 1
    return total


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font_is_embedded(font_ref: object) -> bool:
    font = font_ref.get_object()
    descriptor = font.get("/FontDescriptor")
    if descriptor:
        descriptor = descriptor.get_object()
        if any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")):
            return True
    return any(font_is_embedded(child) for child in font.get("/DescendantFonts", []))


def pdf_evidence() -> dict[str, object]:
    reader = PdfReader(str(PDF))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    normalized_extracted = re.sub(r"\s+", "", extracted)
    fonts: dict[str, bool] = {}
    for page in reader.pages:
        resources = page.get("/Resources")
        if not resources:
            continue
        font_dict = resources.get_object().get("/Font")
        if not font_dict:
            continue
        for name, reference in font_dict.get_object().items():
            base_font = str(reference.get_object().get("/BaseFont", name))
            fonts[base_font] = fonts.get(base_font, False) or font_is_embedded(reference)
    return {
        "pages": len(reader.pages),
        "text_characters": len(extracted),
        "all_fonts_embedded": bool(fonts) and all(fonts.values()),
        "fonts": fonts,
        "contains_title": "衍射深度神经网络鲁棒性与混合光电推理的协议约束数值评估" in normalized_extracted,
        "contains_references": "参考文献" in extracted,
        "contains_internal_placeholders": [
            placeholder for placeholder in INTERNAL_PLACEHOLDERS if placeholder in extracted
        ],
    }


def main() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cited = citation_keys(manuscript)
    bib_keys = bibliography_keys(bibliography)

    report: dict[str, object] = {
        "citation_key_count": len(cited),
        "bibliography_key_count": len(bib_keys),
        "missing_citation_keys": sorted(cited - bib_keys),
        "unused_bibliography_keys": sorted(bib_keys - cited),
        "missing_metrics": {
            label: value for label, value in EXPECTED_METRICS.items() if value not in manuscript
        },
        "protocol": {
            "training_seeds": protocol["training"]["paired_training_seeds"],
            "condition_count": len(protocol["conditions"]),
            "draw_count": sum(condition["draw_count"] for condition in protocol["conditions"]),
            "independent_unit": protocol["statistics"]["independent_unit"],
        },
        "missing_figures": [path for path in FIGURES if not (PAPER_DIR / path).is_file()],
        "figure_source_hash_mismatches": {
            path: {
                "expected": expected,
                "actual": sha256(PAPER_DIR / path) if (PAPER_DIR / path).is_file() else None,
            }
            for path, expected in FIGURE_SOURCE_SHA256.items()
            if not (PAPER_DIR / path).is_file() or sha256(PAPER_DIR / path) != expected
        },
        "missing_reproducibility_text": [
            phrase for phrase in REQUIRED_REPRODUCIBILITY_TEXT if phrase not in manuscript
        ],
        "author_placeholders": manuscript.count("待作者补充："),
        "internal_placeholders": [
            placeholder for placeholder in INTERNAL_PLACEHOLDERS if placeholder in manuscript
        ],
        "figure_references": {
            "图 1": "图 1" in manuscript,
            "图 2": "图 2" in manuscript,
            "图 3": "图 3" in manuscript,
            "图 4": "图 4" in manuscript,
        },
        "figure_order": manuscript_image_order(manuscript),
        "table_caption_count": table_caption_count(manuscript),
        "forbidden_hardware_claims": [
            phrase for phrase in FORBIDDEN_HARDWARE_CLAIMS if phrase in manuscript
        ],
        "pdf": pdf_evidence(),
    }

    report["figure_order_mismatch"] = report["figure_order"] != list(EXPECTED_FIGURE_ORDER)
    report["table_caption_mismatch"] = report["table_caption_count"] != EXPECTED_TABLE_COUNT

    report["passed"] = not any(
        (
            report["missing_citation_keys"],
            report["unused_bibliography_keys"],
            report["missing_metrics"],
            report["missing_figures"],
            report["figure_source_hash_mismatches"],
            report["missing_reproducibility_text"],
            report["forbidden_hardware_claims"],
            report["internal_placeholders"],
            report["pdf"]["contains_internal_placeholders"],
            report["figure_order_mismatch"],
            report["table_caption_mismatch"],
        )
    ) and report["protocol"] == {
        "training_seeds": [42, 43, 44],
        "condition_count": 14,
        "draw_count": 34,
        "independent_unit": "training_seed",
    } and report["pdf"]["all_fonts_embedded"]

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
