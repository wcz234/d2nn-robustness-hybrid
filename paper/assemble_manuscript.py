"""Assemble the evidence-matched English manuscript from reviewed section files."""

from __future__ import annotations

from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
SECTION_FILES = (
    "FINAL_INTRODUCTION_EN.md",
    "RELATED_WORK_DRAFT_EN.md",
    "METHODS_AND_EXPERIMENTS_DRAFT_EN.md",
    "RESULTS_DRAFT_EN.md",
    "DISCUSSION_DRAFT_EN.md",
    "CONCLUSION_DRAFT_EN.md",
    "AVAILABILITY_AND_DECLARATIONS_EN.md",
)


def read_utf8(name: str) -> str:
    return (PAPER_DIR / name).read_text(encoding="utf-8").strip()


def extract_frontmatter() -> tuple[str, str]:
    lines = read_utf8("TITLE_ABSTRACT_EN.md").splitlines()
    evidence_heading = lines.index("## Evidence-matched title")
    abstract_heading = lines.index("# Abstract")
    title = next(line.strip("*") for line in lines[evidence_heading + 1 :] if line.startswith("**"))
    abstract = "\n".join(lines[abstract_heading + 1 :]).strip()
    return title, abstract


def assemble() -> str:
    title, abstract = extract_frontmatter()
    parts = [
        f"# {title}",
        "`AUTHOR_INPUT_NEEDED: authors, affiliations, and corresponding-author details.`",
        f"# Abstract\n\n{abstract}",
    ]
    parts.extend(read_utf8(name) for name in SECTION_FILES)
    parts.append("# References\n\nThe machine-readable bibliography is maintained in `../references.bib`.")
    return "\n\n".join(parts) + "\n"


def main() -> None:
    (PAPER_DIR / "MANUSCRIPT_EN.md").write_text(assemble(), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
