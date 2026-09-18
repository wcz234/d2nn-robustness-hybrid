"""Convert the reviewed Chinese manuscript Markdown subset into Typst."""

from __future__ import annotations

from pathlib import Path
import re

from build_typst import BLOCK_MATH, convert_inline


PAPER_DIR = Path(__file__).resolve().parent
INPUT_PATH = PAPER_DIR / "MANUSCRIPT_ZH.md"
OUTPUT_PATH = PAPER_DIR / "manuscript_zh.typ"

TABLE_CAPTION_RE = re.compile(r"^表\s+\d+\s*\|\s*(.+)$")

FIGURE_LABELS = {
    "figures/figure_1_method_overview.png": "fig-method-zh",
    "figures/figure_2_robustness.png": "fig-robustness-zh",
    "figures/figure_3_complexity.png": "fig-complexity-zh",
    "figures/figure_4_confusion.png": "fig-confusion-zh",
}

PREAMBLE = """#set document(
  title: "衍射深度神经网络鲁棒性与混合光电推理的协议约束数值评估：基于 MNIST 与 Fashion-MNIST 的研究",
)
#set page(
  paper: "a4",
  margin: (top: 22mm, bottom: 22mm, x: 24mm),
  numbering: "1",
  number-align: center,
)
#set text(font: ("Source Han Serif SC", "SimSun"), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.78em, first-line-indent: 2em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 15pt, weight: "bold")
#show heading.where(level: 2): set text(size: 12pt, weight: "bold")
#show figure.caption: set text(size: 9pt)
#show figure.where(kind: table): set figure(supplement: [表])
#set figure(gap: 0.5em)

#align(center)[
  #text(size: 20pt, weight: "bold")[衍射深度神经网络鲁棒性与混合光电推理的协议约束数值评估：基于 MNIST 与 Fashion-MNIST 的研究]
]

#v(1em)
"""


def figure_block(image_line: str, caption_line: str) -> str:
    image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", image_line)
    if image_match is None:
        raise ValueError(f"unsupported image line: {image_line}")
    alt_text, image_path = image_match.groups()
    label = FIGURE_LABELS[image_path]
    vector_path = str(Path(image_path).with_suffix(".svg")).replace("\\", "/")
    caption_match = re.fullmatch(r"\*\*图 \d+ \| ([^*]+)\*\*\s*(.*)", caption_line)
    if caption_match is None:
        raise ValueError(f"unsupported figure caption: {caption_line}")
    title, details = caption_match.groups()
    caption = convert_inline(f"*{title}* {details}")
    escaped_alt = alt_text.replace('"', r'\"')
    return (
        "#figure(\n"
        f'  image("{vector_path}", width: 100%, alt: "{escaped_alt}"),\n'
        "  placement: top,\n"
        f"  caption: [{caption}],\n"
        f") <{label}>"
    )


def warning_block(line: str) -> str:
    text = line.strip("`")
    return f'#block(fill: rgb("#FFF3F1"), inset: 6pt, width: 100%)[#text(fill: rgb("#9C302E"))[{text}]]'


def table_separator_row(line: str) -> bool:
    """True for a Markdown alignment row such as `| --- | :---: |`."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    cells = split_table_row(stripped)
    return bool(cells) and all(
        set(cell) <= {"-", ":", " "} and "-" in cell for cell in cells
    )


def split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def table_block(caption_line: str, header: list[str], rows: list[list[str]]) -> str:
    """Emit a numbered Typst table; the number comes from a counter, not from the source."""
    caption_match = TABLE_CAPTION_RE.match(caption_line.strip())
    if caption_match is None:
        raise ValueError(f"unsupported table caption: {caption_line}")
    caption_text = convert_inline(caption_match.group(1))

    column_count = len(header)
    if column_count == 0:
        raise ValueError("table header has no columns")
    for row in rows:
        if len(row) != column_count:
            raise ValueError(
                f"table row has {len(row)} cells but the header has {column_count}: {row}"
            )

    header_cells = ",\n    ".join(f"[{convert_inline(cell)}]" for cell in header)
    body_cells = []
    for row in rows:
        for cell in row:
            body_cells.append(f"  [{convert_inline(cell)}]")
    body = ",\n".join(body_cells)
    if body:
        body += ",\n"

    alignment = ", ".join(["left"] * column_count)
    return (
        "#figure(\n"
        "  table(\n"
        f"    columns: {column_count},\n"
        f"    align: ({alignment}),\n"
        "    stroke: 0.4pt,\n"
        "    inset: 4pt,\n"
        f"    table.header(\n    {header_cells},\n    ),\n"
        f"{body}"
        "  ),\n"
        f"  caption: [{caption_text}],\n"
        ")"
    )


def convert_body(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0
    in_block_math = False
    math_lines: list[str] = []
    skipped_title = False

    while index < len(lines):
        line = lines[index]
        if line == "$$":
            if in_block_math:
                source = "\n".join(math_lines).strip()
                if source not in BLOCK_MATH:
                    raise ValueError(f"unmapped block math:\n{source}")
                output.append(BLOCK_MATH[source])
                math_lines.clear()
            in_block_math = not in_block_math
            index += 1
            continue
        if in_block_math:
            math_lines.append(line)
            index += 1
            continue
        if not skipped_title and line.startswith("# "):
            skipped_title = True
            index += 1
            continue
        if line == "# 参考文献":
            break
        if line.startswith("!["):
            caption_index = index + 1
            while caption_index < len(lines) and not lines[caption_index].strip():
                caption_index += 1
            output.append(figure_block(line, lines[caption_index]))
            index = caption_index + 1
            continue
        if TABLE_CAPTION_RE.match(line.strip()):
            header_index = index + 1
            while header_index < len(lines) and not lines[header_index].strip():
                header_index += 1
            if header_index < len(lines) and lines[header_index].strip().startswith("|"):
                header = split_table_row(lines[header_index])
                separator_index = header_index + 1
                if separator_index >= len(lines) or not table_separator_row(lines[separator_index]):
                    raise ValueError(f"table is missing a Markdown separator row: {line}")
                rows = []
                row_index = separator_index + 1
                while row_index < len(lines) and lines[row_index].strip().startswith("|"):
                    rows.append(split_table_row(lines[row_index]))
                    row_index += 1
                output.append(table_block(line, header, rows))
                index = row_index
                continue
            # caption without a table body falls through as a plain paragraph
        if line.startswith("`待作者补充："):
            output.append(warning_block(line))
        elif line.startswith("## "):
            output.append("== " + convert_inline(line[3:]))
        elif line.startswith("# "):
            output.append("= " + convert_inline(line[2:]))
        elif re.match(r"^\d+\. ", line):
            output.append("+ " + convert_inline(re.sub(r"^\d+\. ", "", line)))
        else:
            output.append(convert_inline(line))
        index += 1

    if in_block_math:
        raise ValueError("unterminated block math")
    output.append('#bibliography("../references.bib", style: "ieee", title: [参考文献])')
    return "\n".join(output).strip() + "\n"


def main() -> None:
    markdown = INPUT_PATH.read_text(encoding="utf-8")
    OUTPUT_PATH.write_text(PREAMBLE + convert_body(markdown), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
