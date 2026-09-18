"""Convert the reviewed manuscript Markdown subset into a compilable Typst file."""

from __future__ import annotations

from pathlib import Path
import re


PAPER_DIR = Path(__file__).resolve().parent
INPUT_PATH = PAPER_DIR / "MANUSCRIPT_EN.md"
OUTPUT_PATH = PAPER_DIR / "manuscript.typ"

FIGURE_LABELS = {
    "figures/figure_1_method_overview.png": "fig-method",
    "figures/figure_1_robustness.png": "fig-robustness",
    "figures/figure_2_complexity.png": "fig-complexity",
}

UNNUMBERED_HEADINGS = {
    "Abstract",
    "Data Availability",
    "Code Availability",
    "Ethics Statement",
    "Author Contributions",
    "Funding",
    "Competing Interests",
    "Acknowledgements",
}

BLOCK_MATH = {
    "\\widetilde{\\phi}_l = Q(\\phi_l) + \\epsilon_l, \\qquad\nT_l = \\exp(i\\widetilde{\\phi}_l).": (
        "$ tilde(phi)_l = Q(phi_l) + epsilon_l, quad T_l = exp(i tilde(phi)_l). $"
    ),
    (
        "h_d(\\Delta x,\\Delta y) =\n"
        "-i\\frac{\\Delta^2}{\\lambda}\\frac{d}{r^2}\\exp(ikr),\n"
        "\\qquad\n"
        "r=\\sqrt{\\Delta x^2+\\Delta y^2+d^2},\n"
        "\\qquad\n"
        "k=\\frac{2\\pi}{\\lambda}."
    ): (
        "$ h_d(Delta x, Delta y) = -i (Delta^2/lambda) (d/r^2) exp(i k r), \\\n"
        "  r = sqrt((Delta x)^2 + (Delta y)^2 + d^2), quad k = (2 pi)/lambda. $"
    ),
    "s_c=\\sum M_c I.": "$ s_c = sum M_c I. $",
    "z_c=\\log\\left(\\frac{s_c}{\\sum_j s_j+\\varepsilon}+\\varepsilon\\right).": (
        "$ z_c = log(s_c/(sum_j s_j + epsilon) + epsilon). $"
    ),
    (
        "\\mathcal{L}=\\alpha\\mathcal{L}_{\\mathrm{MSE}}\n"
        "+\\beta\\mathcal{L}_{\\mathrm{CE}}\n"
        "+\\gamma R_{\\mathrm{phase}}."
    ): "$ cal(L) = alpha cal(L)_(\"MSE\") + beta cal(L)_(\"CE\") + gamma R_(\"phase\"). $",
}

PREAMBLE = """#set document(
  title: "Protocol-bound robustness evaluation and hybrid optoelectronic inference in diffractive neural networks",
)
#set page(
  paper: "a4",
  margin: (top: 18mm, bottom: 20mm, x: 19mm),
  numbering: "1",
  number-align: center,
)
#set text(font: ("Libertinus Serif", "Times New Roman"), size: 9.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set text(size: 14pt, weight: "bold")
#show heading.where(level: 2): set text(size: 11pt, weight: "bold")
#show figure.caption: set text(size: 8pt)
#set figure(gap: 0.45em)

#align(center)[
  #text(size: 19pt, weight: "bold")[Protocol-bound robustness evaluation and hybrid optoelectronic inference in diffractive neural networks]
  #v(0.8em)
  #text(size: 9pt, fill: rgb("#B64342"))[AUTHOR INPUT NEEDED: authors, affiliations, and corresponding-author details.]
]

#v(1em)
"""


def convert_math(expression: str) -> str:
    value = expression
    replacements = (
        (r"\widetilde{\phi}", "tilde(phi)"),
        (r"\mathcal{L}", "cal(L)"),
        (r"\mathrm{MSE}", '"MSE"'),
        (r"\mathrm{CE}", '"CE"'),
        (r"\mathrm{phase}", '"phase"'),
        (r"\rightarrow", " arrow.r "),
        (r"\times", " times "),
        (r"\geq", " >= "),
        (r"\in", " in "),
        (r"\epsilon", "epsilon"),
        (r"\varepsilon", "epsilon"),
        (r"\lambda", "lambda"),
        (r"\Delta", "Delta"),
        (r"\phi", "phi"),
        (r"\alpha", "alpha"),
        (r"\beta", "beta"),
        (r"\gamma", "gamma"),
        (r"\pi", "pi"),
        (r"\exp", "exp"),
        (r"\log", "log"),
    )
    for source, target in replacements:
        value = value.replace(source, target)
    value = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", value)
    value = re.sub(r"_\{([^{}]+)\}", r"_(\1)", value)
    return value


def convert_inline(text: str) -> str:
    code_spans: list[str] = []

    def protect_code(match: re.Match[str]) -> str:
        code_spans.append(match.group(0))
        return f"CODETOKEN{len(code_spans) - 1}END"

    value = re.sub(r"`[^`]+`", protect_code, text)
    value = re.sub(
        r"\\cite\{([^}]+)\}",
        lambda match: " ".join(f"@{key.strip()}" for key in match.group(1).split(",")),
        value,
    )
    value = re.sub(r"\$([^$]+)\$", lambda match: f"${convert_math(match.group(1))}$", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", value)
    value = value.replace("#", r"\#")
    for index, code in enumerate(code_spans):
        value = value.replace(f"CODETOKEN{index}END", code)
    return value


def figure_block(image_line: str, caption_line: str) -> str:
    image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", image_line)
    if image_match is None:
        raise ValueError(f"unsupported image line: {image_line}")
    alt_text, image_path = image_match.groups()
    label = FIGURE_LABELS[image_path]
    vector_path = str(Path(image_path).with_suffix(".svg")).replace("\\", "/")
    caption_match = re.fullmatch(r"\*\*Figure \d+ \| ([^*]+)\*\*\s*(.*)", caption_line)
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
        if line == "# References":
            break
        if line.startswith("!["):
            caption_index = index + 1
            while caption_index < len(lines) and not lines[caption_index].strip():
                caption_index += 1
            output.append(figure_block(line, lines[caption_index]))
            index = caption_index + 1
            continue
        if line.startswith("`AUTHOR_INPUT_NEEDED: authors, affiliations"):
            index += 1
            continue
        if line.startswith("## "):
            output.append("== " + convert_inline(line[3:]))
        elif line.startswith("# "):
            heading_text = line[2:]
            if heading_text in UNNUMBERED_HEADINGS:
                output.append(f"#heading(numbering: none)[{convert_inline(heading_text)}]")
            else:
                output.append("= " + convert_inline(heading_text))
        elif re.match(r"^\d+\. ", line):
            output.append("+ " + convert_inline(re.sub(r"^\d+\. ", "", line)))
        else:
            output.append(convert_inline(line))
        index += 1

    if in_block_math:
        raise ValueError("unterminated block math")
    output.append('#bibliography("../references.bib", style: "ieee", title: [References])')
    return "\n".join(output).strip() + "\n"


def main() -> None:
    markdown = INPUT_PATH.read_text(encoding="utf-8")
    OUTPUT_PATH.write_text(PREAMBLE + convert_body(markdown), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
