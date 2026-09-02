"""Build the dissertation in the supplied DMU MSc thesis style."""
from __future__ import annotations

import argparse
import re
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "docs" / "When_Explanations_Outlive_Their_Data_DMU_Thesis.docx"
DEFAULT_OUTPUT = ROOT / "docs" / "When_Explanations_Outlive_Their_Data_DMU_Thesis_v9.docx"

CITATIONS = {
    1: "Lin et al., 2018",
    2: "Qin, Zhu and Ye, 2022",
    3: "Lowe et al., 2017",
    4: "Rashid et al., 2020",
    5: "Yu et al., 2022",
    6: "Scarselli et al., 2009",
    7: "Velickovic et al., 2018",
    8: "Iqbal and Sha, 2019",
    9: "Vaswani et al., 2017",
    10: "Heuillet, Couthouis and Diaz-Rodriguez, 2021",
    11: "Madumal et al., 2020",
    12: "Jain and Wallace, 2019",
    13: "Wiegreffe and Pinter, 2019",
    14: "Serrano and Smith, 2019",
    15: "Liu et al., 2022",
    16: "DeYoung et al., 2020",
    17: "Jacovi and Goldberg, 2020",
    18: "Ribeiro, Singh and Guestrin, 2016",
    19: "Alvarez-Melis and Jaakkola, 2018",
    20: "Lipton, 2018",
    21: "Kaul, Yates and Gruteser, 2012",
    22: "Lopez et al., 2018",
    23: "Ying et al., 2019",
    24: "Luo et al., 2020",
    25: "Yuan et al., 2023",
    26: "Abnar and Zuidema, 2020",
    27: "Adebayo et al., 2018",
    28: "Hooker et al., 2019",
    29: "Amara et al., 2022",
    30: "Milani et al., 2024",
    31: "Bekkemoen, 2024",
    32: "Puiutta and Veith, 2020",
    33: "Hu, Feng and Li, 2025",
    34: "Greydanus et al., 2018",
    35: "Mott et al., 2019",
    36: "Yuan et al., 2021",
    37: "Wang et al., 2025",
    38: "Sha et al., 2026",
}

TABLE_TITLES = {
    (3, 1): "Observation graph node types and roles",
    (3, 2): "Policy conditions used in the experiment",
    (3, 3): "Validation-selected checkpoint indices (zero-based)",
    (3, 4): "Shared training and optimization settings",
    (3, 5): "Held-out evaluation structure and episode counts",
    (3, 6): "Telemetry conditions used in held-out evaluation",
    (4, 1): "Policy capability on held-out demand",
    (4, 2): "Construct-validity audit of random controls",
    (4, 3): "Clean-telemetry faithfulness controls by training seed",
    (4, 4): "Valid node-count distribution in scored graphs",
    (4, 5): "Exposure-conditioned paired attention and DEF shifts",
    (4, 6): "Action composition and dispatch-stratified DEF diagnostic",
    (4, 7): "Hypothesis outcomes across training seeds",
    (5, 1): "Freshness-aware explanation assurance approach",
}

FIGURES = [
    ("Figure 3.1", "Local observation graph and request-to-action mapping"),
    ("Figure 3.2", "Graph-attention policy and audit channel"),
    ("Figure 3.3", "Training, selection and held-out evaluation"),
    ("Figure 3.4", "Tunnel-triggered observation-layer telemetry degradation"),
    ("Figure 3.5", "Construct-validity controls for action-linked request nodes"),
    ("Figure 4.1", "GAT training and checkpoint-selection diagnostics"),
    ("Figure 4.2", "Clean-telemetry pickups by training seed"),
    ("Figure 4.3", "Faithfulness perturbation controls by checkpoint"),
    ("Figure 4.4", "Attention aggregation sensitivity"),
    ("Figure 4.5", "Attention query-row sensitivity"),
    ("Figure 4.6", "Top-k overlap and DEF resolution"),
    ("Figure 4.7", "Outage-duration sweep by policy and training seed"),
    ("Figure 4.8", "Paired attention and faithfulness shift by training seed"),
    ("Figure 4.9", "Action-stratified faithfulness diagnostic"),
    ("Figure 4.10", "WAMSN-DEF correlation by training seed"),
    ("Figure 5.1", "Cross-seed evidence matrix"),
]

TABLES = [(f"Table {chapter}.{number}", title)
          for (chapter, number), title in TABLE_TITLES.items()]

FIGURE_PAGES = {
    "Figure 3.1": "9", "Figure 3.2": "10", "Figure 3.3": "11",
    "Figure 3.4": "12", "Figure 3.5": "14",
    "Figure 4.1": "19", "Figure 4.2": "20", "Figure 4.3": "22",
    "Figure 4.4": "23", "Figure 4.5": "24",
    "Figure 4.6": "25", "Figure 4.7": "26", "Figure 4.8": "27",
    "Figure 4.9": "28", "Figure 4.10": "29", "Figure 5.1": "31",
}

TABLE_PAGES = {
    "Table 3.1": "8", "Table 3.2": "10", "Table 3.3": "11",
    "Table 3.4": "11", "Table 3.5": "15", "Table 3.6": "15",
    "Table 4.1": "20", "Table 4.2": "21",
    "Table 4.3": "21", "Table 4.4": "24", "Table 4.5": "26",
    "Table 4.6": "27", "Table 4.7": "29", "Table 5.1": "34",
}

HEADING_PAGES = {
    "Chapter 1: Introduction": "1", "1.1 Context": "1",
    "1.2 Problem statement": "1", "1.3 Research questions": "1",
    "1.4 Objectives": "2", "1.5 Main findings": "2",
    "1.6 Contributions": "3", "1.7 Dissertation structure": "3",
    "Chapter 2: Literature Review": "4",
    "2.1 Reinforcement learning for fleet dispatch": "4",
    "2.2 Explainability in reinforcement learning": "4",
    "2.3 Is attention an explanation?": "5",
    "2.4 Explaining graph neural networks": "6",
    "2.5 Faithfulness evaluation and its pitfalls": "6",
    "2.6 Age of Information and telemetry degradation": "7",
    "2.7 Research gap and positioning": "7",
    "Chapter 3: Methodology": "8", "3.1 Study design": "8",
    "3.2 SUMO environment and data": "8",
    "3.3 Observation graph and action space": "8",
    "3.4 Policy models and training": "9", "3.5 Telemetry degradation": "12",
    "3.6 Explanation measures": "12",
    "3.6.1 Decision-level explanation faithfulness": "12",
    "3.6.2 Stale-node attention": "13", "3.7 Construct-validity audit": "14",
    "3.8 Evaluation matrix": "15", "3.9 Hypotheses and statistics": "16",
    "3.10 Reproducibility": "17", "3.11 Ethics and data governance": "17",
    "Chapter 4: Results": "19",
    "4.1 Policy capability and training stability": "19",
    "4.2 Construct-validity audit": "21",
    "4.3 Evaluator sensitivity and ranking controls": "21",
    "4.4 Small-graph resolution": "24",
    "4.5 Telemetry manipulation and stale exposure": "25",
    "4.6 Exposure-conditioned paired audit": "26",
    "4.7 Action-stratified diagnostic": "27",
    "4.8 Faithfulness hypotheses": "28", "4.9 Result summary": "29",
    "Chapter 5: Discussion": "31", "5.1 Answer to the central problem": "31",
    "5.2 What the stale-exposure result means": "32",
    "5.3 Dependence on checkpoint and analysis choice": "32",
    "5.4 Role of the construct-validity audit": "33",
    "5.5 Degradation-aware training": "33",
    "5.6 Proposed assurance approach": "33",
    "5.7 Limitations": "34", "5.8 Future work": "35",
    "Chapter 6: Conclusion": "36",
}


def paragraph_text(element) -> str:
    # Word may repeat text in proofing or field XML; Paragraph.text reflects
    # the single string shown to the reader.
    return Paragraph(element, None).text.strip()


def find_body_child(doc: Document, text: str):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p") and paragraph_text(child) == text:
            return child
    raise ValueError(f"paragraph not found: {text}")


def clear_between(doc: Document, start_text: str, end_text: str) -> tuple[object, object]:
    body = doc.element.body
    start = find_body_child(doc, start_text)
    end = find_body_child(doc, end_text)
    children = list(body.iterchildren())
    start_index = children.index(start)
    end_index = children.index(end)
    for child in children[start_index + 1:end_index]:
        body.remove(child)
    return start, end


def remove_range_inclusive(doc: Document, start_text: str, end_text: str) -> object:
    body = doc.element.body
    start = find_body_child(doc, start_text)
    end = find_body_child(doc, end_text)
    children = list(body.iterchildren())
    start_index = children.index(start)
    end_index = children.index(end)
    for child in children[start_index:end_index]:
        body.remove(child)
    return end


def move_before(element, anchor) -> None:
    anchor.addprevious(element)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_width(cell, dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_cell_margins(cell, *, top: int = 45, start: int = 90,
                     bottom: int = 45, end: int = 90) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    margins = tc_pr.find(qn("w:tcMar"))
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for side, value in (("top", top), ("start", start),
                        ("bottom", bottom), ("end", end)):
        element = margins.find(qn(f"w:{side}"))
        if element is None:
            element = OxmlElement(f"w:{side}")
            margins.append(element)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    settings = {
        "top": ("single", "8", "7F7F7F"),
        "bottom": ("single", "8", "7F7F7F"),
        "left": ("nil", "0", "FFFFFF"),
        "right": ("nil", "0", "FFFFFF"),
        "insideH": ("single", "4", "D9D9D9"),
        "insideV": ("nil", "0", "FFFFFF"),
    }
    for side, (style, size, colour) in settings.items():
        border = borders.find(qn(f"w:{side}"))
        if border is None:
            border = OxmlElement(f"w:{side}")
            borders.append(border)
        border.set(qn("w:val"), style)
        border.set(qn("w:sz"), size)
        border.set(qn("w:color"), colour)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def set_table_geometry(table, widths: list[int]) -> None:
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)
            set_cell_margins(cell)
    set_table_borders(table)


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    tr_pr.append(marker)


def keep_row(row) -> None:
    prevent_row_split(row)
    for cell in row.cells:
        for paragraph in cell.paragraphs:
            paragraph.paragraph_format.keep_together = True


def add_numbering_definition(doc: Document, *, bullet: bool) -> int:
    numbering = doc.part.numbering_part.element
    abstract_ids = [int(element.get(qn("w:abstractNumId")))
                    for element in numbering.findall(qn("w:abstractNum"))]
    num_ids = [int(element.get(qn("w:numId")))
               for element in numbering.findall(qn("w:num"))]
    abstract_id = max(abstract_ids, default=-1) + 1
    num_id = max(num_ids, default=0) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)

    level = OxmlElement("w:lvl")
    level.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    level.append(start)
    number_format = OxmlElement("w:numFmt")
    number_format.set(qn("w:val"), "bullet" if bullet else "decimal")
    level.append(number_format)
    level_text = OxmlElement("w:lvlText")
    level_text.set(qn("w:val"), "•" if bullet else "%1.")
    level.append(level_text)
    suffix = OxmlElement("w:suff")
    suffix.set(qn("w:val"), "tab")
    level.append(suffix)
    justification = OxmlElement("w:lvlJc")
    justification.set(qn("w:val"), "left")
    level.append(justification)
    paragraph_properties = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "720")
    tabs.append(tab)
    paragraph_properties.append(tabs)
    indent = OxmlElement("w:ind")
    indent.set(qn("w:left"), "720")
    indent.set(qn("w:hanging"), "360")
    paragraph_properties.append(indent)
    level.append(paragraph_properties)
    if bullet:
        run_properties = OxmlElement("w:rPr")
        fonts = OxmlElement("w:rFonts")
        fonts.set(qn("w:ascii"), "Times New Roman")
        fonts.set(qn("w:hAnsi"), "Times New Roman")
        fonts.set(qn("w:eastAsia"), "Times New Roman")
        fonts.set(qn("w:cs"), "Times New Roman")
        run_properties.append(fonts)
        level.append(run_properties)
    abstract.append(level)
    numbering.append(abstract)

    number = OxmlElement("w:num")
    number.set(qn("w:numId"), str(num_id))
    abstract_reference = OxmlElement("w:abstractNumId")
    abstract_reference.set(qn("w:val"), str(abstract_id))
    number.append(abstract_reference)
    numbering.append(number)
    return num_id


def ensure_list_style(doc: Document, name: str, *, bullet: bool):
    styles = doc.styles
    if name in [style.name for style in styles]:
        style = styles[name]
    else:
        style = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    run_properties = style._element.get_or_add_rPr()
    language = run_properties.find(qn("w:lang"))
    if language is None:
        language = OxmlElement("w:lang")
        run_properties.append(language)
    language.set(qn("w:val"), "en-GB")
    language.set(qn("w:eastAsia"), "en-GB")
    language.set(qn("w:bidi"), "en-GB")
    style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    style.paragraph_format.left_indent = Inches(0.5)
    style.paragraph_format.first_line_indent = Inches(-0.25)
    style.paragraph_format.line_spacing = 1.15
    style.paragraph_format.space_before = Pt(0)
    style.paragraph_format.space_after = Pt(3)
    style.paragraph_format.widow_control = True

    p_pr = style._element.get_or_add_pPr()
    existing_numbering = p_pr.find(qn("w:numPr"))
    if existing_numbering is not None:
        p_pr.remove(existing_numbering)
    num_pr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num_pr.append(ilvl)
    num_id = OxmlElement("w:numId")
    num_id.set(qn("w:val"), str(add_numbering_definition(doc, bullet=bullet)))
    num_pr.append(num_id)
    p_pr.append(num_pr)
    return style


def set_paragraph_numbering(paragraph, num_id: int) -> None:
    """Assign a list instance so each independent numbered list restarts at 1."""
    p_pr = paragraph._p.get_or_add_pPr()
    existing = p_pr.find(qn("w:numPr"))
    if existing is not None:
        p_pr.remove(existing)
    num_pr = OxmlElement("w:numPr")
    level = OxmlElement("w:ilvl")
    level.set(qn("w:val"), "0")
    num_pr.append(level)
    number = OxmlElement("w:numId")
    number.set(qn("w:val"), str(num_id))
    num_pr.append(number)
    p_pr.append(num_pr)


def apply_inline(paragraph, text: str) -> None:
    """Add a small Markdown subset while preserving DMU body typography."""
    text = text.replace("`", "")
    pattern = re.compile(r"(\*\*.+?\*\*|\*.+?\*)")
    position = 0
    for match in pattern.finditer(text):
        if match.start() > position:
            paragraph.add_run(text[position:match.start()])
        token = match.group(0)
        run = paragraph.add_run(token.strip("*"))
        run.bold = token.startswith("**")
        run.italic = not token.startswith("**")
        position = match.end()
    if position < len(text):
        paragraph.add_run(text[position:])


def convert_citations(text: str) -> str:
    # Preserve narrative Harvard citations: "Author [n]" becomes
    # "Author (year)", while a standalone [n] remains "(Author, year)".
    ordered_groups = ([6, 7], [12, 14, 13, 15])
    for group in ordered_groups:
        token = ", ".join(f"[{number}]" for number in group)
        citation = "; ".join(CITATIONS[number] for number in group)
        text = text.replace(token, f"({citation})")

    for number, citation in CITATIONS.items():
        authors, year = citation.rsplit(", ", 1)
        text = text.replace(f"{authors} [{number}]", f"{authors} ({year})")

    def replace_range(match: re.Match[str]) -> str:
        start, end = int(match.group(1)), int(match.group(2))
        return "(" + "; ".join(CITATIONS[number] for number in range(start, end + 1)) + ")"

    text = re.sub(r"\[(\d+)\]\s*[-–]\s*\[(\d+)\]", replace_range, text)

    def replace_single(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return f"({CITATIONS[number]})" if number in CITATIONS else match.group(0)

    return re.sub(r"\[(\d+)\]", replace_single, text)


def style_document(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.widow_control = True

    for name, size, alignment in (
        ("Heading 1", 14, WD_ALIGN_PARAGRAPH.CENTER),
        ("Heading 2", 12, WD_ALIGN_PARAGRAPH.LEFT),
        ("Heading 3", 12, WD_ALIGN_PARAGRAPH.LEFT),
    ):
        style = styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.alignment = alignment
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_after = Pt(6)
    styles["Heading 1"].paragraph_format.page_break_before = True
    styles["Heading 1"].paragraph_format.space_after = Pt(12)
    styles["Heading 2"].paragraph_format.space_before = Pt(12)
    styles["Heading 2"].paragraph_format.space_after = Pt(4)
    styles["Heading 3"].paragraph_format.space_before = Pt(9)
    styles["Heading 3"].paragraph_format.space_after = Pt(3)

    ensure_list_style(doc, "DMU Bullet", bullet=True)
    ensure_list_style(doc, "DMU Number", bullet=False)

    if "DMU Caption" not in [style.name for style in styles]:
        styles.add_style("DMU Caption", 1)
    caption = styles["DMU Caption"]
    caption.font.name = "Times New Roman"
    caption.font.size = Pt(10)
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.keep_together = True
    caption.paragraph_format.space_before = Pt(3)
    caption.paragraph_format.space_after = Pt(8)
    caption.paragraph_format.line_spacing = 1.0

    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.top_margin = Inches(0.59)
        section.bottom_margin = Inches(1.03)


def add_paragraph_before(doc: Document, anchor, text: str = "", style: str = "Normal"):
    paragraph = doc.add_paragraph(style=style)
    if text:
        apply_inline(paragraph, text)
    move_before(paragraph._p, anchor)
    return paragraph


def add_caption_before(doc: Document, anchor, text: str):
    paragraph = add_paragraph_before(doc, anchor, style="DMU Caption")
    paragraph.add_run(text).bold = True
    paragraph.paragraph_format.keep_with_next = True
    return paragraph


def add_table_before(doc: Document, anchor, rows: list[list[str]], title: str):
    caption_paragraph = add_caption_before(doc, anchor, title)
    if title.startswith("Table 5.1:"):
        caption_paragraph.paragraph_format.page_break_before = True
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = None
    total = 9020
    if title.startswith("Table 3.2:"):
        widths = [1200, 1650, 2650, 3520]
    elif title.startswith("Table 3.5:"):
        widths = [2350, 1100, 1500, 1250, 1600, 1220]
    elif title.startswith("Table 3.6:"):
        widths = [1150, 2550, 2200, 3120]
    elif title.startswith("Table 4.6:"):
        widths = [1700, 950, 950, 1900, 3520]
    elif title.startswith("Table 5.1:"):
        widths = [1900, 2350, 1900, 2870]
    else:
        weights = []
        for column in range(len(rows[0])):
            longest = max(len(row[column]) for row in rows)
            weights.append(max(8, min(longest, 45)))
        widths = [int(total * weight / sum(weights)) for weight in weights]
        widths[-1] += total - sum(widths)
    set_table_geometry(table, widths)
    for row_index, values in enumerate(rows):
        row = table.rows[row_index]
        keep_row(row)
        if row_index == 0:
            repeat_header(row)
        for column_index, value in enumerate(values):
            cell = row.cells[column_index]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.widow_control = True
            paragraph.alignment = (WD_ALIGN_PARAGRAPH.CENTER
                                   if column_index > 0 and len(value) < 25
                                   else WD_ALIGN_PARAGRAPH.LEFT)
            run = paragraph.add_run(convert_citations(value))
            run.font.name = "Times New Roman"
            run.font.size = Pt(10)
            if row_index == 0:
                run.bold = True
                set_cell_shading(cell, "E7E6E6")
    if len(rows) <= 4:
        for row in table.rows[:-1]:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = True
    move_before(table._tbl, anchor)
    return caption_paragraph


def parse_table(lines: list[str]) -> list[list[str]]:
    parsed = [[cell.strip() for cell in line.strip().strip("|").split("|")]
              for line in lines]
    return [parsed[0], *parsed[2:]]


def add_image_before(doc: Document, anchor, source: Path, alt_text: str):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.keep_together = True
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run()
    run.add_picture(str(source), width=Inches(6.15))
    drawing = run._r.find(qn("w:drawing"))
    if drawing is not None:
        doc_pr = drawing.find(".//" + qn("wp:docPr"))
        if doc_pr is not None:
            doc_pr.set("descr", alt_text)
    move_before(paragraph._p, anchor)
    return paragraph


def collect_blocks(path: Path) -> list[tuple[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[str, object]] = []
    index = 0
    paragraph_lines: list[str] = []

    def flush() -> None:
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines)
            blocks.append(("paragraph", convert_citations(text)))
            paragraph_lines.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush()
            index += 1
            continue
        if stripped.startswith("```"):
            flush()
            code = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            blocks.append(("code", "\n".join(code)))
            index += 1
            continue
        image_match = re.match(r"!\[(.+?)\]\((.+?)\)", stripped)
        if image_match:
            flush()
            blocks.append(("image", (image_match.group(1), (path.parent / image_match.group(2)).resolve())))
            index += 1
            continue
        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append(("heading", (level, stripped[level:].strip())))
            index += 1
            continue
        if stripped.startswith("|"):
            flush()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            blocks.append(("table", parse_table(table_lines)))
            continue
        bullet = re.match(r"^[-*]\s+(.+)", stripped)
        numbered = re.match(r"^\d+\.\s+(.+)", stripped)
        if bullet or numbered:
            flush()
            item_lines = [bullet.group(1) if bullet else numbered.group(1)]
            index += 1
            while index < len(lines):
                continuation = lines[index]
                if not continuation.strip() or not re.match(r"^(?: {2,}|\t)\S", continuation):
                    break
                item_lines.append(continuation.strip())
                index += 1
            list_text = " ".join(item_lines)
            blocks.append(("bullet" if bullet else "number", convert_citations(list_text)))
            continue
        paragraph_lines.append(stripped)
        index += 1
    flush()
    return blocks


def insert_chapters(doc: Document, anchor) -> tuple[list[str], list[str]]:
    headings: list[str] = []
    captions: list[str] = []
    table_count: dict[int, int] = {}
    current_chapter = 0
    for chapter_path in sorted((ROOT / "docs" / "dissertation").glob("0[1-6]_*.md")):
        blocks = collect_blocks(chapter_path)
        numbered_list_id = None
        for block_index, (block_type, payload) in enumerate(blocks):
            if block_type != "number":
                numbered_list_id = None
            if block_type == "heading":
                level, text = payload
                if level == 1:
                    match = re.match(r"(\d+)\.\s+(.+)", text)
                    if not match:
                        continue
                    current_chapter = int(match.group(1))
                    text = f"Chapter {current_chapter}: {match.group(2).title()}"
                    paragraph = add_paragraph_before(doc, anchor, text, "Heading 1")
                else:
                    paragraph = add_paragraph_before(doc, anchor, text, f"Heading {min(level, 3)}")
                headings.append(text)
                add_bookmark(doc, paragraph, bookmark_name("heading", text))
                paragraph.paragraph_format.keep_with_next = True
            elif block_type == "paragraph":
                text = payload
                if re.match(r"\*\*Figure\s+\d+\.\d+\.\*\*", text):
                    clean = text.replace("**", "")
                    paragraph = add_paragraph_before(doc, anchor, style="DMU Caption")
                    apply_inline(paragraph, clean)
                    paragraph.paragraph_format.keep_together = True
                    figure_match = re.match(r"(Figure \d+\.\d+)\.", clean)
                    if figure_match is None:
                        raise ValueError(f"invalid figure caption: {clean}")
                    figure_number = figure_match.group(1)
                    add_bookmark(doc, paragraph, bookmark_name("figure", figure_number))
                    captions.append(figure_number)
                else:
                    paragraph = add_paragraph_before(doc, anchor, text)
                    paragraph.paragraph_format.widow_control = True
                    if (
                        block_index + 1 < len(blocks)
                        and blocks[block_index + 1][0] in {"bullet", "number"}
                    ):
                        paragraph.paragraph_format.keep_with_next = True
            elif block_type in {"bullet", "number"}:
                style = "DMU Bullet" if block_type == "bullet" else "DMU Number"
                paragraph = add_paragraph_before(doc, anchor, payload, style)
                if block_type == "number":
                    if numbered_list_id is None:
                        numbered_list_id = add_numbering_definition(doc, bullet=False)
                    set_paragraph_numbering(paragraph, numbered_list_id)
                paragraph.paragraph_format.keep_together = True
                next_is_same_list = (
                    block_index + 1 < len(blocks)
                    and blocks[block_index + 1][0] == block_type
                )
                paragraph.paragraph_format.space_after = Pt(2 if next_is_same_list else 6)
            elif block_type == "code":
                paragraph = add_paragraph_before(doc, anchor, payload)
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                paragraph.paragraph_format.left_indent = Inches(0.35)
                paragraph.paragraph_format.right_indent = Inches(0.35)
                paragraph.paragraph_format.keep_together = True
                for run in paragraph.runs:
                    run.font.name = "Courier New"
                    run.font.size = Pt(9)
            elif block_type == "image":
                alt_text, image_path = payload
                add_image_before(doc, anchor, image_path, alt_text)
            elif block_type == "table":
                table_count[current_chapter] = table_count.get(current_chapter, 0) + 1
                number = table_count[current_chapter]
                title = TABLE_TITLES[(current_chapter, number)]
                caption = f"Table {current_chapter}.{number}: {title}"
                table_number = f"Table {current_chapter}.{number}"
                caption_paragraph = add_table_before(doc, anchor, payload, caption)
                add_bookmark(doc, caption_paragraph, bookmark_name("table", table_number))
                captions.append(table_number)
    return headings, captions


def bookmark_name(kind: str, label: str) -> str:
    """Return a deterministic Word-safe bookmark name."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", label)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return f"nav_{kind}_{clean}"[:40]


def add_bookmark(doc: Document, paragraph: Paragraph, name: str) -> None:
    """Attach a bookmark spanning one paragraph."""
    bookmark_ids = []
    for element in doc.element.body.iter(qn("w:bookmarkStart")):
        value = element.get(qn("w:id"))
        if value is not None and value.isdigit():
            bookmark_ids.append(int(value))
    bookmark_id = str(max(bookmark_ids, default=0) + 1)

    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), bookmark_id)
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), bookmark_id)

    insert_at = 1 if paragraph._p.pPr is not None else 0
    paragraph._p.insert(insert_at, start)
    paragraph._p.append(end)


def add_internal_hyperlink(paragraph: Paragraph, text: str, target: str) -> None:
    """Append an internal Word hyperlink without changing thesis typography."""
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("w:anchor"), target)
    hyperlink.set(qn("w:history"), "1")
    run = OxmlElement("w:r")
    run_properties = OxmlElement("w:rPr")
    no_proof = OxmlElement("w:noProof")
    run_properties.append(no_proof)
    run.append(run_properties)

    label, page = text.rsplit("\t", 1)
    label_text = OxmlElement("w:t")
    label_text.set(qn("xml:space"), "preserve")
    label_text.text = label
    run.append(label_text)
    run.append(OxmlElement("w:tab"))
    page_text = OxmlElement("w:t")
    page_text.text = page
    run.append(page_text)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def set_outline_level(paragraph: Paragraph, level: int = 0) -> None:
    """Expose a front-matter title in Word/PDF navigation panes."""
    p_pr = paragraph._p.get_or_add_pPr()
    outline = p_pr.find(qn("w:outlineLvl"))
    if outline is None:
        outline = OxmlElement("w:outlineLvl")
        p_pr.append(outline)
    outline.set(qn("w:val"), str(level))


def add_front_entry(
    doc: Document,
    anchor,
    label: str,
    page: str,
    indent: float = 0.0,
    target: str | None = None,
):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(indent)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.keep_together = True
    paragraph.paragraph_format.tab_stops.add_tab_stop(
        Inches(6.1), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS
    )
    entry_text = f"{label}\t{page}"
    if target:
        add_internal_hyperlink(paragraph, entry_text, target)
    else:
        paragraph.add_run(entry_text)
    move_before(paragraph._p, anchor)
    return paragraph


def replace_front_lists(doc: Document, headings: list[str]) -> None:
    _, list_tables = clear_between(doc, "LIST OF FIGURES", "LIST OF TABLES")
    for number, title in FIGURES:
        add_front_entry(
            doc,
            list_tables,
            f"{number}  {title}",
            FIGURE_PAGES[number],
            target=bookmark_name("figure", number),
        )

    _, list_abbr = clear_between(doc, "LIST OF TABLES", "LIST OF ABBREVIATIONS")
    for number, title in TABLES:
        add_front_entry(
            doc,
            list_abbr,
            f"{number}  {title}",
            TABLE_PAGES[number],
            target=bookmark_name("table", number),
        )

    _, abstract_anchor = clear_between(doc, "LIST OF ABBREVIATIONS", "ABSTRACT")
    abbreviations_heading = next(
        p for p in doc.paragraphs if p.text == "LIST OF ABBREVIATIONS"
    )
    abbreviations_heading.paragraph_format.page_break_before = True
    abbreviation_rows = [
        ["Abbreviation", "Full form"],
        ["AoI", "Age of Information"],
        ["CI", "Confidence Interval"],
        ["CTDE", "Centralized Training with Decentralized Execution"],
        ["DEF", "Decision-level Explanation Faithfulness"],
        ["GAT", "Graph Attention Network"],
        ["GNN", "Graph Neural Network"],
        ["GxI", "Gradient x Input"],
        ["LOO", "Leave-One-Out"],
        ["MAPPO", "Multi-Agent Proximal Policy Optimization"],
        ["MARL", "Multi-Agent Reinforcement Learning"],
        ["MLP", "Multilayer Perceptron"],
        ["PPO", "Proximal Policy Optimization"],
        ["RL", "Reinforcement Learning"],
        ["SUMO", "Simulation of Urban Mobility"],
        ["WAMSN", "Weighted Attention Mass on Stale Nodes"],
        ["XAI", "Explainable Artificial Intelligence"],
        ["XRL", "Explainable Reinforcement Learning"],
    ]
    table = doc.add_table(rows=len(abbreviation_rows), cols=2)
    table.style = None
    set_table_geometry(table, [1900, 7120])
    for row_index, values in enumerate(abbreviation_rows):
        row = table.rows[row_index]
        keep_row(row)
        if row_index == 0:
            repeat_header(row)
        for cell, value in zip(row.cells, values):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            cell.paragraphs[0].paragraph_format.line_spacing = 1.0
            run = cell.paragraphs[0].add_run(value)
            run.font.name = "Times New Roman"
            run.font.size = Pt(10)
            if row_index == 0:
                run.bold = True
                set_cell_shading(cell, "E7E6E6")
    move_before(table._tbl, abstract_anchor)

    abstract_heading = next(p for p in doc.paragraphs if p.text == "ABSTRACT")
    abstract_heading.paragraph_format.page_break_before = True

    toc = find_body_child(doc, "TABLE OF CONTENTS")
    toc_heading = next(p for p in doc.paragraphs if p.text == "TABLE OF CONTENTS")
    toc_heading.paragraph_format.page_break_before = True
    chapter_one = find_body_child(doc, "Chapter 1: Introduction")
    between = list(doc.element.body.iterchildren())
    start_index = between.index(toc)
    end_index = between.index(chapter_one)
    section_properties = None
    for child in between[start_index + 1:end_index]:
        sect_pr = child.find(qn("w:pPr") + "/" + qn("w:sectPr")) if child.tag == qn("w:p") else None
        if sect_pr is not None:
            section_properties = deepcopy(sect_pr)
        doc.element.body.remove(child)
    for heading in headings:
        if heading.startswith("Chapter "):
            add_front_entry(
                doc,
                chapter_one,
                heading,
                HEADING_PAGES[heading],
                target=bookmark_name("heading", heading),
            )
        else:
            add_front_entry(
                doc,
                chapter_one,
                heading,
                HEADING_PAGES[heading],
                indent=0.25,
                target=bookmark_name("heading", heading),
            )
    if section_properties is None:
        raise ValueError("front-matter section break not found")
    section_break = doc.add_paragraph()
    section_break._p.get_or_add_pPr().append(section_properties)
    move_before(section_break._p, chapter_one)

    for title, kind in (
        ("TABLE OF CONTENTS", "toc"),
        ("LIST OF FIGURES", "list_of_figures"),
        ("LIST OF TABLES", "list_of_tables"),
    ):
        paragraph = next(p for p in doc.paragraphs if p.text == title)
        add_bookmark(doc, paragraph, bookmark_name("front", kind))
        set_outline_level(paragraph)


def replace_abstract(doc: Document) -> None:
    _, toc = clear_between(doc, "ABSTRACT", "TABLE OF CONTENTS")
    for block_type, payload in collect_blocks(ROOT / "docs" / "dissertation" / "00_abstract.md"):
        if block_type == "heading":
            continue
        if block_type == "paragraph":
            paragraph = add_paragraph_before(doc, toc, payload)
            paragraph.paragraph_format.widow_control = True
    keywords = (
        "Keywords: explainable reinforcement learning; graph attention; "
        "multi-agent reinforcement learning; fleet dispatch; telemetry "
        "degradation; Age of Information; faithfulness."
    )
    paragraph = add_paragraph_before(doc, toc)
    label, terms = keywords.split(":", 1)
    paragraph.add_run(label + ":").bold = True
    paragraph.add_run(terms)


def update_front_text(doc: Document) -> None:
    replacements = {
        "Supervisor: Farhan S. Ujager": "Supervisor: Farhan S. Ujager",
        "August 2026": "September 2026",
        "No publication is claimed in this dissertation draft.":
            "No publication is claimed in this dissertation.",
        "ACKNOWLEDGEMENT": "ACKNOWLEDGMENT",
    }
    for paragraph in doc.paragraphs:
        if paragraph.text in replacements:
            paragraph.text = replacements[paragraph.text]
    acknowledgment = next(p for p in doc.paragraphs if p.text == "ACKNOWLEDGMENT")
    body = acknowledgment._p.getnext()
    while body is not None and body.tag != qn("w:p"):
        body = body.getnext()
    if body is not None:
        for child in list(body):
            if child.tag == qn("w:r"):
                body.remove(child)
        paragraph = next(p for p in doc.paragraphs if p._p is body)
        paragraph.add_run(
            "I would like to express my sincere gratitude to my supervisor, "
            "Dr Farhan S. Ujager, for his valuable support and guidance "
            "throughout this research. His advice, encouragement and careful "
            "feedback helped me refine the experiment, interpret the findings "
            "responsibly and complete this dissertation."
        )


def style_references_and_appendices(doc: Document) -> None:
    references = False
    for paragraph in doc.paragraphs:
        if paragraph.text == "References":
            references = True
            continue
        if paragraph.text == "Appendices":
            references = False
        if references and paragraph.text.strip():
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.left_indent = Inches(0.5)
            paragraph.paragraph_format.first_line_indent = Inches(-0.5)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.space_after = Pt(6)
        if paragraph.text.startswith("The complete reproduction procedure"):
            paragraph.text = (
                "The complete reproduction procedure is provided in "
                "docs/REPRODUCE_EXPERIMENTS.md. Compact citable outputs are "
                "stored in the version-controlled results directory identified "
                "in that guide."
            )
        if paragraph.text.startswith("The project repository contains"):
            paragraph.text = (
                "The repository contains the experiment configuration, source "
                "code, validation selections, preflight reports, per-seed "
                "analyses, summary tables and figure sources used in this "
                "dissertation. Raw cells and checkpoints remain outside the "
                "document because of their size."
            )


def replace_references(doc: Document) -> None:
    """Replace the template's inherited bibliography with cited sources."""
    _, appendices = clear_between(doc, "References", "Appendices")
    source = ROOT / "docs" / "dissertation" / "references.md"
    for block_type, payload in collect_blocks(source):
        if block_type != "paragraph":
            continue
        paragraph = add_paragraph_before(doc, appendices, payload)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        paragraph.paragraph_format.left_indent = Inches(0.5)
        paragraph.paragraph_format.first_line_indent = Inches(-0.5)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(6)


def enable_field_updates(doc: Document) -> None:
    settings = doc.settings._element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")


def ensure_image_alt_text(doc: Document) -> None:
    for index, shape in enumerate(doc.inline_shapes):
        doc_pr = shape._inline.docPr
        if not doc_pr.get("descr") and not doc_pr.get("title"):
            description = (
                "De Montfort University Dubai logo"
                if index == 0
                else "Dissertation figure"
            )
            doc_pr.set("descr", description)


def build(base: Path, output: Path) -> None:
    doc = Document(base)
    style_document(doc)
    update_front_text(doc)
    replace_abstract(doc)

    anchor = remove_range_inclusive(doc, "Chapter 1: Introduction", "List of Publications")
    headings, _ = insert_chapters(doc, anchor)
    replace_front_lists(doc, headings)
    replace_references(doc)
    style_references_and_appendices(doc)
    enable_field_updates(doc)
    ensure_image_alt_text(doc)

    doc.core_properties.title = (
        "When Explanations Outlive Their Data: Faithfulness Decoupling in "
        "Graph-Attention MARL Fleet Dispatch under Telemetry Degradation"
    )
    doc.core_properties.author = "Hongwei Lin"
    doc.core_properties.subject = "MSc Dissertation, De Montfort University Dubai"
    doc.core_properties.keywords = "GAT-MARL, faithfulness, telemetry degradation, AoI"
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    print(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.base, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
