"""Fill the IEEESTEC Word template with the content of paper.md, without touching its formatting.

    /usr/bin/python3 docs/paper/build_paper.py

Runs with the *system* Python, because the LibreOffice UNO bridge (python3-uno) lives there and
not in the Poetry environment. Needs the native (non-snap) LibreOffice: the snap cannot read
/tmp or hidden directories, so a headless profile cannot be created.

How the formatting is preserved
-------------------------------
The organisers' template is edited as .doc throughout. A .doc -> .docx round trip through
LibreOffice was measured to change the author-table row heights and the bullet indentation; a
.doc -> .doc round trip is pixel-identical on the pages it covers.

No paragraph is styled by name. The template's paragraphs carry direct formatting on top of
their styles (spacing, line pitch, the hanging indent of the references), so every new
paragraph is a **copy of a template paragraph of the same kind**, pasted and then given new
text. Tables copy the template table's borders, padding and cell styles. Headings, figure
captions, table titles and references are numbered by the template's own list styles, so no
number is ever typed.

Input : docs/paper/template/IEEESTEC_template.doc (pristine, never written)
        docs/paper/paper.md, figures/paper/*.png
Output: docs/paper/IEEESTEC_paper.doc, docs/paper/IEEESTEC_paper.pdf
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import uno
from com.sun.star.awt import Size
from com.sun.star.beans import PropertyValue
from com.sun.star.beans.PropertyState import DIRECT_VALUE
from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATE = HERE / "template" / "IEEESTEC_template.doc"
SOURCE = HERE / "paper.md"
FIGURES = ROOT / "figures" / "paper"
OUT_DOC = HERE / "IEEESTEC_paper.doc"
OUT_PDF = HERE / "IEEESTEC_paper.pdf"

#: Geometry of the template, in 1/100 mm, read from its own image and table.
COLUMN_WIDTH = 8564
TEXT_WIDTH = 17850

#: Index of each prototype paragraph in the template's body (see the enumeration in
#: `prototypes`). Chosen so that no prototype carries an instruction-coloured run at its start.
PROTO_INDEX = {
    "h1": 9,          # "Ease of Use"
    "h2": 12,         # "Maintaining the Integrity of the Specifications"
    "body": 13,       # "The template is used to format your paper..."
    "bullet": 21,     # second bullet of "Units"
    "figure": 60,     # the paragraph holding the example image
    "caption": 61,    # "Example of a figure caption."
    "h5": 65,         # "References"
    "reference": 67,  # second reference
}
#: Character attributes a pasted paragraph can carry over from the prototype text it replaced.
INHERITED_CHAR_PROPS = (
    "CharPosture", "CharPostureAsian", "CharPostureComplex",
    "CharWeight", "CharWeightAsian", "CharWeightComplex",
    "CharColor", "CharHighlight", "CharBackColor", "CharBackTransparent",
)
#: Direct paragraph attributes of the body prototype (template paragraph 13), which the
#: template's "table head" paragraph does not have.
BODY_DIRECT_PARA_PROPS = ("ParaTopMargin", "ParaBottomMargin", "ParaLineSpacing",
                          "ParaContextMargin")
FIRST_TEMPLATE_BODY = 6   # "Introduction (Heading 1)"
LAST_TEMPLATE_BODY = 73   # last reference


# ------------------------------------------------------------------------------------------
# paper.md

def parse(source: Path) -> dict:
    text = re.sub(r"<!--.*?-->", "", source.read_text(encoding="utf-8"), flags=re.S)
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("@title "))
    paper: dict = {"authors": [], "blocks": [], "references": []}
    i = start
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        if not line.strip():
            continue
        if line.startswith("@title "):
            paper["title"] = line[7:].strip()
        elif line.startswith("@author "):
            paper["authors"].append(line[8:].strip())
        elif line.startswith("@abstract "):
            paper["abstract"] = line[10:].strip()
        elif line.startswith("@keywords "):
            paper["keywords"] = line[10:].strip()
        elif line.startswith("@references"):
            for ref in lines[i:]:
                if ref.strip():
                    paper["references"].append(re.sub(r"^\[\d+\]\s*", "", ref.strip()))
            break
        elif line.startswith("## "):
            paper["blocks"].append(("h2", line[3:].strip()))
        elif line.startswith("# "):
            paper["blocks"].append(("h1", line[2:].strip()))
        elif line.startswith("- "):
            paper["blocks"].append(("bullet", line[2:].strip()))
        elif line.startswith("[FIG "):
            m = re.match(r"\[FIG (\d+): (.+?) \| (column|page) \| (.+)\]$", line)
            paper["blocks"].append(("figure", m.group(2), m.group(3), m.group(4),
                                    int(m.group(1))))
        elif line.startswith("[TABLE "):
            m = re.match(r"\[TABLE [IVX]+: (.+?)(?: \| w=([\d,]+))?\]$", line)
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            widths = [int(w) for w in m.group(2).split(",")] if m.group(2) else None
            paper["blocks"].append(("table", m.group(1), rows, widths))
        else:
            paper["blocks"].append(("body", line.strip()))
    return paper


def segments(text: str) -> list[tuple[str, bool, bool]]:
    """Split *italic* and **bold** markup into (text, italic, bold) runs."""
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            out.append((part[2:-2], False, True))
        elif part.startswith("*"):
            out.append((part[1:-1], True, False))
        else:
            out.append((part, False, False))
    return out


# ------------------------------------------------------------------------------------------
# LibreOffice

def pv(name, value):
    prop = PropertyValue()
    prop.Name, prop.Value = name, value
    return prop


def connect(profile: Path):
    pipe = f"claudepaper{os.getpid()}"
    proc = subprocess.Popen(
        ["soffice", f"-env:UserInstallation={profile.as_uri()}", "--headless", "--invisible",
         "--nologo", "--norestore", f"--accept=pipe,name={pipe};urp;"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(120):
        try:
            ctx = resolver.resolve(f"uno:pipe,name={pipe};urp;StarOffice.ComponentContext")
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        raise SystemExit("LibreOffice did not start")
    desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    return proc, ctx, desktop


def body_elements(doc) -> list:
    out, en = [], doc.Text.createEnumeration()
    while en.hasMoreElements():
        out.append(en.nextElement())
    return out


def paragraph_with_break(doc, paragraph):
    """A cursor over a whole paragraph including its end mark, so a paste carries its attributes."""
    cursor = doc.Text.createTextCursorByRange(paragraph.Start)
    cursor.gotoEndOfParagraph(True)
    cursor.goRight(1, True)
    return cursor


class Builder:
    def __init__(self, ctx, doc):
        self.ctx, self.doc = ctx, doc
        self.controller = doc.CurrentController
        self.view = self.controller.ViewCursor
        elements = body_elements(doc)
        self.proto = {}
        for kind, index in PROTO_INDEX.items():
            self.controller.select(paragraph_with_break(doc, elements[index]))
            self.proto[kind] = self.controller.getTransferable()
        self.table_proto = doc.TextTables.getByName("Table2")
        provider = ctx.ServiceManager.createInstanceWithContext(
            "com.sun.star.graphic.GraphicProvider", ctx)
        self.load_graphic = lambda path: provider.queryGraphic((pv("URL", path.as_uri()),))

    # -- insertion point: the start of the template's first body paragraph -----------------
    def set_anchor(self, paragraph):
        self.view.gotoRange(paragraph.Start, False)

    def _pasted_paragraph(self):
        cursor = self.doc.Text.createTextCursorByRange(self.view.Start)
        cursor.gotoPreviousParagraph(False)
        cursor.gotoEndOfParagraph(True)
        return cursor

    def paragraph(self, kind: str, text: str):
        if kind == "table_head":
            self.controller.insertTransferable(self.proto["body"])
        else:
            self.controller.insertTransferable(self.proto[kind])
        cursor = self._pasted_paragraph()
        cursor.setString("")
        if kind == "table_head":
            cursor.setPropertiesToDefault(BODY_DIRECT_PARA_PROPS)
            cursor.ParaStyleName = "table head"
        self.write_runs(cursor, text)
        return cursor

    @staticmethod
    def write_runs(cursor, text: str):
        """Insert text at the end of `cursor`. Plain runs are reset to the paragraph style, so
        they carry no direct formatting; *italic* and **bold** runs carry only that attribute.

        The reset is needed because inserted text inherits the attributes of the character
        before it, which is how the prototype's instruction colour (red on yellow in the
        template's caption) or a preceding italic run would otherwise leak into the text."""
        insert = cursor.Text.createTextCursorByRange(cursor.End)
        for run, italic, bold in segments(text):
            insert.Text.insertString(insert, run, False)
            insert.collapseToEnd()
            span = insert.Text.createTextCursorByRange(insert.End)
            span.goLeft(len(run), True)
            span.setPropertiesToDefault(INHERITED_CHAR_PROPS)
            if italic:
                span.CharPosture = uno.Enum("com.sun.star.awt.FontSlant", "ITALIC")
            if bold:
                span.CharWeight = 150.0

    def figure(self, filename: str, width_kind: str, caption: str, number: int):
        path = FIGURES / filename
        if width_kind == "column":
            before = set(self.doc.GraphicObjects.ElementNames)
            self.controller.insertTransferable(self.proto["figure"])
            new = set(self.doc.GraphicObjects.ElementNames) - before
            image = self.doc.GraphicObjects.getByName(new.pop())
            self._set_image(image, path, COLUMN_WIDTH)
            # Keep the image paragraph with its caption, or a column break can separate them.
            self._pasted_paragraph().ParaKeepTogether = True
            self.paragraph("caption", caption)
        else:
            self._page_figure(path, caption, number)

    def _set_image(self, image, path: Path, width: int):
        graphic = self.load_graphic(path)
        pixels = graphic.SizePixel
        image.Graphic = graphic
        image.Size = Size(width, round(width * pixels.Height / pixels.Width))

    def _page_figure(self, path: Path, caption: str, number: int):
        """A figure spanning both columns: a frame at the top of the page holding the image
        paragraph and its caption paragraph, both pasted from the template."""
        frame = self.doc.createInstance("com.sun.star.text.TextFrame")
        frame.AnchorType = uno.Enum("com.sun.star.text.TextContentAnchorType", "AT_PARAGRAPH")
        frame.WidthType = 1
        frame.Width = TEXT_WIDTH
        frame.SizeType = 2  # automatic height, grows with the content
        frame.HoriOrient = 2  # centre
        frame.HoriOrientRelation = 8  # page print area
        frame.VertOrient = 1  # top
        frame.VertOrientRelation = 8
        frame.TextWrap = uno.Enum("com.sun.star.text.WrapTextMode", "NONE")
        for side in ("LeftBorder", "RightBorder", "TopBorder", "BottomBorder"):
            line = frame.getPropertyValue(side)
            line.OuterLineWidth = line.InnerLineWidth = line.LineDistance = 0
            line.LineWidth = 0
            frame.setPropertyValue(side, line)
        for margin in ("LeftBorderDistance", "RightBorderDistance",
                       "TopBorderDistance", "BottomBorderDistance"):
            frame.setPropertyValue(margin, 0)
        frame.BottomMargin = 350
        anchor = self.doc.Text.createTextCursorByRange(self.view.Start)
        self.doc.Text.insertTextContent(anchor, frame, False)

        # Fill the frame by pasting the same prototypes inside it.
        self.view.gotoRange(frame.Text.Start, False)
        before = set(self.doc.GraphicObjects.ElementNames)
        self.controller.insertTransferable(self.proto["figure"])
        image = self.doc.GraphicObjects.getByName(
            (set(self.doc.GraphicObjects.ElementNames) - before).pop())
        self._set_image(image, path, TEXT_WIDTH - 200)
        self.controller.insertTransferable(self.proto["caption"])
        cursor = frame.Text.createTextCursorByRange(self.view.Start)
        cursor.gotoPreviousParagraph(False)
        cursor.gotoEndOfParagraph(True)
        cursor.setString("")
        self.write_runs(cursor, caption)
        cursor.ParaIsNumberingRestart = True
        cursor.NumberingStartValue = number
        # The frame starts with one empty paragraph of its own; remove it.
        tail = frame.Text.createTextCursor()
        tail.gotoEnd(False)
        tail.gotoStartOfParagraph(True)
        if tail.String == "":
            tail.goLeft(1, True)
            tail.setString("")
        self.view.gotoRange(anchor.End, False)

    def table(self, title: str, rows: list[list[str]], widths: list[int] | None):
        # A table title stays with its table, and a table is never split: a split table can
        # also end up under a two-column figure that floats to the top of the next page.
        self.paragraph("table_head", title).ParaKeepTogether = True
        proto = self.table_proto
        table = self.doc.createInstance("com.sun.star.text.TextTable")
        table.initialize(len(rows), len(rows[0]))
        self.doc.Text.insertTextContent(self.view.Start, table, False)
        table.Split = False
        table.HoriOrient = proto.HoriOrient
        table.Width = proto.Width
        table.TopMargin = 0
        table.BottomMargin = 200
        table.TableBorder2 = proto.TableBorder2
        if widths:
            total = table.TableColumnRelativeSum
            separators = table.TableColumnSeparators
            position = 0
            for separator, width in zip(separators, widths[:-1], strict=True):
                position += width
                separator.Position = round(total * position / sum(widths))
            table.TableColumnSeparators = separators
        head_cell = proto.getCellByName("A1")
        copy_cell = proto.getCellByName("A3")
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                cell = table.getCellByPosition(c, r)
                source = head_cell if r == 0 else copy_cell
                for prop in ("LeftBorder", "RightBorder", "TopBorder", "BottomBorder",
                             "LeftBorderDistance", "RightBorderDistance",
                             "TopBorderDistance", "BottomBorderDistance", "VertOrient"):
                    cell.setPropertyValue(prop, source.getPropertyValue(prop))
                paragraph = cell.Text.createEnumeration().nextElement()
                paragraph.ParaStyleName = "table col head" if r == 0 else "table copy"
                cursor = cell.Text.createTextCursor()
                self.write_runs(cursor, value)
                if c > 0:
                    paragraph.ParaAdjust = 3  # centre numeric columns, as the sample does
        # Word tables carry no spacing below them and the .doc export drops LibreOffice's, so
        # the text after a table would touch its bottom rule. The template's own sample table
        # is followed by a paragraph for the same reason; an empty body paragraph does it here.
        self.paragraph("body", "")
        return table


# ------------------------------------------------------------------------------------------

def fill_author(doc, lines: list[str]):
    """Single author: keep one cell of the author table, as the template instructs."""
    table = doc.TextTables.getByName("Table1")
    table.Rows.removeByIndex(1, 1)
    table.Columns.removeByIndex(2, 1)
    table.Columns.removeByIndex(0, 1)
    cell = table.getCellByPosition(0, 0)
    paragraphs = []
    en = cell.Text.createEnumeration()
    while en.hasMoreElements():
        paragraph = en.nextElement()
        if paragraph.String.startswith("line "):  # the cell also ends in an empty paragraph
            paragraphs.append(paragraph)
    assert len(paragraphs) == len(lines) == 5, (len(paragraphs), len(lines))
    for paragraph, text in zip(paragraphs, lines, strict=True):
        cursor = cell.Text.createTextCursorByRange(paragraph.Start)
        cursor.goRight(len("line N: "), True)
        assert re.fullmatch(r"line \d: ", cursor.String), cursor.String
        cursor.setString("")
        cursor.gotoEndOfParagraph(True)
        cursor.setString(text)


def replace_after(paragraph, marker: str, text: str):
    """Replace the text after `marker` (e.g. 'Abstract—'), keeping the run formatting there."""
    full = paragraph.String
    offset = full.index(marker) + len(marker)
    cursor = paragraph.Text.createTextCursorByRange(paragraph.Start)
    cursor.goRight(offset, False)
    cursor.gotoEndOfParagraph(True)
    cursor.setString(text)


def direct_properties(obj, prefix: str) -> dict:
    """Properties set directly on `obj` (not inherited from its style) whose name has `prefix`."""
    out = {}
    for prop in obj.PropertySetInfo.Properties:
        if not prop.Name.startswith(prefix) or prop.Name.endswith("InteropGrabBag"):
            continue
        try:
            if obj.getPropertyState(prop.Name) == DIRECT_VALUE:
                out[prop.Name] = obj.getPropertyValue(prop.Name)
        except Exception:
            pass
    return out


def clear_text(paragraph):
    cursor = paragraph.Text.createTextCursorByRange(paragraph.Start)
    cursor.gotoEndOfParagraph(True)
    cursor.setString("")


def build():
    paper = parse(SOURCE)
    with tempfile.TemporaryDirectory(prefix="ieeestec_") as tmp:
        # Work on a copy, so a crashed run can never leave a lock on the pristine template.
        working = Path(tmp) / TEMPLATE.name
        shutil.copyfile(TEMPLATE, working)
        proc, ctx, desktop = connect(Path(tmp) / "profile")
        try:
            doc = desktop.loadComponentFromURL(working.as_uri(), "_blank", 0,
                                               (pv("Hidden", True),))
            elements = body_elements(doc)
            builder = Builder(ctx, doc)

            # Front matter, in place.
            title, notice, _, notice2, abstract, keywords = elements[:6]
            clear_text(title)
            title.Text.insertString(title.Start, paper["title"], False)
            clear_text(notice)
            clear_text(notice2)
            fill_author(doc, paper["authors"])
            replace_after(abstract, "Abstract—", paper["abstract"])
            replace_after(keywords, "Keywords—", paper["keywords"])

            # Body, pasted before the template's first body paragraph.
            first = elements[FIRST_TEMPLATE_BODY]
            builder.set_anchor(first)
            for block in paper["blocks"]:
                kind = block[0]
                if kind in ("h1", "h2", "body", "bullet"):
                    builder.paragraph(kind, block[1])
                elif kind == "figure":
                    builder.figure(block[1], block[2], block[3], block[4])
                elif kind == "table":
                    builder.table(block[1], block[2], block[3])
            builder.paragraph("h5", "References")
            for ref in paper["references"]:
                builder.paragraph("reference", ref)

            # Remove the template's own body: its sample table, its image, then its text.
            doc.TextTables.getByName("Table2").dispose()
            doc.GraphicObjects.getByName("Image1").dispose()
            old = body_elements(doc)
            start = next(i for i, e in enumerate(old)
                         if not e.supportsService("com.sun.star.text.TextTable")
                         and e.String.startswith("Introduction (Heading 1)"))
            end = next(i for i, e in enumerate(old)
                       if not e.supportsService("com.sun.star.text.TextTable")
                       and e.String.startswith("Description of online source"))
            # Each paragraph is disposed on its own. Deleting a text range instead joins the
            # paragraphs at its ends, and the join carries attributes across: it dragged the
            # template's closing single-column section (which balances the last page's
            # columns) into the two-column section as a stray heading.
            for element in old[start:end + 1]:
                element.dispose()
            # The template's closing paragraph (the empty one that forms the single-column
            # section which balances the last page) carries direct red / SimSun character
            # attributes. It has no text, so they are invisible there, but the .doc export
            # hands them to the paragraph before the section break -- the pristine template
            # shows the same defect on its own last reference after a round trip. Clearing
            # them on the empty paragraph changes nothing visible and keeps the last
            # reference in the references style.
            closing = body_elements(doc)[-1]
            for name in direct_properties(closing, "Char"):
                closing.setPropertyToDefault(name)

            doc.storeToURL(OUT_DOC.as_uri(), (pv("FilterName", "MS Word 97"),))
            doc.close(True)

            # Export the PDF from the saved .doc, so the preview is what the file contains.
            doc = desktop.loadComponentFromURL(OUT_DOC.as_uri(), "_blank", 0,
                                               (pv("Hidden", True),))
            doc.storeToURL(OUT_PDF.as_uri(), (pv("FilterName", "writer_pdf_Export"),))
            doc.close(True)
        finally:
            # Terminating only the soffice wrapper leaves soffice.bin running; ask it to quit.
            try:
                desktop.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    print(f"wrote {OUT_DOC}\nwrote {OUT_PDF}")


if __name__ == "__main__":
    build()
