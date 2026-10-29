"""
html_to_latex.py -- convert the generated report HTML (Assignment3_Report.html,
Assignment3_Short_Report.html) into LaTeX sources that compile with pdflatex
(e.g. on Overleaf).

Output: report/latex/<name>.tex plus report/latex/figures/*.png, and a zip of
the folder for uploading to Overleaf. Only the tags emitted by
build_report.py / build_short_report.py are handled: headings, paragraphs,
lists, tables (as longtable, so they can break across pages), figures, <pre>
blocks and inline b/i/sub/sup/code/br. Unicode symbols are mapped to LaTeX
commands so no special fonts or engines are needed.

Run from the project root:   .venv/bin/python report/html_to_latex.py
"""
import re
import shutil
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

HERE = Path(__file__).resolve().parent
OUT = HERE / "latex"
BLOCK_TAGS = {"p", "div", "ul", "ol", "table", "figure", "pre", "h1", "h2", "h3", "h4", "li"}

# Unicode characters used in the reports -> pdflatex-safe LaTeX
UNICODE = {
    "±": r"$\pm$", "–": "--", "—": "---", "→": r"$\rightarrow$", "←": r"$\leftarrow$",
    "σ": r"$\sigma$", "α": r"$\alpha$", "ε": r"$\varepsilon$", "δ": r"$\delta$", "Σ": r"$\Sigma$",
    "Δ": r"$\Delta$", "“": "``", "”": "''", "‘": "`", "’": "'", "−": r"$-$", "∞": r"$\infty$",
    "·": r"$\cdot$", "×": r"$\times$", "∈": r"$\in$", "≥": r"$\geq$", "≤": r"$\leq$",
    "≈": r"$\approx$", "‖": r"$\|$", "é": r"\'e", "ü": r"\"u", "\u00a0": "~",
}

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=16mm]{geometry}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{array}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{float}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{amsmath,amssymb}
\usepackage[hidelinks]{hyperref}
\setlist{nosep,leftmargin=*}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlength{\LTpre}{4pt}
\setlength{\LTpost}{6pt}
\renewcommand{\arraystretch}{1.15}
\sloppy
\emergencystretch=3em
\pagestyle{plain}
\begin{document}
"""


# ---------------------------------------------------------------------------
# text escaping
# ---------------------------------------------------------------------------
def tex_escape(s, mono=False):
    """Escape LaTeX special characters, then map Unicode symbols to LaTeX."""
    s = re.sub(r"\s+", " ", s)
    out = []
    for ch in s:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "<":
            out.append(r"\textless{}")
        elif ch == ">":
            out.append(r"\textgreater{}")
        elif ch == "/":
            out.append(r"/\allowbreak{}")          # lets long paths/URLs wrap
        elif ch in ("\u0304", "\u0303"):
            # combining macron / tilde on the previous letter (g-bar, g-tilde)
            prev = out.pop() if out else ""
            out.append((r"$\bar{%s}$" if ch == "\u0304" else r"$\tilde{%s}$") % prev)
        else:
            out.append(UNICODE.get(ch, ch))
    return "".join(out)


def inline(node, mono=False):
    """LaTeX for the inline content of a node (b/i/sub/sup/code/br kept)."""
    parts = []
    for ch in node.children:
        if isinstance(ch, NavigableString):
            parts.append(tex_escape(str(ch), mono))
            continue
        name = ch.name
        inner = inline(ch, mono)
        if name in ("b", "strong", "th"):
            parts.append(r"\textbf{" + inner + "}")
        elif name in ("i", "em"):
            parts.append(r"\textit{" + inner + "}")
        elif name == "sub":
            parts.append(r"\textsubscript{" + inner + "}")
        elif name == "sup":
            parts.append(r"\textsuperscript{" + inner + "}")
        elif name == "code":
            parts.append(r"\texttt{" + inline(ch, True) + "}")
        elif name == "br":
            parts.append(r"\newline ")
        else:
            parts.append(inner)
    return "".join(parts)


def has_block_children(el):
    return any(getattr(c, "name", None) in BLOCK_TAGS for c in el.children)


def pre_text(el):
    """A <pre> block: monospace, wrapped, line breaks kept."""
    lines = [tex_escape(l) for l in el.get_text().split("\n")]
    return r"{\ttfamily\footnotesize " + r"\newline ".join(lines) + "}"


# ---------------------------------------------------------------------------
# block content
# ---------------------------------------------------------------------------
class Converter:
    def __init__(self):
        self.out = []

    def emit(self, s):
        self.out.append(s)

    def block(self, el):
        name, cls = el.name, el.get("class") or []
        if name in ("h1", "h2", "h3", "h4"):
            text = inline(el).strip()
            if name == "h1":
                self.emit(r"\begin{center}{\Large\bfseries " + text + r"\par}\end{center}")
            elif name == "h2":
                if "pb" in cls:
                    self.emit(r"\clearpage")
                self.emit(r"\section*{" + text + "}")
            elif name == "h3":
                self.emit(r"\subsection*{" + text + "}")
            else:
                self.emit(r"\subsubsection*{" + text + "}")
        elif name in ("ul", "ol"):
            env = "enumerate" if name == "ol" else "itemize"
            self.emit(r"\begin{%s}" % env)
            for li in el.find_all("li", recursive=False):
                self.list_item(li)
            self.emit(r"\end{%s}" % env)
        elif name == "table":
            self.table(el)
        elif name == "figure":
            self.figure(el)
        elif name == "pre":
            self.emit(r"\begin{quote}" + pre_text(el) + r"\end{quote}")
        elif name in ("div", "p", "span") and has_block_children(el):
            if "title" in cls:                       # title page: no page number, some top space
                self.emit(r"\thispagestyle{empty}\vspace*{25mm}")
                self.container(el)
            elif "box" in cls:
                self.emit(r"\noindent\fbox{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule}{")
                self.container(el)
                self.emit("}}")
            else:
                self.container(el)
        elif name in ("p", "div", "span"):
            text = inline(el).strip()
            if not text:
                return
            style = el.get("style", "")
            if "box" in cls:
                text = r"\noindent\fbox{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule}{" + text + "}}"
            if "note" in cls or "plog" in cls:
                text = r"{\small " + text + "}"
            if "meta" in cls or "sub" in cls or "center" in style:
                text = r"\begin{center}" + text + r"\end{center}"
            self.emit(text + "\n")

    def container(self, el):
        """Walk a container, gathering loose inline content into paragraphs."""
        buf = []

        def flush():
            if buf:
                tmp = BeautifulSoup("<p></p>", "html.parser").p
                for b in buf:
                    tmp.append(b)
                self.block(tmp)
                buf.clear()

        for ch in list(el.children):
            if getattr(ch, "name", None) in BLOCK_TAGS:
                flush()
                self.block(ch)
            else:
                buf.append(ch.__copy__() if hasattr(ch, "__copy__") else ch)
        flush()

    def list_item(self, li):
        parts = []
        nested = []
        for ch in li.children:
            if getattr(ch, "name", None) in ("ul", "ol"):
                nested.append(ch)
            else:
                wrap = BeautifulSoup("<span></span>", "html.parser").span
                wrap.append(ch.__copy__() if hasattr(ch, "__copy__") else ch)
                parts.append(inline(wrap))
        self.emit(r"\item " + "".join(parts).strip())
        for n in nested:
            self.block(n)

    def table(self, el):
        rows = el.find_all("tr")
        if not rows:
            return
        ncol = max(len(r.find_all(["td", "th"])) for r in rows)
        # column widths proportional to the longest cell text (clamped), as p{} columns
        lens = [4] * ncol
        for r in rows:
            for j, c in enumerate(r.find_all(["td", "th"])):
                lens[j] = max(lens[j], min(len(c.get_text(" ", strip=True)), 60))
        total = sum(lens)
        avail = 0.98 - 0.012 * ncol                       # leave room for the column padding
        spec = "".join(r">{\raggedright\arraybackslash}p{%.3f\linewidth}" % (avail * l / total) for l in lens)
        size = r"\footnotesize" if ncol <= 6 else r"\scriptsize"
        self.emit(r"{%s\setlength{\tabcolsep}{3pt}" % size)
        self.emit(r"\begin{longtable}{%s}" % spec)
        self.emit(r"\toprule")
        head = el.find("thead")
        body_rows = rows
        if head is not None:
            hr = head.find_all("tr")
            for r in hr:
                self.emit(" & ".join(r"\textbf{" + inline(c).strip() + "}" for c in r.find_all(["td", "th"])) + r" \\")
            self.emit(r"\midrule\endhead")
            body_rows = [r for r in rows if r not in hr]
        for r in body_rows:
            cells = []
            for c in r.find_all(["td", "th"]):
                pre = c.find("pre")
                cells.append(pre_text(pre) if pre is not None else inline(c).strip())
            cells += [""] * (ncol - len(cells))
            self.emit(" & ".join(cells) + r" \\")
        self.emit(r"\bottomrule")
        self.emit(r"\end{longtable}}")

    def figure(self, el):
        img = el.find("img")
        cap = el.find("figcaption")
        self.emit(r"\begin{figure}[H]\centering")
        if img is not None:
            m = re.search(r"width:\s*([\d.]+)%", img.get("style", ""))
            frac = float(m.group(1)) / 100 if m else 1.0
            src = Path(img["src"])
            (OUT / "figures").mkdir(parents=True, exist_ok=True)
            shutil.copy(HERE / src, OUT / "figures" / src.name)
            self.emit(r"\includegraphics[width=%.2f\linewidth]{figures/%s}" % (frac, src.name))
        if cap is not None:
            self.emit(r"\caption*{\small " + inline(cap).strip() + "}")
        self.emit(r"\end{figure}")

    def convert(self, html_path, out_path, title):
        soup = BeautifulSoup(Path(html_path).read_text(encoding="utf-8"), "html.parser")
        for ch in soup.body.children:
            if getattr(ch, "name", None):
                self.block(ch)
        body = "\n".join(self.out)
        tex = (PREAMBLE.replace(r"\begin{document}",
                                r"\hypersetup{pdftitle={%s},pdfauthor={Payal Verma}}" % title + "\n" + r"\begin{document}")
               + body + "\n\\end{document}\n")
        Path(out_path).write_text(tex, encoding="utf-8")
        print("LaTeX written:", out_path)


def check_balance(path):
    """Cheap structural check (no TeX installed here): braces and environments balance."""
    tex = Path(path).read_text(encoding="utf-8")
    t = re.sub(r"\\[{}]", "", tex)                      # ignore escaped braces
    depth = 0
    for ch in t:
        depth += (ch == "{") - (ch == "}")
        assert depth >= 0, f"{path}: unbalanced '}}'"
    assert depth == 0, f"{path}: {depth} unclosed '{{'"
    begins = re.findall(r"\\begin\{(\w+\*?)\}", tex)
    ends = re.findall(r"\\end\{(\w+\*?)\}", tex)
    for env in set(begins) | set(ends):
        assert begins.count(env) == ends.count(env), f"{path}: environment {env} unbalanced"
    print(f"structure OK: {path.name} ({len(begins)} environments)")


if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for name, title in (("Assignment3_Short_Report", "RAI-FM Assignment 3 Short Report"),
                        ("Assignment3_Report", "RAI-FM Assignment 3 Report")):
        Converter().convert(HERE / f"{name}.html", OUT / f"{name}.tex", title)
        check_balance(OUT / f"{name}.tex")
    shutil.make_archive(str(HERE / "Assignment3_LaTeX"), "zip", OUT)
    print("zip written:", HERE / "Assignment3_LaTeX.zip")
