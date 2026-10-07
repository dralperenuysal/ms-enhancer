"""One-shot transform: paper/main.tex -> paper/neurogenetics/main.tex.

Applies the Neurogenetics (Springer) submission-guideline changes:
- sn-nature -> sn-basic class, section order Intro -> Materials and Methods ->
  Results -> Discussion (Methods block moved before Results)
- abstract trimmed to <= 250 words
- figure paths renamed to figures/FigN.eps (journal file-naming rule)
- panel letters lowercase (a)/(b)/(c), "Fig." in-text citation style
- no trailing punctuation in figure captions, no internal log paths
- Fig. 1 (study design) now cited in the Introduction
- added "Consent for publication" declaration
"""
import re

SRC = "paper/main.tex"
DST = "paper/neurogenetics/main.tex"

with open(SRC, encoding="utf-8") as f:
    t = f.read()

# 1. Class option: standard Springer style (numbered sections, Methods before
#    Results) instead of the Nature-portfolio style.
t = t.replace(
    r"\documentclass[pdflatex,sn-nature]{sn-jnl}",
    r"\documentclass[pdflatex,sn-basic]{sn-jnl}",
)
# pdflatex cannot include EPS directly; convert on the fly (TeX Live ships
# the restricted repstopdf helper for this).
t = t.replace(
    "\\usepackage{graphicx}",
    "\\usepackage{graphicx}\n\\usepackage{epstopdf}",
)
# 1b. Bookmark-level fixes: the class declares \bmhead at subparagraph level
# (5) and Methods uses \paragraph (4), which makes hyperref emit
# "Difference between bookmark levels" warnings at every backmatter heading
# and at the first \paragraph. Map both to level 3 (subsubsection) so no
# consecutive jump exceeds one.
t = t.replace(
    "\\usepackage{epstopdf}",
    "\\usepackage{epstopdf}\n\n\\makeatletter\n\\renewcommand*\\toclevel@paragraph{3}%\n\\renewcommand*\\toclevel@subparagraph{2}%\n\\makeatother",
)
# 1c. Floats: replace the bare [h] table specifier (LaTeX auto-promotes it to
# [ht] with a warning) with [ht] directly.
t = t.replace("\\begin{table}[h]", "\\begin{table}[ht]")

# 2. Move the Methods block before Results and retitle it.
m = re.search(r"\\section\{Methods\}\\label\{sec:methods\}.*?(?=\\backmatter)", t, re.S)
assert m, "Methods block not found"
block = m.group(0).rstrip()
t = t.replace(block, "")
block = block.replace(r"\section{Methods}", r"\section{Materials and Methods}")
t = t.replace(r"\section{Results}", block + "\n\n\\section{Results}", 1)

# 3. Trimmed abstract (<= 250 words).
abstract_new = r"""Sequence-to-function deep-learning oracles are increasingly used to score and
select designed regulatory elements, but it is not established that an effect measured for
a sequence in one genomic context holds in another. We generated candidate cis-regulatory
elements for multiple sclerosis risk loci with a per-cell-type generative model, selected
them with a chromatin-accessibility oracle (Enformer), and tested, by direct
intervention, what the oracle rewards. Motif density and a specific transcription-factor site were
not causal for the oracle's preference; a composition-preserving CpG-content edit was, and
its sign was set by the host locus rather than by the edit itself. Placing the same insert
and the same CpG edit at 24 host loci showed the effect varies with
$I^2 = 95\%$ (Cochran's $Q = 458.6$, $p = 1.4\times10^{-82}$) and reverses at
some loci, with no tested host property predicting its direction. The heterogeneity,
though not the locus-specific sign reversal, replicated in a second, independently trained
oracle (Borzoi). Scored against measured reporter activity from a lentiviral massively
parallel reporter assay, the selection objective did not predict measured cell-type
specificity, including for a matched positive control. Separately, a sixth-order Markov
chain, fitted in seconds, matched or exceeded a trained autoregressive
transformer on every generative property this task required. These results
indicate that a sequence-level effect learned or measured by a genomic oracle is not, by
default, a property of the sequence alone: it depends on where in the genome the sequence
sits, and that dependence is not predictable from cheap host features."""
m = re.search(r"\\abstract\{.*?\}(?=\s*\\keywords)", t, re.S)
assert m, "abstract not found"
t = t[: m.start()] + "\\abstract{" + abstract_new + "}\n" + t[m.end():]

# 4. Figure file names: figures/fig_*.pdf -> figures/FigN.eps (journal rule:
#    name figure files "Fig" + figure number; EPS preferred for vector art).
name_map = {
    "fig_design.pdf": "Fig1.eps",
    "fig_pooling_effect.pdf": "Fig2.eps",
    "fig_markov_vs_transformer.pdf": "Fig3.eps",
    "fig_variance.pdf": "Fig4.eps",
    "fig_transfer.pdf": "Fig5.eps",
    "fig_motif_dose.pdf": "Fig6.eps",
    "fig_jaspar_gc.pdf": "Fig7.eps",
    "fig_locus_survey.pdf": "Fig8.eps",
    "fig_mpra_calibration.pdf": "Fig9.eps",
}
for old, new in name_map.items():
    t = t.replace("figures/" + old, "figures/" + new)

# 5. Panel letters lowercase in captions and in-text references.
t = re.sub(r"\\textbf\{\((A|B|C)\)\}", lambda m: r"\textbf{(" + m.group(1).lower() + ")}", t)
t = re.sub(r"(\\ref\{fig:[^}]*\})([ABC])", lambda m: m.group(1) + m.group(2).lower(), t)
# "narrower than in A" style references inside captions/body
t = t.replace("than in A", "than in a").replace("in A;", "in a;")

# 6. Figure captions: no trailing punctuation (label kept intact).
def strip_caption_trailing_period(mo):
    body = mo.group(1)
    if body.rstrip().endswith("."):
        body = body.rstrip()[:-1]
    return "\\caption{" + body + "}\n\\label{" + mo.group(2) + "}"

t = re.sub(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}\s*\\label\{(fig:[^}]*)\}", strip_caption_trailing_period, t)

# 7. No internal log paths in captions.
t = t.replace("; real scored data from \\texttt{logs/transfer.json})", ")")

# 8. Cite the study-design figure in the Introduction.
t = t.replace(
    "measured reporter activity in an independent reporter assay.",
    "measured reporter activity in an independent reporter assay. The overall design and "
    "its checks are summarised in Fig.~\\ref{fig:design}.",
    1,
)

# 9. Springer convention: "Fig." for in-text figure citations.
t = t.replace("Figure \\ref", "Fig.~\\ref").replace("Figure~\\ref", "Fig.~\\ref")

# 10. Section-name reference.
t = t.replace("checkpoints named in Methods.", "checkpoints named in the Materials and Methods.")

# 11. Add "Consent for publication" declaration after the ethics statement.
t = t.replace(
    r"""\bmhead{Ethics approval and consent to participate}

This study used only publicly available, previously published datasets (GEO, ENCODE, GWAS Catalog) and involved no human or animal participants, so no ethics approval was required.
""",
    r"""\bmhead{Ethics approval and consent to participate}

This study used only publicly available, previously published datasets (GEO, ENCODE, GWAS Catalog) and involved no human or animal participants, so no ethics approval was required.

\bmhead{Consent for publication}

Not applicable.

\bmhead{Use of AI and AI-assisted technologies}

The large language model DeepSeek was used for copywriting support and for
editorial revisions of the manuscript (wording and clarity). All content was
reviewed and approved by the author, who takes full responsibility for the
final version of the text.
""",
)

with open(DST, "w", encoding="utf-8") as f:
    f.write(t)

# Report abstract word count.
body = re.search(r"\\abstract\{(.*?)\}(?=\s*\\keywords)", t, re.S).group(1)
body = re.sub(r"\\[a-zA-Z]+", " ", body)
words = len(body.split())
print(f"wrote {DST}; abstract words = {words}")
