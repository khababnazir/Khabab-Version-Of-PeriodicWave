"""Build speed_study_report.pdf: why 18 electrons were slow, and why folx + slogdet fixes it.

Run from the folder that holds the figure PNGs and speed_results.jsonl.
"""
import json
import os

import matplotlib as mpl
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(mpl.get_data_path(), "fonts", "ttf")
for name, file in [("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
                   ("Sans-Italic", "DejaVuSans-Oblique.ttf"), ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONTS, file)))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic", boldItalic="Sans-Bold")

R = {json.loads(l)["run"]: json.loads(l)["result"]
     for l in open(os.path.join(HERE, "speed_results.jsonl")).read().strip().splitlines()}
S = json.load(open(os.path.join(HERE, "singular_example.json")))

body = ParagraphStyle("body", fontName="Sans", fontSize=9.5, leading=13.5, spaceAfter=6, alignment=TA_LEFT)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, spaceAfter=4)
caption = ParagraphStyle("caption", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#333333"),
                         spaceBefore=2, spaceAfter=10)
h1 = ParagraphStyle("h1", parent=body, fontName="Sans-Bold", fontSize=13, leading=16, spaceBefore=12, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=body, fontName="Sans-Bold", fontSize=10.5, leading=13.5, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", parent=body, fontName="Sans-Bold", fontSize=17, leading=21, spaceAfter=4)
sub = ParagraphStyle("sub", parent=body, fontSize=10, leading=13, textColor=colors.HexColor("#444444"))
cell = ParagraphStyle("cell", parent=body, fontSize=7.6, leading=9.2, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Sans-Bold")
mono = ParagraphStyle("mono", parent=body, fontName="Mono", fontSize=7.6, leading=9.8, leftIndent=10,
                      backColor=colors.HexColor("#f4f4f4"), borderPadding=4, spaceBefore=4, spaceAfter=8)

story = []
def P(text, style=body):
    if style in (h1, h2):
        story.append(CondPageBreak(3 * cm))     # never leave a heading alone at the bottom of a page
    story.append(Paragraph(text, style))


def M(text):
    story.append(KeepTogether([Paragraph(text.replace(" ", "&nbsp;").replace("\n", "<br/>"), mono)]))


def figure(file, width_cm, text):
    path = os.path.join(HERE, file)
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(path).getSize()
    story.append(KeepTogether([Image(path, width=width_cm * cm, height=width_cm * cm * h / w),
                               Paragraph(text, caption)]))


def table(rows, widths, header=True, shade=None):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="LEFT", repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")))
    for r in shade or []:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e6f4f1")))
    t.setStyle(TableStyle(style))
    story.append(KeepTogether([t]) if len(rows) <= 9 else t)
    story.append(Spacer(1, 8))


def box(paragraphs):
    inner = [[Paragraph(p, body)] for p in paragraphs]
    t = Table(inner, colWidths=[16.4 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f7fb")),
                           ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#1f5fa8")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story.append(t)
    story.append(Spacer(1, 8))


def gib(x):
    return f"{x:.2f} GiB" if x >= 0.1 else f"{x * 1024:.0f} MiB"


full18 = R["full_18e_attention_folx_slogdet_b1024"]["seconds_per_iteration"]
full18_4096 = R["full_18e_attention_folx_slogdet_b4096"]["seconds_per_iteration"]
mem18_4096 = R["full_18e_attention_folx_slogdet_b4096"]["peak_GiB"]
full6 = R["full_6e_attention_folx_slogdet"]["seconds_per_iteration"]
sn18 = R["full_18e_slaternet_folx_slogdet_b1024"]["seconds_per_iteration"]
sn18_4096 = R["full_18e_slaternet_folx_slogdet_b4096"]["seconds_per_iteration"]
d64 = {(n, r): R[f"det_alone_{n}e_{r}_b64"] for n in (6, 18) for r in ("folx_det", "folx_slogdet", "hessian_det", "hessian_slogdet")}
d1k = {(n, r): R[f"det_alone_{n}e_{r}_b1024"] for n in (6, 18) for r in ("folx_det", "folx_slogdet", "hessian_det", "hessian_slogdet")}
est_fail = d64[(18, "folx_det")]["compiled_scratch_GiB"] * 16

# ---------------------------------------------------------------------------------------------- title
P("Why 18 electrons were slow in our code, and why folx + slogdet fixes it", title)
P("Stage 8 speed study, moire-nnvmc. 30 October 2026. Measurements on khabab-server, one RTX 3090 (GPU 0).", sub)
P("Written for a reader who knows a little calculus and has never heard of folx. Every number in this report "
  "was measured unless it is marked <i>estimate</i> or <i>interpretation</i>.", sub)
story.append(Spacer(1, 8))

box(["<b>The short version</b>",
     "1. When we moved from 6 to 18 electrons, one training iteration of our attention network took "
     "<b>0.577 s</b>, against <b>0.33 s</b> for the authors' code, and 4096 walkers did not fit on the GPU at all. "
     "82 % of our time went into one piece: the <b>local energy</b>, which needs second derivatives.",
     "2. Our Stage 4 code computes those second derivatives the simple way: it builds the full table of "
     "36 × 36 = 1,296 second derivatives and then keeps only the 36 on the diagonal. At 6 electrons the table "
     "has only 144 entries, so it did not matter.",
     "3. The authors use a library called <b>folx</b> that computes only what is needed. When we tried folx in "
     "our code, it failed with <font face='Mono'>INTERNAL: No valid config found</font>. The reason is our "
     "<b>determinant step</b>: Stage 3 adds the determinants with plain <font face='Mono'>det</font>. folx has a "
     "special shortcut rule for <font face='Mono'>slogdet</font> but none for <font face='Mono'>det</font>, so for "
     "<font face='Mono'>det</font> it falls back to building a huge table, which at 18 electrons needs about "
     f"{est_fail:.0f} GiB for 1024 walkers. The card has 24 GiB.",
     "4. Adding the determinants with <font face='Mono'>slogdet</font>, as the authors do, lets folx use its "
     "shortcut. The local energy then takes <b>0.130 s and 1.37 GiB</b> instead of 0.476 s and 16.9 GiB, which is "
     "the same as the authors' own code (0.1305 s, 1.37 GiB).",
     f"5. A whole training iteration now takes <b>{full18:.3f} s</b> at 1024 walkers (before: 0.577 s; authors: "
     f"0.33 s) and <b>{full18_4096:.2f} s</b> at 4096 walkers, using {mem18_4096:.2f} GiB (authors: 1.06 s, 16.6 GiB). "
     "So your friend was right: one RTX 3090 handles 18 electrons easily, once the second derivatives are "
     "computed the right way.",
     "6. Why our code was faster at 6 electrons: there, the local energy was small for everyone, and what mattered "
     "was the rest of the iteration. Our rest took about 0.04 s; the authors' took about 0.12 s. At 18 electrons "
     "our slow local energy outgrew that advantage.",
     "7. The price: <font face='Mono'>slogdet</font> returns NaN if one determinant is <i>exactly</i> zero, which is "
     "why Stage 3 chose <font face='Mono'>det</font> in the first place (Section 8 shows an example). The authors, "
     "FermiNet and Psiformer all accept this risk. We will keep both versions and test that they agree."])

# ---------------------------------------------------------------------------------------------- 1 words
P("1. Words used in this report", h1)
table([["word", "meaning", "example"],
       ["walker", "One guess of where all the electrons are: an (x, y) position for each electron.",
        "18 electrons → 36 numbers per walker."],
       ["batch", "How many walkers are processed together in one iteration. A bigger batch gives a less noisy "
        "energy but needs more time and memory.", "1024 (the authors' released settings) or 4096 (paper Table II)."],
       ["splitting the batch", "Giving part of the walkers to each GPU. Each GPU does its share, and the results "
        "are combined.", "4096 walkers on 2 GPUs = 2048 per GPU."],
       ["iteration", "One training step: move the walkers, compute the energy at each walker, update the network.",
        "Table II asks for 150,000 iterations."],
       ["local energy", "E_loc = (HΨ)/Ψ at one walker. Its average over the walkers is the energy we minimise.",
        "Kinetic + moiré potential + Coulomb."],
       ["Laplacian", "The sum of the second derivatives of Ψ, one for each coordinate. The kinetic energy needs it.",
        "36 second derivatives for 18 electrons."],
       ["Hessian", "The full table of all second derivatives ∂²f/∂x_i∂x_j, including the mixed ones.",
        "36 × 36 = 1,296 entries for 18 electrons."],
       ["determinant", "The antisymmetric part of the wave function. Our networks add 4 determinants.",
        "4 matrices of size 18 × 18 for 18 electrons."],
       ["det, slogdet", "Two ways to compute a determinant in JAX: det gives the number itself; slogdet gives its "
        "sign and the logarithm of its size.", "det = −0.002 ↔ slogdet = (sign −1, log 0.002 = −6.21)."],
       ["folx", "A library that computes a Laplacian in one forward pass, without building the Hessian "
        "('forward Laplacian'). The authors use it.", "Version 0.2.19 in our environment."],
       ["scratch memory", "Temporary GPU memory a calculation needs while it runs.", "An RTX 3090 has 24 GiB."]],
      [2.6, 8.4, 5.4])

# ---------------------------------------------------------------------------------------------- 2 iteration
P("2. What one training iteration does, and where the time goes", h1)
P("Every iteration does three things: (a) <b>move</b> each walker 30 times with the Metropolis rule, which needs "
  "only log|Ψ|; (b) compute the <b>local energy</b> at every walker, which needs the Laplacian; (c) update the "
  "network with <b>KFAC</b>. We timed each piece on its own (attention network, released sizes, 1024 walkers):")
table([["piece", "6 electrons, ours", "18 electrons, ours", "18 electrons, authors"],
       ["log|Ψ| once (for reference)", "0.001 s", "0.0028 s, 0.07 GiB", "0.0027 s, 0.07 GiB"],
       ["(a) 30 Metropolis moves", "0.023 s", "0.071 s", "not timed separately"],
       ["(b) local energy", "0.049 s, 1.93 GiB", "<b>0.476 s, 16.9 GiB</b>", "0.1305 s, 1.37 GiB"],
       ["(c) KFAC update", "0.020 s", "0.034 s, 0.58 GiB", "not timed separately"],
       ["whole iteration", "0.087 s", "0.577 s", "0.33 s"]],
      [4.6, 3.6, 4.0, 4.2])
P("At 18 electrons the local energy is 0.476 s of our 0.577 s. Everything else together is about 0.1 s. "
  "So the whole difference from the authors' code is in one piece, and the rest of this report is about that piece.")

# ---------------------------------------------------------------------------------------------- 3 Laplacian
P("3. Two ways to compute a Laplacian", h1)
P("<b>A small example.</b> Take f(x, y) = x²y. Its four second derivatives are ∂²f/∂x² = 2y, ∂²f/∂y² = 0 and "
  "the two mixed ones, ∂²f/∂x∂y = ∂²f/∂y∂x = 2x. The Hessian is the table [[2y, 2x], [2x, 0]]. The Laplacian is "
  "only the diagonal: 2y + 0 = 2y. The mixed derivatives are never used. With 2 coordinates you compute 4 "
  "numbers and use 2. With 36 coordinates you compute 1,296 and use 36 (Figure 1).")
figure("figA_hessian_vs_laplacian.png", 15.5,
       "<b>Figure 1.</b> The Laplacian adds up only the diagonal of the Hessian (orange). Going from 6 to 18 "
       "electrons, the needed part grows 3 times (12 → 36) but the full table grows 9 times (144 → 1,296).")
P("<b>Route 1: the full Hessian (our Stage 4).</b> Stage 4's <font face='Mono'>kinetic_energy</font> calls "
  "<font face='Mono'>jax.hessian</font>, once for log|Ψ| and once for the phase. JAX does this in two passes. "
  "First it records the whole network calculation so that it can run it backwards (that gives the 36 first "
  "derivatives). Then it repeats this forward-and-backward calculation in 36 directions at once, keeping the "
  "intermediate results of all 36 directions in memory together. Its docstring says so honestly: <i>\"fast enough for a few "
  "electrons\"</i>. Measured cost of the local energy: 0.049 s and 1.93 GiB at 6 electrons; 0.476 s and 16.9 GiB "
  "at 18 electrons. At 4096 walkers it asked for a single 52 GB block of memory and stopped.")
P("<b>Route 2: the forward Laplacian (folx).</b> folx goes through the network once, forwards only. For every "
  "intermediate number u it carries three things: the value u, its 36 first derivatives ∇u, and one running "
  "total, the Laplacian Δu. It never builds the 36 × 36 table and throws intermediate results away as soon as "
  "they are used. For this to work, folx needs a <b>rule</b> for each kind of operation, saying how (u, ∇u, Δu) "
  "become (y, ∇y, Δy). Example, for y = exp(u):")
M("y  = exp(u)\n∇y = exp(u) · ∇u\nΔy = exp(u) · (Δu + |∇u|²)        ← no table needed, only u, ∇u and Δu")
P("folx has such rules for adding, multiplying, exp, matrix products and most other building blocks of a network. "
  "The question is what happens when it meets an operation for which it has <i>no</i> rule.")

# ---------------------------------------------------------------------------------------------- 4 det
P("4. Why folx + our Stage 3 determinant failed", h1)
P("<b>What Stage 3 does.</b> Our <font face='Mono'>sum_of_determinants</font> (Stage 3, "
  "<font face='Mono'>slaternet.py</font>) first rescales the 4 matrices so nothing overflows, and then adds "
  "<font face='Mono'>jnp.linalg.det</font> of each. We chose plain det on purpose: its derivative stays finite "
  "even when a matrix is singular (Section 8). That was a good choice for the Hessian route. It is a bad one for "
  "folx.")
P("<b>What folx knows.</b> We read folx's own source code (version 0.2.19). Its list of special rules "
  "(<font face='Mono'>folx/wrapped_functions.py</font>) contains an entry <font face='Mono'>'slogdet'</font>, "
  "implemented in <font face='Mono'>folx/custom_hessian.py</font> as "
  "<font face='Mono'>slogdet_jac_hessian_jac</font>, with the comment <i>\"For slogdet we know how to compute "
  "the determinant faster.\"</i> There is no entry for det. We printed what JAX hands to folx for an 18 × 18 "
  "matrix (<font face='Mono'>jaxpr_probe.py</font>):")
M("slogdet:  jit[name=slogdet]  →  folx looks up the name 'slogdet', finds its rule, done\n"
  "det:      jit[name=det]      →  no rule for 'det', so folx looks inside and finds\n"
  "            custom_jvp_call  →  det's own derivative rule: a closed box for folx\n"
  "              (inside the box: jit[name=slogdet], exp, mul)")
P("folx finds its rules by the <i>name</i> of the function (<font face='Mono'>eval_pjit</font> in "
  "<font face='Mono'>folx/interpreter.py</font>). Inside det it meets a <font face='Mono'>custom_jvp_call</font>: "
  "JAX gives det a hand-written derivative (the one that stays finite at singular matrices), and folx treats any "
  "such function as a closed box and applies its <b>general fallback</b> (<font face='Mono'>folx/hessian.py</font>, "
  "<font face='Mono'>general_jac_hessian_jac</font>). The irony: inside that box JAX computes the value of det "
  "<i>from a slogdet</i>. folx never sees it, because the box is sealed before folx gets there.")
P("<b>What the fallback does.</b> It flattens all inputs of the box into one long list and, because that list is "
  "longer than the 36 coordinates, it computes the <i>full Hessian of the box with respect to every input</i> "
  "(<font face='Mono'>JHJ_via_hessian</font>). The inputs of our det box are all 4 matrices at once:")
table([["", "6 electrons", "18 electrons"],
       ["matrix entries going into det (4 matrices of N × N)", "4 × 6 × 6 = 144", "4 × 18 × 18 = 1,296"],
       ["second-derivative table per determinant, per walker", "144 × 144 = 20,736", "1,296 × 1,296 = 1,679,616"],
       ["for all 4 determinants, per walker (the code's recipe)", "82,944 numbers", "6.7 million numbers"],
       ["measured scratch memory per walker (det step alone)",
        f"{d64[(6, 'folx_det')]['compiled_scratch_GiB'] * 1024 / 64:.2f} MiB",
        f"{d64[(18, 'folx_det')]['compiled_scratch_GiB'] * 1024 / 64:.0f} MiB"]],
      [7.6, 4.0, 4.8])
P("folx does not know that determinant 1 depends only on matrix 1, so every determinant gets a table over all "
  "1,296 entries. The size grows like N⁴: from 6 to 18 electrons that is 3⁴ = 81 times. We measured 78 times "
  f"({gib(d64[(6, 'folx_det')]['compiled_scratch_GiB'])} → {gib(d64[(18, 'folx_det')]['compiled_scratch_GiB'])} "
  "for 64 walkers). The measured memory (about 5 million 8-byte numbers per walker at 18 electrons) is of the same "
  "size as the recipe predicts. This is also why the problem never showed up before: at 6 electrons a whole "
  f"training iteration with folx + plain det runs fine "
  f"({R['full_6e_attention_folx_det']['seconds_per_iteration']:.3f} s, "
  f"{R['full_6e_attention_folx_det']['peak_GiB']:.2f} GiB).")
P("<b>The test that isolates it.</b> To be sure the determinant is the culprit, we replaced the network with a tiny "
  "made-up function that produces 4 complex N × N matrices, and changed only how the determinants are added and "
  "differentiated (<font face='Mono'>det_alone.py</font>). All four routes give the same kinetic energy "
  f"({d64[(18, 'folx_det')]['mean_kinetic_real']:.4f}, {d64[(18, 'folx_slogdet')]['mean_kinetic_real']:.4f}, "
  f"{d64[(18, 'hessian_det')]['mean_kinetic_real']:.4f}, {d64[(18, 'hessian_slogdet')]['mean_kinetic_real']:.4f} "
  "for 18 electrons, 64 walkers, float32), so they compute the same thing and differ only in cost:")
figure("figC_determinant_alone.png", 15.5,
       "<b>Figure 2.</b> Scratch memory of the determinant step alone, 1024 walkers (compiler's memory analysis). "
       "Only folx + plain det explodes at 18 electrons, and it fails with exactly the error we saw in the full "
       f"network. The ≈{est_fail:.0f} GiB is an <i>estimate</i>: 64 walkers need "
       f"{d64[(18, 'folx_det')]['compiled_scratch_GiB']:.2f} GiB, and memory grows in proportion to the walkers "
       "(at 6 electrons we measured exactly 16 × from 64 to 1024).")
P("<b>The error message.</b> <font face='Mono'>INTERNAL: No valid config found!</font> comes from XLA, the "
  "compiler under JAX, while it searches for a way to run one of the operations on the GPU. It does not say "
  "\"out of memory\". <i>Interpretation:</i> it gave up because no way of running the giant operation fits in the "
  "card; the same calculation at 64 walkers runs, and 1024 walkers would need about "
  f"{est_fail:.0f} GiB.")

# ---------------------------------------------------------------------------------------------- 5 attempts
P("5. Everything we tried, and what happened", h1)
table([["attempt (18 electrons)", "result", "why"],
       ["Stage 4 full Hessian + Stage 3 det (our code as it was)", "works; 0.476 s, 16.9 GiB at 1024; asks for 52 GB "
        "at 4096 and stops", "the Hessian route stores every intermediate result for 36 directions"],
       ["Stage 4 full Hessian + slogdet", "0.432 s, 16.1 GiB: hardly better",
        "the cost of this route is in the network, not in the determinant"],
       ["folx (dense) + Stage 3 det, attention network", "fails at 1024: \"No valid config found\"; asks for 58 GB "
        "at 4096", "no folx rule for det → full Hessian over all 1,296 matrix entries"],
       ["folx (dense) + Stage 3 det, SlaterNet", "runs at 1024 (0.14 s, 9.84 GiB); fails at 4096",
        "same fallback; SlaterNet happened to stay under 24 GiB at 1024 (we did not study why it needs less than "
        "the attention network)"],
       ["folx with sparsity (threshold 0.6) + det", "SlaterNet: 0.119 s, 9.84 GiB, no memory saved. Attention: folx "
        "crashes (IndexError) inside <font face='Mono'>jax.nn.dot_product_attention</font>",
        "a folx limitation with that operation. The authors' own code comment says finite sparsity values crash, "
        "and they use 0. With an exact attention swapped in, sparse folx ran and agreed to 1e-12."],
       ["<b>folx (dense) + slogdet</b>", "<b>0.1296 s, 1.37 GiB at 1024; 0.4636 s, 4.87 GiB at 4096</b>",
        "folx's hand-written slogdet rule (next section)"]],
      [5.0, 5.6, 5.8], shade=[6])

# ---------------------------------------------------------------------------------------------- 6 slogdet
P("6. Why folx + slogdet works so well", h1)
P("For log det A there is a compact formula. Its first derivative in a direction k is tr(A⁻¹ ∂ₖA), and the part "
  "of its second derivative that folx needs is −Σₖ tr(A⁻¹ ∂ₖA A⁻¹ ∂ₖA). Both need only <b>one</b> inverse "
  "A⁻¹ (an 18 × 18 matrix) and the 36 first-derivative matrices ∂ₖA, which folx already carries. folx's rule does "
  "exactly this: invert once, multiply, take a trace. Per determinant and walker that is about 18 × 18 × 36 = "
  "11,664 numbers instead of 1.7 million. The same operations, arranged by someone who knew the formula.")
figure("figD_local_energy_18e.png", 16,
       "<b>Figure 3.</b> The local energy of the real attention network (released sizes, 18 electrons, 1024 walkers, "
       "one GPU). Switching only the determinant step to slogdet makes folx work, and the result matches the "
       "authors' code to within 1 %: 3.7 times faster and 12 times less memory than our old route.")
P("<b>Is it correct?</b> In float64 on the CPU, the folx kinetic energy equals Stage 4's full-Hessian kinetic "
  "energy to 7.0e-13 (real part) and 4.2e-12 (imaginary part) for SlaterNet, and to 1.7e-12 and 3.3e-12 for the "
  "attention network (measure: largest difference divided by the largest value). With JAX's library attention "
  "the attention difference was 1.5e-4 and 1.1e-3, because <font face='Mono'>jax.nn.dot_product_attention</font> "
  "computes its softmax in float32 even in a float64 run. With an exact float64 attention swapped in it drops to "
  "1e-12, so folx itself is exact. Our folx version keeps both the real and the imaginary part of the kinetic "
  "energy; the authors' code keeps only the real part.")

# ---------------------------------------------------------------------------------------------- 7 6e vs 18e
P("7. Why our code was fast at 6 electrons and slow at 18, and the authors' the other way round", h1)
P("<b>An analogy.</b> Think of each iteration as cooking a meal. There is the chopping, which grows with the size "
  "of the meal (the local energy, which grows with the number of electrons), and there are fixed chores such as "
  "setting the table and washing up (moving walkers, the KFAC update, and the bookkeeping between steps). Our old "
  "kitchen had quick chores but a slow way of chopping. The authors' kitchen has a fast chopping machine but slower "
  "chores. For a small meal (6 electrons) the chores decide, and we were faster: 0.087 s against 0.141 s. For a "
  "big meal (18 electrons) the chopping decides, and we were slower: 0.577 s against 0.33 s. Now we have the "
  f"chopping machine and keep our quick chores: {full6:.3f} s at 6 electrons and {full18:.3f} s at 18.")
figure("figE_iteration_split.png", 15,
       "<b>Figure 4.</b> Seconds per training iteration, attention network, 1024 walkers, one GPU. Dark: the local "
       "energy, measured alone. Light: everything else (moves, KFAC, the training loop) = whole iteration minus "
       "local energy. The authors' 6-electron total (0.141 s) is their measured training speed, 7.1 iterations per "
       "second.")
P("<b>The numbers behind it.</b> From 6 to 18 electrons our Hessian local energy grew 9.7 times (0.049 → 0.476 s), "
  "and the authors' folx local energy grew 7.7 times (0.017 → 0.1305 s). The rest of the iteration grew much less: "
  "ours from about 0.04 to 0.10 s, the authors' from about 0.12 to 0.20 s.")
P("<b>Where the authors' extra fixed time goes</b> (<i>interpretation, not profiled</i>). At 6 electrons their GPU "
  "was busy only 14–28 % of the time, so most of each iteration the GPU was waiting. Their compiled step contains "
  "the same three pieces as ours (30 Metropolis moves, folx local energy, kfac_jax update) for a network with the "
  "same 103,232 parameters. After every step their training loop (<font face='Mono'>periodicwave/train.py</font>, "
  "lines 676–737) runs several small uncompiled JAX operations (statistics, a square root, a "
  "<font face='Mono'>jax.lax.cond</font> for the move width) and copies the energy and statistics to the CPU "
  "(<font face='Mono'>np.asarray</font>, every iteration because <font face='Mono'>stats_frequency = 1</font>). "
  "Each such operation makes Python wait for the GPU and the GPU wait for Python. <i>Estimate:</i> if their moves "
  "and KFAC update cost what ours do, about 0.08 s (6 electrons) to 0.1 s (18 electrons) of each iteration is "
  "spent this way. We did not time these operations one by one, so we cannot say how that time splits between "
  "them. Their NaN check would copy all parameters "
  "to the CPU, but it is switched off (<font face='Mono'>check_nan: False</font>).")

# ---------------------------------------------------------------------------------------------- 8 price
P("8. The price of slogdet", h1)
P("Stage 3's docstring explains why we did not use slogdet: <i>\"The derivative of slogdet is d log det A = "
  "tr(A^-1 dA), which is infinite when A is singular.\"</i> When one of the 4 determinants is exactly zero while "
  "the others are not, Ψ itself is perfectly smooth, but the slogdet route multiplies that infinity by the "
  "vanishing determinant: 0 × ∞ = NaN. Plain det differentiates through the adjugate matrix and stays finite.")
figure("figB_singular_derivative.png", 11,
       "<b>Figure 5.</b> For A = [[1, 1], [1, 1+ε]], det A = ε. The derivative of det A is 1 for every ε, but the "
       "derivative of log|det A| is 1/ε, which grows without limit as A becomes singular.")
P("<b>A concrete test</b> (<font face='Mono'>singular_example.py</font>, run with our real Stage 3 code). "
  "Ψ(ε) = det A(ε) + det B with B = diag(2, 1), so Ψ = ε + 2 and the exact answer is d log|Ψ|/dε = 1/(ε + 2):")
rows = [["ε", "precision", "exact", "Stage 3 (det)", "slogdet", "exact 2nd deriv.", "det 2nd", "slogdet 2nd"]]
for s in S:
    if s["e"] in (0.1, 1e-3, 1e-7, 0.0) or (s["precision"] == "float32" and abs(s["e"] - 1e-3) < 1e-9):
        rows.append([f"{s['e']:.0e}" if s["e"] else "0 (singular)", s["precision"], f"{s['exact']:.7f}",
                     f"{s['det']:.7f}", "NaN" if s["slogdet"] != s["slogdet"] else f"{s['slogdet']:.7f}",
                     f"{s['exact_second']:.6f}", f"{s['det_second']:.6f}",
                     "NaN" if s["slogdet_second"] != s["slogdet_second"] else f"{s['slogdet_second']:.6f}"])
table(rows, [1.9, 1.8, 1.8, 1.9, 1.8, 2.2, 1.9, 2.1])
P("So: at an exactly singular matrix, slogdet gives NaN and det gives the right answer. Close to singular, both "
  "work in float64; in float32 the slogdet second derivative loses digits a little earlier "
  "(−0.249725 instead of −0.249750 at ε = 10⁻³).")
P("<b>Why we accept it anyway.</b> Walkers are random real numbers, so landing exactly on a configuration where "
  "one determinant is exactly zero essentially never happens. The authors, FermiNet and Psiformer all add their "
  "determinants this way, with the same largest-value shift, and train normally. Our training loop checks every "
  "energy for NaN, but, as the review you pasted pointed out, it does not yet check the <i>updated</i> parameters "
  "before saving a checkpoint; that fix comes together with these changes. <b>What we will do:</b> keep Stage 3's det version "
  "unchanged, add the slogdet version next to it, test that both give the same log|Ψ| and the same local energy "
  "on ordinary configurations, and add a test that documents the NaN at an exactly singular matrix.")

# ---------------------------------------------------------------------------------------------- 9 consequences
P("9. What this means for the 18-electron training", h1)
P("Wall-time <i>estimates</i> for one run on one GPU, from the measured seconds per iteration (5 iterations from "
  "random walkers, released network sizes; a real run adds checkpoints and logging, and paper Table II uses a "
  "slightly smaller attention network: 3 layers, 6 heads):")
table([["network, settings", "before (Hessian + det)", "after (folx + slogdet)", "authors' code, 1 GPU"],
       ["attention, 1024 walkers, 200,000 iterations", "0.577 s/it → 32 h", f"{full18:.3f} s/it → "
        f"{full18 * 200000 / 3600:.1f} h", "0.33 s/it → 18 h"],
       ["attention, 4096 walkers, 150,000 iterations (Table II)", "does not fit on the GPU",
        f"{full18_4096:.2f} s/it → {full18_4096 * 150000 / 3600:.0f} h", "1.06 s/it → 44 h"],
       ["SlaterNet, 1024 walkers, 200,000 iterations", "0.125 s/it → 6.9 h",
        f"{sn18:.3f} s/it → {sn18 * 200000 / 3600:.1f} h", "not measured"],
       ["SlaterNet, 4096 walkers, 150,000 iterations", "does not fit on the GPU",
        f"{sn18_4096:.3f} s/it → {sn18_4096 * 150000 / 3600:.1f} h", "not measured"]],
      [5.6, 3.4, 3.8, 3.6])
P("All of these are for <b>one GPU</b>. The authors' code on two GPUs took 0.615 s per iteration at 4096 walkers. "
  "Our code does not yet split a batch across two GPUs; adding that is the next step after the folx and slogdet "
  "changes, and its speed is not measured yet. Running two different jobs, one per GPU, works already.")

# ---------------------------------------------------------------------------------------------- 10 methods
P("10. How the measurements were made", h1)
P("Hardware: khabab-server, NVIDIA RTX 3090 (24 GiB), driver 595.99.02. Every measurement in this study ran on "
  "<b>GPU 0 only</b>; GPU 1 stayed idle (its log shows 0 % use and 4 MiB). Software: Python 3.13, JAX 0.8.0, folx "
  "0.2.19, kfac_jax 0.0.7, float32 with <font face='Mono'>jax_default_matmul_precision = float32</font>.")
P("Pieces (<font face='Mono'>pieces.py</font>): each piece is a compiled function, called 6 times; the first call "
  "includes compilation and is dropped; the time is the mean of calls 2–6; peak memory is JAX's "
  "<font face='Mono'>peak_bytes_in_use</font> on GPU 0 for the whole process. Determinant step alone "
  "(<font face='Mono'>det_alone.py</font>): the same, plus the compiler's memory analysis "
  "(<font face='Mono'>temp_size_in_bytes</font>) before running. Full iteration "
  "(<font face='Mono'>full_iteration.py</font>): moves + local energy + KFAC, 6 iterations from random walkers, "
  "mean of the last 5. Authors' full iteration (earlier smoke tests): total time of 240 iterations minus total "
  "time of 40, divided by 200, memory with <font face='Mono'>XLA_PYTHON_CLIENT_PREALLOCATE=false</font>. "
  "The 18-electron piece numbers (pieces18.py) were measured a few hours before this study, with the same method "
  "on the same GPU; the 6-electron pieces, the determinant-step test and the new full iterations are from this "
  "study. GPU usage was logged every 2 s for both GPUs (<font face='Mono'>gpu_usage.csv</font>): "
  "GPU 0 peaked at 18.4 GiB, 100 % use, 56 °C.")
P("Files (in <font face='Mono'>learn/stage08_own_code/speed_study/</font>): this report and its builder, the "
  "measurement scripts (pieces.py, det_alone.py, full_iteration.py, singular_example.py, jaxpr_probe.py, "
  "run_all.sh), <font face='Mono'>speed_results.jsonl</font> (one line per run), the logs of every run, "
  "the GPU usage log, and the earlier 18-electron logs.")

# ---------------------------------------------------------------------------------------------- appendix
start = len(story)
P("Appendix A. How to repeat the study", h1)
M("cd ~/moire-nnvmc/learn/stage08_own_code/speed_study\n"
  "./run_all.sh          # about 16 minutes, GPU 0 only\n"
  "cat progress.txt      # which run is going\n"
  "uv run --no-project --with matplotlib --with numpy python make_figures.py\n"
  "uv run --no-project --with matplotlib --with reportlab python build_report.py")
P("The project environment has no matplotlib or reportlab; <font face='Mono'>uv run --no-project --with ...</font> "
  "adds them for one command without changing the environment, as in the Stage 8 analysis folder.", small)
story[start:] = [KeepTogether(story[start:])]
P("Appendix B. Every run of today's study", h1)
rows = [["run", "seconds", "peak memory", "compiled scratch", "note"]]
for name, r in R.items():
    t = r.get("seconds_per_iteration", r.get("seconds_per_call"))
    rows.append([name.replace("_", " "), "—" if t is None else f"{t:g}",
                 "—" if "peak_GiB" not in r else f"{r['peak_GiB']:g} GiB",
                 "—" if "compiled_scratch_GiB" not in r else f"{r['compiled_scratch_GiB']:g} GiB",
                 r.get("error", "") or (f"kinetic {r['mean_kinetic_real']:.5f}" if "mean_kinetic_real" in r else "")])
table(rows, [6.4, 1.6, 2.2, 2.4, 3.8])
P("Seconds: per call (pieces, determinant step) or per iteration (full). All runs: GPU 0, float32, ε = 5, "
  "1024 walkers unless the name says b64 or b4096.", small)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.2 * cm, "Stage 8 speed study: why 18 electrons were slow")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(os.path.join(HERE, "speed_study_report.pdf"), pagesize=A4, leftMargin=2.2 * cm,
                        rightMargin=2.2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Why 18 electrons were slow, and why folx + slogdet fixes it", author="moire-nnvmc")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("written", os.path.join(HERE, "speed_study_report.pdf"))
