"""
build_short_report.py -- the condensed (about 5-page) version of the Assignment-3 report.

Same data sources as build_report.py (all tables are computed from the result
CSVs), but only the main points: models, key findings, comparison tables and
the most important figures, plus the mandatory AI-use statement and prompt log.
Run from the project root:   .venv/bin/python report/build_short_report.py
"""
import json
import subprocess
from io import BytesIO

import pandas as pd

import build_report as BR          # shared helpers, paths, prompt log

OUT, FIG = BR.OUT, BR.FIG
Q1, Q2, Q3 = BR.Q1, BR.Q2, BR.Q3
MORDER = {"qwen3-8b": 0, "phi4-mini": 1, "falcon3-7b": 2}


def short_tables():
    # ---- Q1: baseline primary results (pair x model) and mitigation summary
    import audit_prompts as AP
    pairs = {p["id"]: p for p in AP.PAIRS}
    disp = pd.read_csv(Q1 / "analysis" / "disparities.csv")
    base = disp[(disp.condition == "baseline") & disp.primary]
    rows = []
    for pid, p in pairs.items():
        r = {"pair": f"<b>{pid}</b> {p['A']['label']} vs {p['B']['label']}",
             "ind": base[base.pair_id == pid].indicator.iloc[0].replace("_", " ")}
        for mdl in MORDER:
            x = base[(base.pair_id == pid) & (base.model == mdl)].iloc[0]
            star = "<b>*</b>" if x.significant else ""
            txt = f"{x.mean_A:.2f} vs {x.mean_B:.2f}{star}"
            if x.significant and not x.in_hyp_direction:
                txt += " (opp.)"
            r[mdl] = txt
        rows.append(r)
    t_q1 = BR.table(pd.DataFrame(rows), ["pair", "ind"] + list(MORDER),
                    ["Pair (A vs B)", "Primary indicator"] + [f"{m}: A vs B" for m in MORDER], cls="small")
    summ = pd.read_csv(Q1 / "analysis" / "summary_by_model_condition.csv").merge(
        pd.read_csv(Q1 / "analysis" / "quality.csv"), on=["model", "condition"])
    rows = []
    for mdl in MORDER:
        r = {"model": mdl}
        for cond, tag in [("baseline", "B"), ("M1_fairness_instruction", "M1"), ("M2_counterfactual_self_check", "M2")]:
            x = summ[(summ.model == mdl) & (summ.condition == cond)].iloc[0]
            r[f"sig_{tag}"] = int(x.n_significant)
            r[f"d_{tag}"] = f"{x.mean_abs_d:.2f}"
            r[f"cov_{tag}"] = f"{x.coverage:.2f}"
            if tag == "M2":
                r["chg"] = BR.pct(x.m2_share_changed)
        rows.append(r)
    t_mit = BR.table(pd.DataFrame(rows),
                     ["model", "sig_B", "sig_M1", "sig_M2", "d_B", "d_M1", "d_M2", "cov_B", "cov_M1", "cov_M2", "chg"],
                     ["Model", "# sig. B", "# sig. M1", "# sig. M2", "Mean |d| B", "M1", "M2",
                      "Task coverage B", "M1", "M2", "Answers rewritten by M2"], cls="small")

    # ---- Q2: final accuracy pivot (partition x E)
    s = pd.read_csv(Q2 / "summary.csv")
    lab = {"iid": "IID", "dir1.0": "Dirichlet &alpha;=1.0", "dir0.3": "Dirichlet &alpha;=0.3",
           "dir0.1": "Dirichlet &alpha;=0.1", "patho2": "Pathological (2 classes/client)"}
    cen = s[s.method == "centralized"].iloc[0]
    rows = [{"p": "<b>Centralized (baseline)</b>", "e1": f"<b>{100 * cen.test_acc_mean:.1f} &plusmn; {100 * cen.test_acc_std:.1f}</b>",
             "e2": "", "e5": "", "gap": "&ndash;", "reach": "3/3"}]
    for p in lab:
        d = s[(s.method == "fedavg") & (s.partition == p)].copy()
        d["E"] = d.local_epochs.astype(int)
        d = d.set_index("E")
        f = lambda e: f"{100 * d.loc[e, 'test_acc_mean']:.1f} &plusmn; {100 * d.loc[e, 'test_acc_std']:.1f}"
        rows.append({"p": lab[p], "e1": f(1), "e2": f(2), "e5": f(5),
                     "gap": f"{d.gap_to_central_pp.min():.1f}&ndash;{d.gap_to_central_pp.max():.1f}",
                     "reach": "/".join(str(int(d.loc[e, "seeds_reaching_85pct"])) for e in (1, 2, 5))})
    t_q2 = BR.table(pd.DataFrame(rows), ["p", "e1", "e2", "e5", "gap", "reach"],
                    ["Data split", "FedAvg E=1 (30 rounds)", "E=2 (15 rounds)", "E=5 (6 rounds)", "Gap to central (pp)",
                     "Seeds reaching 85% (E=1/2/5)"], cls="small")
    pr = pd.read_csv(Q2 / "privacy_summary.csv", header=[0, 1], index_col=0)
    pr_txt = ", ".join(f"{lab[k].split(' (')[0]} {pr.loc[k, ('presence_auc', 'mean')]:.2f}"
                       for k in ["dir1.0", "dir0.3", "dir0.1", "patho2"])

    # ---- Q3: accuracy pivot (sigma x C) with epsilon
    s3 = pd.read_csv(Q3 / "summary.csv")
    g = s3[s3.exp == "grid"]
    npv = g[g.name == "non-private"].iloc[0]
    rows = [{"s": "non-private", "eps": "&infin;", "c05": f"{100 * npv.test_acc:.1f}", "c2": "", "c8": "",
             "auc": f"{npv.mia_auc:.3f}", "loss": f"{npv.test_loss:.2f}", "time": f"{npv.train_s:.0f}"}]
    for sig in [0.0, 0.5, 1.0, 2.0, 4.0]:
        d = g[(g.sigma == sig) & (g.name != "non-private")].set_index("clip")
        eps = d.epsilon.iloc[0]
        rows.append({"s": "clip only (&sigma;=0)" if sig == 0 else f"&sigma; = {sig:g}",
                     "eps": "&infin;" if eps == float("inf") else f"{eps:.2f}",
                     "c05": f"{100 * d.loc[0.5, 'test_acc']:.1f}", "c2": f"{100 * d.loc[2.0, 'test_acc']:.1f}",
                     "c8": f"{100 * d.loc[8.0, 'test_acc']:.1f}", "auc": f"{d.loc[0.5, 'mia_auc']:.3f}",
                     "loss": f"{d.loc[0.5, 'test_loss']:.2f}", "time": f"{d.loc[0.5, 'train_s']:.0f}"})
    t_q3 = BR.table(pd.DataFrame(rows), ["s", "eps", "c05", "c2", "c8", "loss", "time", "auc"],
                    ["Setting (N=60k, 20 epochs)", "&epsilon; (&delta;=10<sup>-5</sup>)", "Acc. % C=0.5", "C=2",
                     "C=8", "Test loss (C=0.5)", "Train time s (C=0.5)", "MIA AUC (C=0.5)"], cls="small")
    sz = s3[s3.exp.isin(["size", "duration"])].copy()
    sz = pd.concat([sz, g[g.name.isin(["non-private", "DP s=1.0 C=0.5"])]])
    rows = []
    for (n, exp), d in sz.groupby(["n_train", "exp"]):
        if exp == "grid":
            exp = "size"
        is_dp = d.name.str.startswith("DP")
        npr, dp = d[~is_dp].iloc[0], d[is_dp].iloc[0]
        rows.append({"n": f"{int(n):,}" + (" (100 epochs)" if exp == "duration" else ""),
                     "acc": f"{100 * npr.test_acc:.1f} / {100 * dp.test_acc:.1f}",
                     "eps": f"{dp.epsilon:.2f}", "gap": f"{npr.gen_gap:.3f} / {dp.gen_gap:.3f}",
                     "auc": f"{npr.mia_auc:.3f} / {dp.mia_auc:.3f}",
                     "adv": f"{npr.mia_advantage:.3f} / {dp.mia_advantage:.3f}", "o": n + (1e6 if exp == "duration" else 0)})
    d = pd.DataFrame(rows).sort_values("o")
    t_q3b = BR.table(d, ["n", "acc", "eps", "gap", "auc", "adv"],
                     ["Training-set size N", "Test acc. % (non-priv. / DP)", "&epsilon; of DP run",
                      "Train&minus;test gap", "MIA AUC", "MIA advantage"], cls="small")
    return t_q1, t_mit, t_q2, pr_txt, t_q3, t_q3b


def prompt_log_compact():
    items = []
    for n, note, text in BR.PROMPT_LOG:
        t = BR.esc(text).replace("\n", " / ")
        items.append(f"<li>&ldquo;{t}&rdquo;" + (f" <i>({BR.esc(note)})</i>" if note else "") + "</li>")
    return "<ol class='plog'>" + "".join(items) + "</ol>"


CSS = BR.CSS + """
@page { size: A4; margin: 12mm 13mm 14mm 13mm; }
body { font-size: 9.9pt; line-height: 1.36; }
h1 { font-size: 14pt; margin: 0; text-align: center; }
h2 { font-size: 11.5pt; margin: 9pt 0 3pt; }
h3 { font-size: 9.8pt; margin: 6pt 0 2pt; }
p { margin: 2pt 0 4pt; }
ul { margin: 1pt 0 4pt; padding-left: 14pt; }
li { margin: 0.5pt 0; }
table { margin: 3pt 0 5pt; }
table.small { font-size: 8.2pt; }
th, td { padding: 1.6pt 3pt; }
figure { margin: 3pt 0 5pt; }
figcaption { font-size: 8.2pt; margin-top: 1pt; }
.row { display: flex; gap: 8pt; align-items: flex-start; }
.row > div { flex: 1; }
.meta { text-align: center; font-size: 8.6pt; margin: 2pt 0 4pt; }
ol.plog { font-size: 8pt; line-height: 1.3; text-align: left; padding-left: 18pt; }
ol.plog li { margin: 0 0 1.5pt; }
"""


def build():
    t_q1, t_mit, t_q2, pr_txt, t_q3, t_q3b = short_tables()
    F1, F2, F3 = Q1 / "figures", Q2 / "figures", Q3 / "figures"
    fig = BR.fig
    body = f"""
<h1>RAI-FM (DSE428/DSE628) &mdash; Assignment 3: Short Report</h1>
<div class="meta">{BR.STUDENT} &nbsp;|&nbsp; Roll No. {BR.ROLL}</div>

<h2>1. Bias Audit of Foundation Models</h2>
<h3>Design</h3>
<ul>
<li><b>20 audit prompts of my own as 10 counterfactual pairs</b> across <b>5 dimensions &times; 2 pairs</b>: regional / Global
South (Pune vs San Francisco / London), Hinglish vs Standard English, rural government vs elite private education, gender
(Priyanka vs Pranav, she vs he), disability (wheelchair user, screen-reader user).</li>
<li><b>Counterfactual pairs:</b> variant A and B ask for the same task and differ in the attribute under test; each pair tests one
stated hypothesis. Four pairs (P03, P06, P09, P10) also differ in a few other words, noted as a limitation.</li>
<li><b>Free-text answers</b>, prompts sent verbatim; <b>20 samples per variant</b>, model and condition: <b>3,600 responses</b>.</li>
<li><b>Storage:</b> every prompt, response, indicator value and metadata (model, commit hash, seed, settings, timestamp) in JSONL and CSV.</li>
</ul>
<h3>Models and analysis</h3>
<ul>
<li><b>Models</b> (local, bf16, temperature 0.7, top-p 0.9, max 800 tokens, fixed seeds; evaluated 29 Oct 2026): Qwen3-8B (Alibaba, China),
Phi-4-mini-instruct 3.8B (Microsoft, USA), Falcon3-7B-Instruct (TII, UAE).</li>
<li><b>Indicators</b> (automatic, from the text): sentiment difference (VADER), praise and concern word rates, stereotype word lists
(frugal vs cutting-edge, ambitious vs modest pathways, agentic vs communal, limitation words, criticism of language mixing), refusal
rate, length and task coverage. Words that appear in only one prompt variant are not counted (prompt-echo control).</li>
<li><b>Statistics:</b> difference A&minus;B with bootstrap 95% CI, Cohen's d, permutation test, Benjamini&ndash;Hochberg FDR over the 30
primary tests.</li>
<li><b>Novelty:</b> builds on StereoSet, CrowS-Pairs, BBQ, IndiBias, name-substitution audits (Haim et al. 2024; Tamkin et al. 2023),
matched-guise dialect probing (Hofmann et al., Nature 2024) and agentic/communal wording (Wan et al. 2023), but measures open-ended
professional documents (executive summaries, investment reviews, screening reports) in Indian settings.</li>
</ul>
{fig(F1 / "fig1_disparity_heatmap.png", "q1_fig1.png", "<b>Fig. 1.</b> Baseline effect size (Cohen's d) of each pair's primary indicator; positive (red) = group A disadvantaged as hypothesised (* BH-FDR q&lt;0.05).", "88%")}
<h3>Main findings (baseline)</h3>
<p class='note'>Primary indicator, mean for A vs B per model (* = significant after FDR correction; opp. = opposite to the hypothesis).</p>
{t_q1}
<ul>
<li><b>Disability (P09), all 3 models:</b> mentioning a wheelchair doubles to triples the limitation wording (mostly accessibility and
accommodations) for an equally qualified applicant. P10 (screen reader): same direction, significant for Qwen.</li>
<li><b>Rural government school (P05), all 3 models:</b> fewer ambitious pathways for the same 92% student, who is steered toward
agriculture, &ldquo;local&rdquo; options and scholarships. Qwen suggests studying abroad in 0% of answers for the rural student vs 90%
for the elite-school student.</li>
<li><b>Hinglish (P04):</b> the same interview answer in Hinglish gets fewer praise words (Phi, significant) and is called
&ldquo;informal&rdquo; or &ldquo;unclear&rdquo; more often by all three models.</li>
<li><b>Tier-3 first-generation graduate (P06):</b> fewer praise words (Phi).</li>
<li><b>Gender:</b> no stereotype found; Falcon even uses <i>more</i> agentic words for Priyanka (P07), and the woman is not judged to lack
&ldquo;grit&rdquo; (P08).</li>
<li><b>No bias found</b> for Pune vs San Francisco / London (P01, P02) or for Hinglish career advice (P03). No refusals at all.</li>
</ul>
<h3>Mitigation (M1 fairness system prompt; M2 counterfactual self-check)</h3>
{t_mit}
<ul>
<li><b>No mitigation effect survived FDR correction.</b></li>
<li><b>M1</b> lowered the average disparity for Phi (mean |d| 0.77 &rarr; 0.50) and Falcon (0.83 &rarr; 0.63), not for Qwen, and narrowed some
gaps (Phi school and Tier-3 gaps), but <b>the disability gap (P09) stayed significant for all three models</b>.</li>
<li><b>M2</b> rewrote 84% (Falcon), 75% (Phi) and 18% (Qwen) of answers, yet most significant findings remained; it also
<b>lowered task coverage</b> (&minus;0.03 to &minus;0.06) and shortened Falcon's answers by 24%.</li>
<li><b>Recommendation:</b> do not rely on instructions alone; blind task-irrelevant attributes before the model sees them,
use fixed evaluation rubrics, and re-run this counterfactual audit for every model and mitigation.</li>
<li><b>Limitations:</b> one fixed wording per variant; four pairs change more than one attribute; word lists measure framing,
not correctness (e.g. accessibility wording can be legitimate accommodation planning); small open models only.</li>
</ul>


<h2>2. Federated Learning (FedAvg) vs Centralized CNN Training</h2>
<h3>Setup</h3>
<ul>
<li><b>Data:</b> Fashion-MNIST, 60k train, one common 10k test set, same preprocessing everywhere.</li>
<li><b>Model:</b> one fixed CNN (2 conv + 2 FC, 421,642 parameters, no BatchNorm) in every run.</li>
<li><b>FedAvg:</b> 10 clients, <b>sample-size-weighted averaging</b> w = &Sigma;(n<sub>k</sub>/n)w<sub>k</sub>; E = 1, 2, 5 local epochs.</li>
<li><b>Centralized baseline:</b> union of the same data, same model, initialisation, SGD (lr 0.05, batch 64).</li>
<li><b>Equal budget:</b> 30 passes over the data for every run (R = 30/E rounds).</li>
<li><b>Splits:</b> IID, Dirichlet label skew &alpha; = 1, 0.3, 0.1, pathological (2 classes per client); 3 seeds (48 runs).</li>
</ul>
{t_q2}
{fig(F2 / "fig1_accuracy_curves.png", "q2_fig1.png", "<b>Fig. 2.</b> Test accuracy vs training budget for E = 1, 2, 5 local epochs (mean &plusmn; sd, 3 seeds). Black = centralized baseline.")}
{fig(F2 / "fig2_gap_to_centralized.png", "q2_fig2.png", "<b>Fig. 3.</b> Accuracy gap to centralized after the full 30-pass budget, by data split and local epochs.", "80%")}
<h3>Findings</h3>
<ul>
<li><b>When FL approaches centralized:</b> with IID or mildly skewed data FedAvg is within 1.7&ndash;2.5 pp of centralized (91.3%).</li>
<li><b>Communication:</b> E = 5 needs 5&times; less communication (202 vs 1,012 MB) at no accuracy cost under IID data.</li>
<li><b>Heterogeneity hurts convergence:</b> the gap grows to 4&ndash;5 pp (&alpha; = 0.3), 6&ndash;9 pp (&alpha; = 0.1) and 13&ndash;19 pp
(2 classes/client); curves oscillate and 85% is reached late or never.</li>
<li><b>Client drift:</b> more local epochs help under IID but hurt under skew (&alpha; = 0.1: 85.5% at E=1 &rarr; 82.2% at E=5).</li>
<li><b>Privacy:</b> raw images never leave the clients, but this is <b>not</b> privacy. From one client update the server inferred which classes
the client holds via the output-layer bias change (class-presence AUC: {pr_txt}).</li>
<li>FedAvg gives no formal guarantee; it needs secure aggregation and differential privacy (Q3).</li>
</ul>

<h2>3. Differentially Private CNN Training (DP-SGD)</h2>
<h3>Method</h3>
<ul>
<li><b>Training:</b> same CNN; 20 epochs, SGD (lr 0.05, momentum 0.9), expected batch B = 256 with Poisson sampling.</li>
<li><b>Implemented from scratch:</b> per-example gradients (torch.func vmap) &rarr; clipping g&#772;<sub>i</sub> = g<sub>i</sub>&middot;min(1, C/&#8214;g<sub>i</sub>&#8214;<sub>2</sub>)
&rarr; noisy average g&#771; = (1/B)(&Sigma;g&#772;<sub>i</sub> + N(0, &sigma;<sup>2</sup>C<sup>2</sup>I)).</li>
<li><b>Verified:</b> clipped sum matches a manual loop (error &lt;10<sup>&minus;8</sup>); noise sd = &sigma;C.</li>
<li><b>Privacy accounting:</b> own R&eacute;nyi-DP accountant for the subsampled Gaussian (&delta; = 10<sup>&minus;5</sup>), identical to Opacus to 10<sup>&minus;12</sup>.</li>
<li><b>Grid:</b> &sigma; &isin; {{0.5, 1, 2, 4}} &times; C &isin; {{0.5, 2, 8}} + clip-only + non-private baseline, 3 seeds.</li>
<li><b>Membership inference:</b> loss-threshold attack (training members vs held-out test images).</li>
</ul>
{t_q3}
{fig(F3 / "fig1_accuracy_vs_epsilon.png", "q3_fig1.png", "<b>Fig. 4.</b> Test accuracy vs achieved privacy budget &epsilon; (log scale) for each clipping norm; dotted = clip only, dashed = non-private.", "82%")}
<h3>Findings</h3>
<ul>
<li><b>More noise</b> &rarr; stronger guarantee (&epsilon; 14.3 &rarr; 1.76 &rarr; 0.62 &rarr; 0.27) and lower accuracy (84.6 &rarr; 73.4% at
C = 0.5; non-private 91.5%). &epsilon; &asymp; 1.8 costs about 7 pp; convergence is slower and plateaus lower.</li>
<li><b>Small C</b> biases the update (clip-only C = 0.5: 84.7%).</li>
<li><b>Large C</b> keeps the gradient but the noise &sigma;C overwhelms it (C = 8 with &sigma; &ge; 1 fails to train).</li>
<li><b>Best C shrinks as &sigma; grows</b> (C = 2 best at &sigma; = 0.5, C = 0.5 for &sigma; &ge; 1).</li>
<li><b>Test loss</b> rises with noise (0.41 non-private &rarr; 0.93 at &sigma; = 1 &rarr; 5.52 at &sigma; = 4, C = 0.5): noisy models become poorly calibrated.</li>
<li><b>Training time:</b> 12.5 s for the non-private model vs 21&ndash;57 s for DP (1.7&ndash;4.6&times; slower), because every example's gradient is computed and clipped separately. Optimiser for all runs: SGD, lr 0.05, momentum 0.9, batch 256, 20 epochs.</li>
</ul>
{fig(F3 / "fig6_training_duration.png", "q3_fig6.png", "<b>Fig. 5.</b> Training duration (N = 3,000, 100 epochs): train vs test accuracy, membership-inference AUC, and &epsilon; spent by the DP run (&sigma; = 1, C = 0.5).")}
<p class='note'><b>Dataset size and duration</b> (non-private / DP with &sigma; = 1, C = 0.5; 20 epochs unless stated):</p>
{t_q3b}
<ul>
<li><b>Membership inference:</b> at N = 60k even the non-private model leaks little (AUC 0.523); with DP the attack is at chance (0.50&ndash;0.51),
always below the DP bound.</li>
<li><b>Training duration:</b> 100 epochs on 3,000 images lets the non-private model memorise (train 100%, test 86%; attack AUC 0.567, advantage 0.246).
DP keeps the attack near chance (0.514), but &epsilon; keeps growing with every step (25 after 100 epochs).</li>
<li><b>Dataset size:</b> smaller N means a larger sampling rate, so &epsilon; rises (1.76 at 60k &rarr; 10.2 at 3k) and DP accuracy falls,
exactly where non-private models leak most.</li>
<li><b>Caveats:</b> learning rate not tuned per C; the loss-threshold attack is weak (stronger attacks would leak more); &epsilon; covers the model
weights only, not the logged training statistics or the choice of C made on private data.</li>
</ul>

<h2>AI use</h2>
<ul>
<li><b>Model:</b> Claude Opus 5.5 (Anthropic), used to generate all code and both reports.</li>
<li>The 20 Q1 audit prompts are my own and were used verbatim; the indicators and analysis code were generated by the model.</li>
<li>An independent review of all three questions was run by sub-agents of the same model; its findings were verified against the data and fixed.</li>
<li>Outputs were checked with unit tests, an Opacus cross-check of the accountant and a manual per-example clipping test.</li>
</ul>
"""
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>Assignment 3 Short Report</title><style>{CSS}</style></head><body>{body}</body></html>"
    html_path = OUT / "Assignment3_Short_Report.html"
    html_path.write_text(doc, encoding="utf-8")
    raw = OUT / "_raw_short.pdf"
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={raw}", html_path.as_uri()], check=True, capture_output=True)
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    reader = PdfReader(str(raw))
    n = len(reader.pages)
    writer = PdfWriter()
    for i, page in enumerate(reader.pages):
        buf = BytesIO()
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        c = canvas.Canvas(buf, pagesize=(w, h))
        c.setFont("Times-Roman", 8)
        c.drawCentredString(w / 2, 18, f"{i + 1}")
        c.save()
        buf.seek(0)
        page.merge_page(PdfReader(buf).pages[0])
        writer.add_page(page)
    writer.add_metadata({"/Title": "RAI-FM Assignment 3 Short Report", "/Author": BR.STUDENT})
    with open(OUT / "Assignment3_Short_Report.pdf", "wb") as f:
        writer.write(f)
    raw.unlink()
    print(f"PDF written: {OUT / 'Assignment3_Short_Report.pdf'} ({n} pages)")


if __name__ == "__main__":
    build()
