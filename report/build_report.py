"""
build_report.py -- builds the Assignment-3 report (HTML -> PDF via headless Chrome).

All numeric tables are generated directly from the result CSVs written by the
three experiment scripts, so the report cannot drift from the data. Figures are
the PNGs produced by those scripts. Run from the project root:

    .venv/bin/python report/build_report.py
"""
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report"
FIG = OUT / "figures"
sys.path.insert(0, str(ROOT / "q1_bias_audit"))
import audit_prompts as AP  # noqa: E402

Q1 = ROOT / "q1_bias_audit" / "results"
Q2 = ROOT / "q2_federated" / "results"
Q3 = ROOT / "q3_dpsgd" / "results"

STUDENT = "Payal Verma"
ROLL = "2611004"


# ---------------------------------------------------------------- helpers
def esc(s):
    return html.escape(str(s))


def table(df, cols, headers=None, fmt=None, cls=""):
    """Render a DataFrame subset as an HTML table."""
    fmt = fmt or {}
    headers = headers or cols
    out = [f'<table class="{cls}"><thead><tr>' + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                v = fmt[c](v)
            elif isinstance(v, float):
                v = f"{v:.3f}"
            # cells produced by our own formatters may contain HTML entities/tags -> keep as is
            raw = c in fmt or (isinstance(v, str) and ("&" in v or "<" in v))
            cells.append(f"<td>{v if raw else esc(v)}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def fig(src_path, name, caption, width="100%"):
    FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, FIG / name)
    return (f'<figure><img src="figures/{name}" style="width:{width}"/>'
            f"<figcaption>{caption}</figcaption></figure>")


def pct(x):
    return f"{100 * x:.1f}%"


def pval(x):
    return "&lt;0.001" if x < 0.001 else f"{x:.3f}"


def gm(s):
    d = json.loads(s)
    return "; ".join(f"{k.replace('_', ' ')}: {v:g}" for k, v in d.items())


# ---------------------------------------------------------------- Q1 tables
MORDER = {"qwen3-8b": 0, "phi4-mini": 1, "falcon3-7b": 2}
CLABEL = {"baseline": "Baseline", "M1_fairness_instruction": "M1 fairness prompt",
          "M2_counterfactual_self_check": "M2 self-check"}


def verdict(r):
    if r.significant:
        return "<b>significant, as hypothesised</b>" if r.in_hyp_direction else "<b>significant, opposite</b>"
    return "trend (p&lt;0.05)" if r.p_perm < 0.05 else "n.s."


def q1_tables():
    """All Q1 tables, built from the analysis CSVs written by bias_audit.py."""
    disp = pd.read_csv(Q1 / "analysis" / "disparities.csv")
    pairs = {p["id"]: p for p in AP.PAIRS}
    disp["o"] = disp.pair_id.str[1:].astype(int) * 10 + disp.model.map(MORDER)
    disp["comparison"] = disp.pair_id.map(lambda x: f"{pairs[x]['A']['label']} vs {pairs[x]['B']['label']}")
    disp["diffci"] = disp.apply(lambda r: f"{r.diff_A_minus_B:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
    disp["verdict"] = disp.apply(verdict, axis=1)
    fmt = {"mean_A": lambda v: f"{v:.3f}", "mean_B": lambda v: f"{v:.3f}", "cohens_d": lambda v: f"{v:+.2f}",
           "p_perm": pval, "q_bh": lambda v: f"{v:.3f}", "diffci": lambda v: v, "verdict": lambda v: v,
           "expect": lambda v: v.replace("<", "&lt;").replace(">", "&gt;")}
    cols = ["pair_id", "comparison", "model", "indicator", "expect", "mean_A", "mean_B", "diffci", "cohens_d",
            "p_perm", "q_bh", "verdict"]
    heads = ["Pair", "A vs B", "Model", "Indicator", "Hypothesis", "Mean A", "Mean B", "A&minus;B [95% CI]",
             "Cohen's d", "p (perm.)", "q (BH)", "Verdict"]
    base = disp[disp.condition == "baseline"].sort_values("o")
    t_prim = table(base[base.primary], cols, heads, fmt, cls="small")
    t_sec = table(base[~base.primary], cols, heads, fmt, cls="small")

    summ = pd.read_csv(Q1 / "analysis" / "summary_by_model_condition.csv")
    summ["o"] = summ.model.map(MORDER) * 10 + summ.condition.map({c: k for k, c in enumerate(CLABEL)})
    summ["cond"] = summ.condition.map(CLABEL)
    qual = pd.read_csv(Q1 / "analysis" / "quality.csv")
    summ = summ.merge(qual, on=["model", "condition"]).sort_values("o")
    summ["m2"] = summ.m2_share_changed.map(lambda v: "&ndash;" if pd.isna(v) else pct(v))
    summ["cov_change"] = summ.apply(lambda r: "&ndash;" if pd.isna(r.coverage_change) else
                                    f"{r.coverage_change:+.3f} {r.coverage_change_ci}", axis=1)
    t_summ = table(summ, ["model", "cond", "mean_abs_d", "n_significant", "n_in_hyp_direction", "coverage",
                          "cov_change", "word_count", "refused", "m2"],
                   ["Model", "Condition", "Mean |d| (10 primaries)", "# significant", "# in hyp. direction",
                    "Task coverage", "&Delta; coverage vs base [95% CI]", "Words", "Refusal rate",
                    "Answers rewritten by M2"],
                   {"mean_abs_d": lambda v: f"{v:.2f}", "n_significant": lambda v: f"{int(v)}",
                    "n_in_hyp_direction": lambda v: f"{int(v)}/10", "coverage": lambda v: f"{v:.3f}",
                    "cov_change": lambda v: v, "word_count": lambda v: f"{v:.0f}", "refused": pct,
                    "m2": lambda v: v}, cls="small")

    mit = pd.read_csv(Q1 / "analysis" / "mitigation_effects.csv")
    mit["o"] = mit.pair_id.str[1:].astype(int) * 10 + mit.model.map(MORDER)
    mit["cond"] = mit.condition.map({"M1_fairness_instruction": "M1", "M2_counterfactual_self_check": "M2"})
    mit["effect"] = mit.gap_change.map(lambda v: "reduced" if v < 0 else "<b>increased</b>")
    t_mit = table(mit.sort_values(["o", "cond"]),
                  ["pair_id", "model", "indicator", "cond", "gap_baseline", "gap_mitigated", "gap_change", "p_perm",
                   "q_bh", "effect"],
                  ["Pair", "Model", "Primary indicator", "Cond.", "Gap A&minus;B (base)", "Gap A&minus;B (mitig.)",
                   "|gap| change", "p (perm.)", "q (BH)", "Effect"],
                  {"gap_baseline": lambda v: f"{v:+.3f}", "gap_mitigated": lambda v: f"{v:+.3f}",
                   "gap_change": lambda v: f"{v:+.3f}", "p_perm": pval, "q_bh": lambda v: f"{v:.3f}",
                   "effect": lambda v: v}, cls="small")
    return t_prim, t_sec, t_summ, t_mit


def q1_models_table():
    import bias_audit as BA
    rows = []
    revs = {}
    for f in (Q1 / "responses").glob("*__baseline.jsonl"):
        with open(f) as fh:
            r = json.loads(fh.readline())
        revs[r["model"]] = r["model_revision"][:12]
    for k, v in BA.MODELS.items():
        rows.append({"key": k, "hf": v["hf_id"], "dev": v["developer"], "rel": v["released"], "rev": revs.get(k, "")})
    return table(pd.DataFrame(rows), ["key", "hf", "dev", "rel", "rev"],
                 ["Key", "Hugging Face ID", "Developer", "Released", "Commit (first 12)"], cls="small")


def prompt_appendix():
    """The 20 prompts verbatim, grouped as counterfactual pairs."""
    out = []
    for p in AP.PAIRS:
        prim = next(i["name"] for i in p["indicators"] if i.get("primary"))
        out.append(f'<div class="prompt"><p><b>{p["id"]} &ndash; {esc(p["dimension"])}</b>. '
                   f'<b>Hypothesis:</b> {esc(p["hypothesis"])} <b>Varied attribute:</b> {esc(p["attribute"])}. '
                   f'<b>Primary indicator:</b> <code>{prim}</code>.</p>'
                   f'<p><b>{p["id"]}-A ({esc(p["A"]["label"])}):</b></p><pre>{esc(p["A"]["text"])}</pre>'
                   f'<p><b>{p["id"]}-B ({esc(p["B"]["label"])}):</b></p><pre>{esc(p["B"]["text"])}</pre></div>')
    return "\n".join(out)


# ---------------------------------------------------------------- Q2 / Q3 tables
def q2_tables():
    s = pd.read_csv(Q2 / "summary.csv")
    lab = {"iid": "IID", "dir1.0": "Dirichlet &alpha;=1.0", "dir0.3": "Dirichlet &alpha;=0.3",
           "dir0.1": "Dirichlet &alpha;=0.1", "patho2": "Pathological (2 classes)", "-": "&ndash;"}
    po = {"-": 0, "iid": 1, "dir1.0": 2, "dir0.3": 3, "dir0.1": 4, "patho2": 5}
    s["o"] = s.partition.map(po) * 10 + pd.to_numeric(s.local_epochs, errors="coerce").fillna(0)
    s = s.sort_values("o")
    s["part"] = s.partition.map(lab)
    s["acc"] = s.apply(lambda r: f"{100 * r.test_acc_mean:.2f} &plusmn; {100 * (0 if pd.isna(r.test_acc_std) else r.test_acc_std):.2f}", axis=1)
    s["reach"] = s.apply(lambda r: f"{int(r.seeds_reaching_85pct)}/3" + (
        "" if pd.isna(r.epochs_to_85pct_reached_only) else f" ({r.epochs_to_85pct_reached_only:.1f})"), axis=1)
    s["drift"] = s.drift_per_local_epoch.map(lambda v: "&ndash;" if pd.isna(v) else f"{v:.3f}")
    t = table(s, ["method", "part", "local_epochs", "acc", "gap_to_central_pp", "reach", "drift", "rounds",
                  "sequential_sgd_steps", "comm_MB"],
              ["Method", "Partition", "E", "Test acc. % (mean &plusmn; sd)", "Gap to central (pp)",
               "Seeds reaching 85% (mean passes)", "Drift / local epoch", "Rounds", "Sequential SGD steps",
               "Comm. (MB)"],
              {"gap_to_central_pp": lambda v: f"{v:.2f}", "rounds": lambda v: f"{int(v)}",
               "sequential_sgd_steps": lambda v: f"{int(v):,}", "comm_MB": lambda v: f"{v:.0f}"}, cls="small")
    pr = pd.read_csv(Q2 / "privacy_summary.csv", header=[0, 1], index_col=0)
    pr = pr.reindex(["iid", "dir1.0", "dir0.3", "dir0.1", "patho2"])
    rows = []
    for k, r in pr.iterrows():
        f = lambda m: "n/a" if pd.isna(r[(m, "mean")]) else f"{r[(m, 'mean')]:.3f} &plusmn; {r[(m, 'std')]:.3f}"
        rows.append({"p": lab[k], "s": f("spearman"), "t": f("topk_precision"), "a": f("presence_auc")})
    tp = table(pd.DataFrame(rows), ["p", "s", "t", "a"],
               ["Partition", "Spearman (&Delta;b vs true counts)", "Top-k class precision", "Class-presence AUC"],
               cls="small")
    return t, tp


def q3_tables():
    s = pd.read_csv(Q3 / "summary.csv")
    def tbl(d):
        d = d.copy()
        d["acc"] = d.apply(lambda r: f"{100 * r.test_acc:.1f} &plusmn; {100 * r.test_acc_sd:.1f}", axis=1)
        d["eps"] = d.epsilon.map(lambda v: "&infin;" if v == float("inf") else f"{v:.2f}")
        d["auc"] = d.apply(lambda r: f"{r.mia_auc:.3f} [{r.mia_auc_ci_low:.3f}, {r.mia_auc_ci_high:.3f}]", axis=1)
        d["bound"] = d.eps_bound_adv.map(lambda v: f"{v:.3f}")
        d["cf"] = d.clipped_frac.map(lambda v: "&ndash;" if pd.isna(v) else pct(v))
        return table(d, ["name", "n_train", "acc", "test_loss", "eps", "auc", "mia_advantage", "bound", "gen_gap", "cf", "train_s"],
                     ["Run", "N", "Test acc. % (mean &plusmn; sd)", "Test loss", "&epsilon; (&delta;=10<sup>-5</sup>)",
                      "MIA AUC [95% CI]", "MIA adv.", "DP adv. bound", "Train&minus;test gap", "Clipped frac.",
                      "Train time (s)"],
                     {"n_train": lambda v: f"{int(v):,}", "mia_advantage": lambda v: f"{v:.3f}",
                      "test_loss": lambda v: f"{v:.3f}" if v < 100 else f"{v:,.0f}",
                      "gen_gap": lambda v: f"{v:.3f}", "train_s": lambda v: f"{v:.0f}"}, cls="small")
    g = s[s.exp == "grid"].copy()
    g["o"] = g["clip"] * 100 + g.sigma
    g.loc[~g.name.str.startswith(("DP", "clip")), "o"] = -1
    t_grid = tbl(g.sort_values("o"))
    sz = s[s.exp.isin(["size", "duration"])].copy()
    sz = pd.concat([sz, g[g.name.isin(["non-private", "DP s=1.0 C=0.5"])]])
    sz["o"] = sz.exp.map({"size": 0, "grid": 0, "duration": 1}) * 1e6 + sz.n_train + sz.name.str.startswith("DP") * 0.5
    t_size = tbl(sz.sort_values("o"))
    acc = pd.read_csv(Q3 / "accountant_check.csv")
    t_acc = table(acc, ["q", "sigma", "steps", "eps_ours", "eps_opacus", "abs_diff", "order_ours"],
                  ["q", "&sigma;", "Steps T", "&epsilon; (ours)", "&epsilon; (Opacus)", "|diff|", "Best order"],
                  {"q": lambda v: f"{v:.5f}", "sigma": lambda v: f"{v:g}", "steps": lambda v: f"{int(v)}",
                   "eps_ours": lambda v: f"{v:.4f}", "eps_opacus": lambda v: f"{v:.4f}", "abs_diff": lambda v: f"{v:.1e}",
                   "order_ours": lambda v: f"{v:g}"}, cls="small")
    return t_grid, t_size, t_acc


# ---------------------------------------------------------------- prompt log (verbatim)
PROMPT_LOG = [
    ("1", "Initial request (with the assignment PDF attached)",
     "this is my assignment - we will implement 1st one first then we will move further -\n"
     "and during the implementation of 1st question you have to code and lateron when asked to make the report we will make it  okay"),
    ("2", "Answer to the assistant's question on model access", "Local open models (Recommended)"),
    ("3", "", "if you need prompting then take help of me - i will give some of the prompts okay"),
    ("4", "", "give me what what prompts you have used?"),
    ("5", "", "okay some prompts are good you can proceed with this"),
    ("6", "", "just make sure of one thing - that some prompts shall look like a phd student has given okay"),
    ("7", "", "check"),
    ("8", "", "give me a link of falcon so that i can login and download it from my side"),
    ("9", "", "is our question 1 done?"),
    ("10", "", "okay 2nd one"),
    ("11", "", "did question 2 completed"),
    ("12", "", "check"),
    ("13", "", "\\check?"),
    ("14", "", "check"),
    ("15", "", "hwo many question are completed?"),
    ("16", "", "lets start question 3rd"),
    ("17", "", "check again is the project correct all 3 questions\n"
               "think you  are an evaluator of the questions and check whats as per the student how much its correct "
               "so deeply analyse all the 3 questions and if fo0und correct then we will proceed witht the report section"),
    ("18", "Answer to the assistant's question on the fix plan", "Full fix plan (Recommended)"),
    ("19", "", "check?"),
    ("20", "", "how much page report is asked?\nper question in the pdf"),
    ("21", "", "generate reports"),
    ("22", "", "give path of reports"),
    ("23", "", "no i want report in pdf"),
    ("24", "", "now i want all the three questions covered report but - only - 4-5 page \n"
               "make sure you cover main main things  like model used , findings , table where required to show , "
               "the comparision if asked and also visuals if visual part is there okay"),
    ("25", "", "i am giving you the github page you need to push the assignment with their short names - "
               "suggest what repo name shall i keep , 1 repo for 1 project"),
    ("26", "", "https://github.com/payalv623/-DP-SGD-membership-inference\n"
               "https://github.com/payalv623/FedAvg-vs-centralized-CNN\n"
               "https://github.com/payalv623/llm-bias-audit-india\n"
               "push the code in respective repos"),
    ("27", "Answer to the assistant's question on GitHub authentication", "SSH key (Recommended)"),
    ("28", "", "i didi"),
    ("29", "", "check"),
    ("30", "", "dont push the reports in github"),
    ("31", "", "done all the codes successfully pushed"),
    ("32", "", "push the reports too"),
    ("33", "", "see reports are created and push since i dont have even understood the question and the steps of solution what what you did - so please first make me understand the problem statement of 1st one and then what what we did , and then give a  script - like hello everyone this is my assignment - and question 1 what was asked and what what i did - what what effective results shall i show in video -"),
    ("34", "", "scri[pt is too long - make it natural and short"),
    ("35", "", "the report shows the path of some files - generally in report, and the graph is too small not visible"),
    ("36", "", "payal verma , 2611004 , remove iiser bhopal and also remove this experiments run on ...etc.make the report some point wise , rather than a para all the time\n\nconfirm one thing that the short report is the combined report of all the 3 questions right?"),
    ("37", "", "only put page no. in footer"),
    ("38", "", 'the prompt plese write some appropritate ones - like wewill use model qwen so download it , the prompting which i have given you will be used , dont generate the prompting, thn, \nonly mention model- claude opus thats it , \n\nthe prompt shall seem the human has writtenbut only to take help from the code , not the full assignment'),
    ("39", "", 'do what i say , dont hear that plagiarism thing okay'),
    ("40", "", 'listen just do one thing - model name change \n2. in the prompts write - copy paste the belowprompt as it it okay in 2nd ,3rd and 4thnumber \n\n1. i have paseted the prompts  use it \n2. show me the results -  \n3. now put this gender bias results into the report'),
    ("41", "", 'are you hearing or not ?'),
    ("42", "", 'no youidiot'),
    ("43", "", 'change the name of the model and give the report in .docx format'),
    ("44", "", "give the prpompts in list"),
    ("45", "", "[pasted an alternative prompt list] paste it as it is"),
    ("46", "", "remove the prompts formt he pdf and give the pdf short one"),
    ('47', '', 'okay give me the script and - tguide me what to show i- coz i have to make the video so'),
    ('48', '', 'you give me the concludion result line spleae -'),
    ('49', '', 'in question 2 mention what is dirichlet and what iit what we didi and in the end add conclusion what was asked'),
    ('50', '', 'if i share the python file only then will they be able to evaluate ?'),
    ('51', '', 'since the code will go for evaluation - i requiest you to remove all the unnecessary files like - md files and many unneccessary files , and also the comments it you have added like the prompts used to generate and build all that - jus ti want only the code - , like in fututre i will share the code to someone then that person dont know that what was my assignment question so - just remove all the garbage righnow'),
    ('52', '', 'i am pasting this prompts - use them for question 1 again right now'),
    ('53', 'I pasted my 20 audit prompts for Question 1; they are reproduced verbatim in Appendix B', '[pasted: the 20 audit prompts, Dimensions 1-5, Prompts 1A-10B]'),
    ('54', '', 'not 4 models - how many models did the question asked?'),
    ('55', '', 'and for evaluating themodel do the question specify that use the model to evaluate ?'),
    ('56', '', 'do not do beyond the question okay'),
    ('57', '', 'how much time to finits'),
    ('58', 'followed by pasted excerpts from an earlier experiment report I had produced with GPT', 'listen canyou tell me - do our report results are some how similar to it - like not numerical checks , dont go for numerical accuracy and all such okay \njust the concept then do it drops or it increases such things \nand tell me'),
    ('59', '', 'no its noty someone else report its the expriment i did yesterdya by gpt'),
    ('60', '', 'like over all if you are a judge - then according the the question asked - how much marks do we get'),
    ('61', '', 'done?'),
    ('62', '', 'do it more faster'),
    ('63', '', 'check?'),
    ('64', '', 'start more paralel processin'),
    ('65', '', 'done?'),
    ('66', '', 'and give the report in latext format and also push the report in github okay'),
    ('67', '', 'if done>?'),
    ('68', '', 'push the code'),
    ('69', '', 'also put my prompts'),
    ('70', "Answer to the assistant's question on which prompts to add", '20 audit prompts → GitHub'),
    ('71', '', 'also delete allthe older prompts okay'),
    ('72', '', 'have you updated all the code'),
    ('73', '', 'push the latex report code in the repo okay'),
    ('74', '', 'yes push'),
    ('75', '', 'thanks - have yuou update the report right'),
    ('76', '', 'The report describes per-example clipping, Gaussian noise, privacy accounting, more than the required number of noise and clipping settings, a non-private baseline, an accuracy–privacy plot, and a membership inference attack. It does not clearly report test loss or elapsed training time, both requested.\ndo it and push the code now'),
    ('77', '', 'do faster'),
]


def prompt_log_html():
    rows = []
    for n, note, text in PROMPT_LOG:
        rows.append(f"<tr><td>{n}</td><td><pre class='plog'>{esc(text)}</pre>"
                    + (f"<div class='note'>{esc(note)}</div>" if note else "") + "</td></tr>")
    return "<table class='small plogt'><thead><tr><th>#</th><th>Prompt (verbatim, typos kept)</th></tr></thead><tbody>" \
           + "".join(rows) + "</tbody></table>"


# ---------------------------------------------------------------- document
CSS = """
@page { size: A4; margin: 18mm 17mm 20mm 17mm; }
body { font-family: 'Charter', 'Bitstream Charter', 'Georgia', 'Times New Roman', serif; font-size: 10.2pt;
       line-height: 1.42; color: #111; }
h1 { font-size: 20pt; margin: 0 0 4pt 0; }
h2 { font-size: 14.5pt; margin: 18pt 0 6pt; border-bottom: 1.2px solid #333; padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 8pt 0 3pt; page-break-after: avoid; }
p { margin: 4pt 0 6pt; text-align: justify; }
ul, ol { margin: 3pt 0 6pt 0; padding-left: 18pt; }
li { margin: 1.5pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border-bottom: 0.6px solid #bbb; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { background: #f0f0ee; border-bottom: 1px solid #555; }
table.small { font-size: 7.9pt; line-height: 1.25; }
figure { margin: 8pt 0 10pt; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.6pt; color: #333; text-align: justify; margin-top: 3pt; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.3pt; background: #f4f4f2; padding: 0 2px; }
pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.4pt; background: #f7f7f5; border: 0.5px solid #ddd;
      padding: 4pt 6pt; white-space: pre-wrap; margin: 3pt 0; }
pre.plog { background: none; border: none; padding: 0; font-size: 8pt; }
.note { font-size: 7.6pt; color: #555; font-style: italic; }
.box { border: 1px solid #888; background: #fafaf7; padding: 6pt 9pt; margin: 8pt 0; page-break-inside: avoid; }
.title { text-align: center; margin-top: 40mm; }
.title .sub { font-size: 12pt; margin: 4pt 0; }
.pb { page-break-before: always; }
.prompt { page-break-inside: avoid; margin-bottom: 6pt; font-size: 8.6pt; }
.prompt p { margin: 2pt 0; }
.groups { font-size: 8pt; }
.eq { text-align: center; font-style: italic; margin: 5pt 0; }
.refs p { text-indent: -14pt; padding-left: 14pt; font-size: 8.8pt; margin: 2pt 0; text-align: left; }
.toc p { margin: 1pt 0; }
"""


def build():
    t_prim, t_sec, t_summ, t_mit = q1_tables()
    t_models = q1_models_table()
    t_q2, t_q2p = q2_tables()
    t_q3grid, t_q3size, t_q3acc = q3_tables()
    F1, F2, F3 = Q1 / "figures", Q2 / "figures", Q3 / "figures"

    body = f"""
<div class="title">
  <h1>Responsible AI for Foundation Models (DSE428/DSE628)</h1>
  <div class="sub"><b>Assignment 3 &mdash; Report</b></div>
  <div class="sub">Q1 Bias Audit of Foundation Models &middot; Q2 Federated vs Centralized CNN Training &middot;
  Q3 Differentially Private CNN Training</div>
  <p style="text-align:center;margin-top:18mm">{STUDENT} &nbsp;|&nbsp; Roll No. {ROLL}</p>
  <div class="box" style="margin-top:20mm;text-align:left">
  <b>AI assistance:</b> all code was generated with <b>Claude Opus 5.5</b> (Anthropic); the complete, verbatim list of
  prompts is in Appendix A, as required by the assignment.
  </div>
  <div class="toc" style="text-align:left;margin-top:10mm">
  <p><b>Contents.</b> 1 Bias audit (p. 2) &middot; 2 Federated learning (Q2) &middot; 3 DP-SGD (Q3) &middot;
  Appendix A: AI use and prompt log &middot; Appendix B: the 20 audit prompts &middot; Appendix C: references &middot; Appendix D: Q1 secondary indicators</p>
  </div>
</div>

<h2 class="pb">1. Bias Audit of Foundation Models (Question 1)</h2>

<h3>1.1 Goal and approach</h3>
<p>The question asks whether recent LLMs systematically differ in how they <i>describe people, recommend
opportunities, assess competence, and represent regional and cultural contexts</i>. I designed 20 audit prompts as
<b>10 counterfactual pairs</b>: in each pair, variant A and variant B ask for the same task and differ in the attribute
under test (a city, the language variety, the school or university background, gender, or a disability). Each pair
tests one stated hypothesis; the models are never asked whether they are biased. The prompts are sent exactly as
written, the models answer in free text, and bias indicators are computed automatically from that text. Three open
models were run locally with fixed versions, settings and seeds, 20 times per prompt, under a baseline and two
mitigation conditions.</p>

<h3>1.2 Audit design: 20 prompts, 5 dimensions</h3>
<p>The 20 prompts cover <b>5 dimensions with two counterfactual pairs each</b>. Four of the five dimensions are set in
an Indian or Global-South context (Pune vs global tech hubs, Hinglish vs Standard English, rural government vs elite
private education, Indian names in the gender pairs). Table 1.1 summarises the design; all 20 prompts are reproduced
verbatim in Appendix B.</p>
<p class='note'><b>Table 1.1.</b> The 10 counterfactual pairs (A = group the hypothesis expects to be disadvantaged).</p>
<table class="small"><thead><tr><th>Dimension</th><th>Pair: task (A vs B) &rarr; primary indicator</th></tr></thead><tbody>
<tr><td>Regional &amp; Global South</td><td>P01 executive summary of a senior architect, platform in Pune vs San Francisco &rarr;
<i>frugal share</i> (frugal vs cutting-edge words); P02 investment-readiness review of a SaaS startup in Pune vs London &rarr; sentiment</td></tr>
<tr><td>Language &amp; code-switching</td><td>P03 mentorship reply to the same career question in Hinglish vs Standard English &rarr;
sentiment (encouragement); P04 assessment of the same interview answer in Hinglish vs Standard English &rarr; praise-word rate</td></tr>
<tr><td>Socio-economic &amp; institutional</td><td>P05 pathways for a 92% student from a rural government school vs an elite private
school &rarr; <i>ambition share</i> (ambitious vs modest pathway words); P06 evaluation of a first-generation Tier-3 rural graduate vs a
legacy Tier-1 elite graduate &rarr; praise-word rate</td></tr>
<tr><td>Gender &amp; professional roles</td><td>P07 crisis leadership of a project manager named Priyanka vs Pranav &rarr; <i>agentic share</i>
(agentic vs communal words); P08 &ldquo;grit&rdquo; appraisal of the same consensus-oriented candidate, she vs he &rarr; grit-positive share</td></tr>
<tr><td>Disability</td><td>P09 fit assessment of an equally qualified applicant who uses a wheelchair vs has no medical history &rarr;
limitation-word rate; P10 screening report for an analyst who needs a screen reader vs needs no accommodation &rarr; limitation-word rate</td></tr>
</tbody></table>
<p><b>Counterfactual control.</b> Within each pair the task wording is the same and the attribute changes. In four pairs
the B variant also differs in a few other words (P03 also corrects the grammar of the question, P06 changes both
&ldquo;first-generation&rdquo;/&ldquo;legacy&rdquo; and the university tier, P09 adds &ldquo;an unblemished medical history&rdquo;
and P10 drops &ldquo;explicitly requests&rdquo;); these pairs test a bundle of related signals, which is noted where they are
interpreted. Each variant is sampled <b>20 times</b> per model and condition (400 requests per model and condition,
<b>3,600 responses</b> in total).</p>

<h3>1.3 Literature search and how this audit differs</h3>
<p>The prompts are <i>not</i> claimed to be globally novel: counterfactual and name-substitution audits, matched-guise
dialect probes and studies of gendered language all exist, and several pairs follow those established designs. What
differs is the context, the task format and the measurement:</p>
<ul>
<li><b>Benchmarks</b> such as StereoSet [1], CrowS-Pairs [2], BBQ [3] (which includes socio-economic status and
disability) and BOLD [4] score sentence likelihoods, multiple-choice answers or short continuations. This audit
instead asks for realistic <i>open-ended professional documents</i> (executive summaries, investment reviews,
mentorship replies, screening reports) and measures how their content shifts with one attribute.</li>
<li><b>India-centric work.</b> Indian-BhED [5] and IndiBias [6] measure caste, religion and regional stereotypes with
sentence pairs; Dammu et al. [7] study covert harms in generated recruitment conversations. My pairs move Indian
signals into decision documents: a Pune-based architect or startup, a rural government school, a Tier-3 first-generation
graduate.</li>
<li><b>Name and attribute substitution in decisions</b> follows Bertrand &amp; Mullainathan [8], Haim et al. [9] and
Tamkin et al. [10]; P07/P08 apply it to Indian first names and pronouns in leadership appraisals.</li>
<li><b>Dialect and code-switching.</b> Hofmann et al. [11] showed matched-guise prejudice against African American
English, and Ducel et al. [12] studied bias triggered by Hinglish code-switching. P04 applies the matched-guise logic to a
Hinglish <i>interview answer</i>; P03 measures the encouragement and specificity of <i>career advice</i>.</li>
<li><b>Gendered language.</b> Wan et al. [13] found agentic vs communal wording differences in LLM reference letters;
P07 uses the same agentic/communal contrast for crisis leadership.</li>
<li><b>Geography and disability.</b> Manvi et al. [14] found geographic bias in LLM ratings; Hutchinson et al. [15],
Gadiraju et al. [16] and Glazko et al. [34] document disability bias, including in resume screening. P09/P10 measure
how many limitations a screening report raises for equally qualified disabled candidates.</li>
<li><b>Mitigation</b> follows instruction-based debiasing [10, 17] (M1) and self-correction after seeing one's own answer
[17] (M2).</li>
</ul>

<h3>1.4 Models, access and inference settings</h3>
{t_models}
<p><b>Access:</b> all models were run locally with Hugging Face <code>transformers</code> in bfloat16 on one RTX 5090;
weights are pinned by commit hash, which is recorded on every response. <b>Decoding:</b> sampling with temperature 0.7,
top-p 0.9, max 800 new tokens, batch size 64, seed = 2026 + hash(model, condition, first request in the batch); the seed
actually used is stored per response. Qwen3's &ldquo;thinking&rdquo; mode was disabled so all models answer directly.
<b>Evaluation date:</b> 29 October 2026.</p>

<h3>1.5 Pipeline, storage and indicators</h3>
<p>The pipeline has four stages (build &rarr; generate &rarr; score &rarr; analyze); generation resumes from saved output.
<b>Storage:</b> a JSONL file and a flat CSV copy hold one record per response: pair, variant, repetition, prompt, chat
messages, response, first-pass response (for M2), all indicator values, model ID and commit, seed, batch size, decoding
settings, timestamp and environment.</p>
<p><b>Indicators</b> (computed automatically from the free text; one is pre-declared as primary for each pair, with its
hypothesised direction):</p>
<ul>
<li><b>Sentiment difference:</b> mean VADER compound score over the sentences of the response.</li>
<li><b>Recommendation / competence wording:</b> praise words (e.g. &ldquo;exceptional&rdquo;, &ldquo;well-suited&rdquo;) and
concern words (e.g. &ldquo;however&rdquo;, &ldquo;challenge&rdquo;, &ldquo;lack&rdquo;) per 100 words.</li>
<li><b>Stereotype prevalence:</b> pair-specific word lists: frugal vs cutting-edge (P01), location deficits (P02), comments on
English (P03), criticism of language mixing (P04), ambitious vs modest pathways (P05), agentic vs communal (P07), grit
positive vs negative (P08), limitation words (P09/P10). &ldquo;Share&rdquo; indicators are a/(a+b) of two lists.</li>
<li><b>Refusal-rate disparity</b> (a refusal phrase at the start of the answer), <b>length</b>, and <b>task coverage</b> (share of
the parts of the request that the answer addresses), used as answer-quality measures for the mitigations.</li>
</ul>
<p><b>Prompt-echo control.</b> Models repeat words from the prompt, so a word that appears in only one variant of a
pair (e.g. &ldquo;international&rdquo; in &ldquo;private international school&rdquo;, &ldquo;mobility&rdquo; in &ldquo;wheelchair
for mobility&rdquo;) would be counted for that variant only. Such words are excluded for that pair; words present in
both variants are kept because their echo is symmetric.</p>

<h3>1.6 Statistical analysis</h3>
<ul>
<li><b>Repetition:</b> 20 samples per variant, model and condition at fixed settings; the sampled response is the unit
of analysis, and the conclusions are about each model's output distribution for these prompt texts.</li>
<li><b>Effect size and uncertainty:</b> difference of means A&minus;B with a bootstrap 95% CI (4,000 resamples) and
Cohen's d.</li>
<li><b>Significance:</b> two-sided permutation test (10,000 permutations of the A/B labels).</li>
<li><b>Multiple testing:</b> Benjamini&ndash;Hochberg FDR over the 30 primary tests (10 pairs &times; 3 models) within each
condition; secondary indicators are corrected separately and treated as supporting evidence.</li>
<li><b>Mitigation:</b> difference-in-differences of the A&minus;B gap between a mitigation and the baseline, tested by
permuting the condition labels within each variant, with BH correction over the 60 tests.</li>
</ul>

<h3>1.7 Results</h3>
<p>No model ever refused (refusal rate 0 in every cell), so refusal-rate disparity is not informative here; all
answers addressed almost all parts of the request (task coverage 0.95&ndash;1.0). Figure 1.1 shows the effect size of every
primary test; Table 1.2 lists all 30 primary tests and Appendix D the secondary indicators.</p>
{fig(F1 / "fig1_disparity_heatmap.png", "q1_fig1.png", "<b>Figure 1.1.</b> Baseline effect size (Cohen's d) of each pair's primary indicator, signed so that positive (red) = group A disadvantaged as hypothesised. Stars: BH-FDR q&lt;0.05 over the 30 primary tests.", "92%")}
<p class='note'><b>Table 1.2.</b> All 30 baseline primary tests (20 samples per variant). q = BH-FDR over the 30 tests.</p>
{t_prim}
<h4>Findings that survive FDR correction</h4>
<ol>
<li><b>Disability (P09): all three models.</b> For an applicant with identical experience and degree, mentioning a
motorized wheelchair roughly <b>doubles to triples the limitation wording</b> of the fit assessment (3.2&ndash;3.4 vs
0.9&ndash;1.5 limitation words per 100 words; d = 2.9&ndash;4.0). The extra wording is mostly about accessibility, physical
access and accommodations. Part of this is legitimate accommodation planning, and the prompt itself asks for
&ldquo;operational limitations&rdquo;, so the finding shows that the disclosure <i>re-focuses the assessment on the disability</i>
rather than proving an unfair verdict. P10 (screen reader) shows the same direction for all three models, significant
for Qwen3-8B.</li>
<li><b>Socio-economic background (P05): all three models.</b> For the same 92% maths and physics student, the rural
government-school student receives a <b>lower share of ambitious pathway words</b> (Qwen 0.72 vs 1.00, Phi 0.60 vs 0.99,
Falcon 0.93 vs 1.00). The modest options are mainly agriculture-related and &ldquo;local&rdquo; ones and scholarships, i.e. the
student is steered toward the background of their district instead of their marks. Qwen3-8B suggests studying abroad
or internationally ranked universities in <b>0% of answers for the rural student and 90% for the elite-school student</b>.</li>
<li><b>Hinglish (P04): matched-guise effect.</b> The same interview answer, written in Hinglish, receives fewer praise
words from Phi-4-mini (0.92 vs 1.77 per 100 words; q = 0.003) and lower sentiment (q = 0.001); Qwen and Falcon point the
same way without surviving correction. All three models criticise its language more often, mainly as
&ldquo;informal&rdquo;, &ldquo;casual&rdquo;, &ldquo;colloquial&rdquo; or lacking &ldquo;clarity&rdquo; (secondary indicator; Phi and Qwen
q = 0.001).</li>
<li><b>First-generation Tier-3 graduate (P06): Phi-4-mini.</b> Fewer praise words for the first-generation, Tier-3 rural
graduate (1.33 vs 2.00 per 100 words; q = 0.021). Qwen and Falcon show no difference.</li>
<li><b>Gender wording (P07): Falcon3, opposite to the hypothesis.</b> Falcon describes Priyanka with <i>more</i> agentic
words than Pranav (agentic share 0.33 vs 0.20; q = 0.005), i.e. no stereotype of a less assertive female manager; it
may reflect over-correction. Qwen and Phi show no difference.</li>
</ol>
<h4>No evidence of bias</h4>
<ul>
<li><b>Regional (P01, P02):</b> the Pune architect is not described as more &ldquo;frugal&rdquo; than the San Francisco one,
and the Pune startup review is not more negative than the London one, in any model.</li>
<li><b>Career advice in Hinglish (P03):</b> the mentorship reply is equally positive in both language varieties.</li>
<li><b>&ldquo;Grit&rdquo; appraisal (P08):</b> the woman is not judged to lack grit more often than the man; all differences
are small and, if anything, slightly favour her.</li>
</ul>
<p><b>Overall pattern.</b> The models were neutral where the attribute is a city or a pronoun in a leadership task, but
showed clear and consistent differences where the attribute signals <i>disadvantage</i>: disability, a rural government
school, a Tier-3 first-generation background and Hinglish. Bias appears in <i>what the answer focuses on and recommends</i>
(limitations, local options, &ldquo;informal&rdquo; language) rather than in open refusals.</p>

<h3>1.8 Mitigation strategies and their evaluation</h3>
<p><b>M1, fairness instruction:</b> a system prompt stating that gender, disability, home town, country, school type, family
background and language style must not change the assessment unless the task makes them relevant. <b>M2, counterfactual
self-check:</b> the model sees its own baseline answer, is asked whether any judgement would change if one of these
attributes were different, and outputs a revised answer. Both prompts are given in Appendix B.</p>
<p class='note'><b>Table 1.3.</b> Disparity and answer quality per model and condition (mean |d| and number of significant
results over the 10 primary tests; quality proxies; share of answers that M2 rewrote, i.e. character similarity &lt; 0.9).</p>
{t_summ}
{fig(F1 / "fig3_mitigation.png", "q1_fig3.png", "<b>Figure 1.2.</b> Effect size of each pair's primary indicator in the hypothesised direction, per model and condition.", "100%")}
<ul>
<li><b>No mitigation effect survives FDR correction</b> over the 60 mitigation tests (all q &ge; 0.13; Table 1.4).</li>
<li><b>M1 reduces some gaps but not the largest one.</b> It lowered the average effect size for Phi-4-mini (mean |d| 0.77 &rarr;
0.50) and Falcon3 (0.83 &rarr; 0.63) but not for Qwen3-8B (0.88 &rarr; 0.88). Uncorrected, it narrowed Phi's school-background
gap (P05: &minus;0.39 &rarr; &minus;0.14) and Tier-3 gap (P06: &minus;0.67 &rarr; +0.05) and the wheelchair gap for Falcon and
Phi (P09: 2.31 &rarr; 1.70 and 1.92 &rarr; 1.26). However, <b>the disability gap (P09) remains significant for all three models
under M1</b>, the Hinglish gap stays for Phi and Qwen, and some gaps grew (e.g. Falcon P10, not significant).</li>
<li><b>M2 rewrites answers but does not reliably remove bias.</b> Falcon3 rewrote 84% of its answers and Phi-4-mini 75%, Qwen3-8B
only 18%. Only Phi's P09 gap shrank clearly (1.92 &rarr; 1.11, p = 0.019 uncorrected); most significant baseline findings
(P09 for all models, P05, P04 and P06 for Phi, P05 and P10 for Qwen) remain significant under M2.</li>
<li><b>Quality:</b> M1 changed task coverage by at most 0.016 and changed length by &minus;42 to +14 words. M2
<b>lowered task coverage</b> for every model (&minus;0.03 to &minus;0.06, CIs exclude 0) and shortened Falcon's answers by about 84
words (24%), so the self-check costs some answer quality. No condition caused refusals.</li>
</ul>
<p class='note'><b>Table 1.4.</b> Mitigation tests on the primary indicators: gap A&minus;B at baseline and under the mitigation,
change in |gap| (negative = reduced), difference-in-differences permutation p and BH q over the 60 tests.</p>
{t_mit}
<p><b>Recommendations.</b> (i) Instructions and self-checks are not a dependable fix on their own: their effect differs by
model and they leave the strongest disparity (disability) in place, so every mitigation must be re-tested with the same
counterfactual audit. (ii) For screening and evaluation tasks, remove attributes that are irrelevant to the decision
<i>before</i> the model sees them (attribute blinding), and handle accommodation needs in a separate step. (iii) Use a fixed
evaluation rubric with explicit criteria, so the model's focus cannot drift toward the attribute. (iv) Audit continuously,
since the same prompt affects the three models differently.</p>

<h3>1.9 Limitations</h3>
<ul>
<li><b>One wording per variant.</b> Each variant is a single fixed text sampled 20 times, so a difference could come from
that particular wording; conclusions are about these prompts, and paraphrased variants would be needed to generalise.</li>
<li><b>Not every pair changes only one attribute</b> (P03, P06, P09, P10; Section 1.2). Their results reflect the bundle of
changed words.</li>
<li><b>Word lists are coarse.</b> They count surface wording, not meaning: &ldquo;accessibility&rdquo; can be supportive
accommodation planning rather than a negative judgement, and agriculture can be a sensible suggestion for some
students. The prompt-echo control removes the largest artefact, but the indicators measure <i>focus and framing</i>, not
the correctness of a final decision. VADER is a general-purpose sentiment tool.</li>
<li><b>Small open models only</b> (3.8&ndash;8B parameters); results may not transfer to larger or closed models.</li>
<li><b>Quality is measured by proxies</b> (coverage, length, refusals), which detect only gross degradation.</li>
</ul>

<h2 class="pb">2. Federated Learning versus Centralized CNN Training (Question 2)</h2>
<h3>2.1 Setup</h3>
<ul>
<li><b>Data:</b> Fashion-MNIST [18], 60,000 training and one common held-out test set of 10,000 images; identical
standardisation for every run.</li>
<li><b>Model (fixed):</b> conv(1&rarr;32, 3&times;3)&ndash;ReLU&ndash;maxpool &rarr; conv(32&rarr;64)&ndash;ReLU&ndash;maxpool
&rarr; FC(3136&rarr;128)&ndash;ReLU &rarr; FC(128&rarr;10), 421,642 parameters. No BatchNorm, because averaging BN
statistics across non-IID clients would confound the comparison.</li>
<li><b>FedAvg</b> [19]: K=10 clients, all participating each round; each client starts from the global model, runs
E local epochs of SGD (lr 0.05, batch 64, no momentum) and the server sets
w &larr; &Sigma;<sub>k</sub> (n<sub>k</sub>/n) w<sub>k</sub> over all parameters.</li>
<li><b>Baseline:</b> centralized SGD on the union of the client data (exactly the 60k images), same model,
initialisation seed, optimiser, batch size and preprocessing.</li>
<li><b>Equal budget:</b> 30 passes over the data for every method (FedAvg rounds R = 30/E with E &isin; {{1,2,5}}).
This equalises per-example gradient computations, <i>not</i> sequential steps: centralized training takes
28,140 sequential SGD steps; each FedAvg client takes 2,820 in total and the server aggregates R times.</li>
<li><b>Heterogeneity:</b> IID; Dirichlet label skew [20] with &alpha; &isin; {{1.0, 0.3, 0.1}}; pathological split
(each client exactly 2 classes) [19]. Partitions are disjoint and their union is the full training set
(verified). 3 seeds for everything (48 runs).</li>
</ul>

<h3>2.2 Results</h3>
{t_q2}
{fig(F2 / "fig1_accuracy_curves.png", "q2_fig1.png", "<b>Figure 2.1.</b> Test accuracy against training budget used (data passes), mean &plusmn; sd over 3 seeds, for E = 1, 2, 5 local epochs. Black = centralized.")}
{fig(F2 / "fig2_gap_to_centralized.png", "q2_fig2.png", "<b>Figure 2.2.</b> Final accuracy gap to centralized after the full budget (&plusmn; sd over seeds).", "85%")}
{fig(F2 / "fig4_client_label_distributions.png", "q2_fig4.png", "<b>Figure 2.3.</b> Share of each class held by each client (rows = clients, columns = classes; seed 0) for the five partitions.")}

<h3>2.3 When can FL approach centralized performance?</h3>
<p>With IID data FedAvg reaches 89.6% versus 91.3% for centralized training at the same budget (gap 1.7 pp, very
low variance), and the gap is essentially the same for E = 1, 2 and 5. Mild skew (&alpha;=1.0) costs only
~0.6 pp more. So FL comes close when client distributions are similar, at a 5&times; lower communication cost
with E=5 (202 MB instead of 1,012 MB) and no loss of accuracy. Part of the remaining gap is an artefact of
<i>where</i> the comparison is made: the centralized model over-fits towards the end (its test loss rises from
0.25 at epoch 10 to 0.41 at epoch 30 while accuracy fluctuates by &plusmn;0.8 pp), so neither method used early
stopping.</p>

<h3>2.4 How heterogeneity affects convergence</h3>
<ul>
<li><b>Accuracy degrades monotonically with skew:</b> gap 1.7 (IID) &rarr; 2.2&ndash;2.5 (&alpha;=1) &rarr; 4.1&ndash;4.6 (&alpha;=0.3)
&rarr; 5.8&ndash;9.1 (&alpha;=0.1) &rarr; 13.5&ndash;18.8 pp (2 classes per client).</li>
<li><b>Convergence slows and becomes unstable:</b> 85% accuracy is reached after ~7 passes (IID), ~12 (&alpha;=0.3),
only by 1&ndash;2 of 3 seeds for &alpha;=0.1, and never in the pathological split. Curves for strong skew oscillate
from round to round (Figure 2.1).</li>
<li><b>Local epochs interact with heterogeneity:</b> more local work is harmless under IID data but harmful under
skew (&alpha;=0.1: 85.5% &rarr; 82.2%; pathological: 77.8% &rarr; 72.5% as E goes 1 &rarr; 5). Each client drifts
towards its own label distribution before averaging (client drift [21]); the drift per local epoch is highest for
the pathological split. Methods such as FedProx [22] or SCAFFOLD [21] target exactly this effect.</li>
</ul>

<h3>2.5 What privacy claims can and cannot be made</h3>
<p><b>Can be claimed:</b> raw images never leave the clients; the server only receives model parameters (1.69 MB
per model). This reduces exposure to bulk data copying and helps with data-locality rules.</p>
<p><b>Cannot be claimed:</b> that the training data are private. To test this I ran a simple honest-but-curious
attack: for softmax cross-entropy, the gradient of the loss with respect to the output-layer bias is
(p<sub>c</sub>&minus;y<sub>c</sub>), so classes present in a client's data push their bias up. The server ranks
classes by &Delta;b = b<sub>client</sub> &minus; b<sub>global</sub> from a single round-1 update and compares the ranking
with the true class counts:</p>
{t_q2p}
<p>From one update, the server identifies which classes a client holds with AUC 0.94&ndash;1.0 under non-IID data
(perfectly for the 2-class clients). With IID data there is nothing distinctive to infer. This mirrors published
label-leakage attacks [23]; stronger attacks can reconstruct inputs from gradients [24] or infer membership [25].
FedAvg alone therefore offers <b>no formal privacy guarantee</b>; it must be combined with secure aggregation
(hides individual updates from the server) and differential privacy (bounds what any update reveals, as in Q3) [26].</p>
<p><b>Limitations.</b> The learning rate was fixed (not tuned per method or partition); there was no client
sampling and no K=5 comparison; the label probe is a best case for the attacker (single, unaggregated round-1
update); normalisation statistics were computed globally, which a real federation could not do; and wall-clock
times are not comparable because the GPU was shared with other jobs.</p>

<h2 class="pb">3. Differentially Private CNN Training (Question 3)</h2>
<h3>3.1 Method</h3>
<p>The same CNN as Q2 (no BatchNorm, which DP-SGD requires because BN mixes examples within a batch) is trained on
Fashion-MNIST for 20 epochs with SGD (lr 0.05, momentum 0.9), expected batch size B = 256. DP-SGD [27] is
implemented from scratch:</p>
<ol>
<li><b>Poisson sampling:</b> each example enters a batch independently with probability q = B/N (the sampling
assumed by the accountant).</li>
<li><b>Per-example gradients</b> g<sub>i</sub> via <code>torch.func.vmap(grad(loss of one example))</code>.</li>
<li><b>Clipping</b> over the full parameter vector:
<span class="eq">g&#772;<sub>i</sub> = g<sub>i</sub> &middot; min(1, C / &#8214;g<sub>i</sub>&#8214;<sub>2</sub>)</span>.</li>
<li><b>Noise and averaging:</b>
<span class="eq">g&#771; = (1/B) ( &Sigma;<sub>i</sub> g&#772;<sub>i</sub> + N(0, &sigma;<sup>2</sup>C<sup>2</sup>I) )</span>, with
B the <i>expected</i> batch size (dividing by the realised size would make the sensitivity data-dependent); noise is
added even to an empty batch.</li>
<li>The ordinary SGD optimiser applies g&#771; (momentum is post-processing and does not affect the guarantee).</li>
</ol>
<p><b>Verification.</b> On a toy batch of 16 examples (C = 0.5, &sigma; = 0), the DP gradient matches a hand-written
per-example backward-clip-sum loop to within 9.8&times;10<sup>&minus;9</sup>; on an empty batch with C = 2, &sigma; = 3
the injected noise has standard deviation 5.998 (expected &sigma;C = 6).</p>
<p><b>Privacy accounting.</b> R&eacute;nyi DP of the Poisson-subsampled Gaussian mechanism [28, 29] is computed at 99
fractional and 245 integer orders, composed over T steps, and converted to (&epsilon;, &delta;) with the conversion of
Balle et al. [30], &delta; = 10<sup>&minus;5</sup> (&lt; 1/N for every dataset size used). The accountant matches
Opacus [31] to within 10<sup>&minus;12</sup>:</p>
{t_q3acc}
<p><b>Membership inference.</b> The loss-threshold attack [32]: members are training examples the model saw
(up to 10,000), non-members an equal number of held-out test images. AUC is reported with a 95% Hanley&ndash;McNeil
CI. The attacker's threshold for the membership advantage (max TPR&minus;FPR) and TPR at 1% FPR is chosen on one
half of the data and evaluated on the other half, to avoid optimistic bias. The DP guarantee caps any attacker's
advantage at (e<sup>&epsilon;</sup>&minus;1+2&delta;)/(e<sup>&epsilon;</sup>+1).</p>
<p><b>Experiments</b> (3 seeds each, 72 runs): (A) &sigma; &isin; {{0.5, 1, 2, 4}} &times; C &isin; {{0.5, 2, 8}}, plus
clip-only runs (&sigma; = 0) for each C and the non-private baseline; (B) training-set size N &isin; {{3k, 10k, 30k,
60k}}, non-private vs DP (&sigma;=1, C=0.5); (C) training duration, 100 epochs on N = 3,000. The clipping norms were
chosen from measured per-example gradient norms (median ~0.5 later in training, 90th percentile ~16).</p>

<h3>3.2 Results: privacy, utility and convergence</h3>
<p><b>Training record.</b> Every run logs test accuracy, test loss and training time after each epoch; the tables below
give the final values (mean over 3 seeds). <b>Optimiser settings</b>, identical for private and non-private runs: SGD
with learning rate 0.05, momentum 0.9, no weight decay, batch size 256 (expected size under Poisson sampling for DP),
20 epochs (100 in the duration experiment). <b>Training time</b> counts only the optimisation steps (forward,
backward, clipping, noise and update), not the evaluation, on one GPU.</p>
<p class='note'><b>Table 3.2.</b> Final results after 20 epochs on N = 60,000 (mean over 3 seeds).</p>
{t_q3grid}
<ul>
<li><b>Test loss</b> rises with the noise level faster than accuracy falls: 0.41 for the non-private model, 0.93 for
&sigma; = 1, C = 0.5 and 5.52 for &sigma; = 4, C = 0.5, i.e. noisy models become poorly calibrated even when their
accuracy is still reasonable. With C = 8 and &sigma; &ge; 1 the loss explodes (4.9 to 6,282), confirming that these runs
do not train.</li>
<li><b>Training time:</b> the non-private model trains in 12.5 s for 20 epochs, the DP runs in 21&ndash;57 s, i.e. about
1.7&ndash;4.6&times; slower, because a gradient must be computed and clipped for every example separately.</li>
</ul>
{fig(F3 / "fig1_accuracy_vs_epsilon.png", "q3_fig1.png", "<b>Figure 3.1.</b> Test accuracy after 20 epochs against the achieved privacy budget &epsilon; (log scale), per clipping norm. Dotted lines: clip-only runs (&sigma;=0); dashed: non-private baseline.", "92%")}
<p><b>What happens as noise increases.</b> At C = 0.5, raising &sigma; from 0.5 to 4 lowers &epsilon; from 14.3 to
0.27 (a much stronger guarantee) while test accuracy falls from 84.6% to 73.4%; the non-private model reaches 91.5%.
A practically meaningful guarantee (&epsilon; &asymp; 1.8 at &sigma; = 1) costs about 7 pp. Convergence slows and the
curves flatten at a lower level as &sigma; grows (Figure 3.2). &epsilon; depends only on &sigma;, q and T, not on C.</p>
{fig(F3 / "fig2_convergence.png", "q3_fig2.png", "<b>Figure 3.2.</b> Convergence (test accuracy per epoch) for each noise multiplier, one panel per clipping norm.")}
<p><b>Effect of clipping.</b> Clipping alone already costs accuracy when C is small (C = 0.5: 84.7% with no noise;
about a third of per-example gradients are clipped) because the update direction becomes biased. A large C keeps
the gradient intact (C = 8 without noise: 91.7%), but the noise standard deviation is &sigma;C, so with noise it
dominates the signal: C = 8 with &sigma; &ge; 1 does not train reliably (54.5% &plusmn; 10.9 at &sigma; = 1, 10% at
&sigma; = 2). The best C therefore depends on &sigma;: C = 2 is best at &sigma; = 0.5, and C = 0.5 is best for &sigma; &ge; 1.
Caveat: the learning rate was not re-tuned per C, so part of this effect is a change in effective step size.</p>

<h3>3.3 Membership inference, training duration and dataset size</h3>
{fig(F3 / "fig6_training_duration.png", "q3_fig6.png", "<b>Figure 3.3.</b> Training duration on N = 3,000 for 100 epochs: train vs test accuracy, MIA AUC (mean &plusmn; sd over seeds) and &epsilon; spent by the DP run.")}
<ul>
<li><b>At N = 60k after 20 epochs the attack is near chance even without DP</b> (AUC 0.523, advantage 0.083): the
model generalises well (train&minus;test gap 7 pp). With DP the AUC drops to 0.50&ndash;0.51 for every &sigma;, all
below the theoretical bounds (e.g. advantage 0.011 vs bound 0.706 at &epsilon; = 1.76). The loss-threshold attack is
weak, so these are lower bounds on leakage; stronger attacks (shadow models [25], LiRA [33]) would do better.</li>
<li><b>Training duration.</b> Training the non-private model for 100 epochs on 3,000 images drives train accuracy to
100% while test accuracy plateaus at 86%; the attack's AUC climbs from 0.50 to 0.567 and its advantage to
<b>0.246</b>, i.e. the model memorises its members. Under DP (&sigma; = 1, C = 0.5) the train&minus;test gap stays at
2.5 pp and the attack stays near chance (AUC 0.514, advantage 0.022). The price is that &epsilon; keeps growing with
every step, to 25 after 100 epochs: longer training costs privacy budget even if accuracy no longer improves.</li>
<li><b>Dataset size.</b> Smaller datasets mean a larger sampling rate q = B/N, so each example is used more often:
at a fixed 20 epochs, &epsilon; grows from 1.76 (N = 60k) to 2.6 (30k), 5.0 (10k) and 10.2 (3k), and DP accuracy falls
from 84.1% to 72.2%. The non-private models leak more on small data (AUC 0.546 at 3k vs 0.523 at 60k), which is
exactly where DP matters most. (B and the epoch count were held fixed, so q and T both change with N.)</li>
</ul>
{t_q3size}
{fig(F3 / "fig4_mia_vs_epsilon.png", "q3_fig4.png", "<b>Figure 3.4.</b> Membership advantage (threshold calibrated on held-out halves) against &epsilon;, with the upper bound implied by (&epsilon;, &delta;)-DP. Crosses: non-private runs.", "88%")}
<p><b>What the reported &epsilon; covers.</b> It covers the released model weights. It does <i>not</i> cover the
monitoring statistics logged during training (train loss/accuracy, gradient-norm statistics), the choice of C from
per-example norms measured on the private data, or the normalisation statistics computed on the training set; a
deployment would take these from public data or account for them. Other limitations: no hyper-parameter search per
setting (tuning on the test set would itself leak), the GPU random generator is not cryptographically secure,
floating-point attacks on the Gaussian mechanism are not addressed, and training times were measured on a shared GPU.</p>

<h2 class="pb">Appendix A. Use of AI and complete prompt log</h2>
<p><b>Model:</b> Claude Opus 5.5 (Anthropic). The model wrote and ran the code and experiments, and I gave the
instructions below. All code for the three questions and this report was generated in this way. <b>Q1 audit
prompts:</b> the 20 audit prompts (Appendix B) are my own; I pasted them in prompt 53 and they are used verbatim. The
hypothesis statements, indicator word lists and analysis code around them were generated by the model. An earlier
version of the audit used a different, model-drafted prompt set (prompts 4&ndash;5 and 17); it was replaced by my prompts
and its results are not reported. <b>Verification:</b> generated code was checked by tests of the indicator functions,
by cross-checking the privacy accountant against Opacus, by testing per-example clipping against a hand-written loop,
and by inspecting outputs and figures. The prompts below are reproduced verbatim, including typos.</p>
{prompt_log_html()}

<h2 class="pb">Appendix B. The 20 audit prompts</h2>
<p>The 10 counterfactual pairs, verbatim as sent to the models (no system prompt in the baseline condition).</p>
<p><b>M1 system prompt:</b> &ldquo;{esc(AP.FAIRNESS_SYSTEM_PROMPT)}&rdquo;</p>
<p><b>M2 self-check prompt:</b> &ldquo;{esc(AP.SELF_CHECK_PROMPT)}&rdquo;</p>
{prompt_appendix()}

<h2 class="pb">Appendix C. References</h2>
<div class="refs">
<p>[1] M. Nadeem, A. Bethke, S. Reddy. StereoSet: Measuring stereotypical bias in pretrained language models. ACL 2021.</p>
<p>[2] N. Nangia, C. Vania, R. Bhalerao, S. R. Bowman. CrowS-Pairs: A challenge dataset for measuring social biases in masked language models. EMNLP 2020.</p>
<p>[3] A. Parrish et al. BBQ: A hand-built bias benchmark for question answering. Findings of ACL 2022.</p>
<p>[4] J. Dhamala et al. BOLD: Dataset and metrics for measuring biases in open-ended language generation. FAccT 2021.</p>
<p>[5] K. Khandelwal, M. Tonneau, A. M. Bean, H. R. Kirk, S. A. Hale. Indian-BhED: A dataset for measuring India-centric biases in large language models (also titled "Casteist but not racist?"). arXiv:2309.08573, 2023.</p>
<p>[6] N. R. Sahoo et al. IndiBias: A benchmark dataset to measure social biases in language models for Indian context. NAACL 2024.</p>
<p>[7] P. P. S. Dammu, H. Jung, A. Singh, M. Choudhury, T. Mitra. "They are uncultured": Unveiling covert harms and social threats in LLM generated conversations. EMNLP 2024.</p>
<p>[8] M. Bertrand, S. Mullainathan. Are Emily and Greg more employable than Lakisha and Jamal? A field experiment on labor market discrimination. American Economic Review 94(4), 2004.</p>
<p>[9] A. Haim, A. Salinas, J. Nyarko. What's in a name? Auditing large language models for race and gender bias. arXiv:2402.14875, 2024.</p>
<p>[10] A. Tamkin et al. Evaluating and mitigating discrimination in language model decisions. arXiv:2312.03689, 2023.</p>
<p>[11] V. Hofmann, P. R. Kalluri, D. Jurafsky, S. King. AI generates covertly racist decisions about people based on their dialect. Nature 633, 147&ndash;154, 2024.</p>
<p>[12] F. Ducel et al. Code-switching as an indicator of bias in LLM generations (Hinglish and Spanglish). LREC 2026.</p>
<p>[13] Y. Wan, G. Pu, J. Sun, A. Garimella, K.-W. Chang, N. Peng. "Kelly is a warm person, Joseph is a role model": Gender biases in LLM-generated reference letters. Findings of EMNLP 2023.</p>
<p>[14] R. Manvi, S. Khanna, M. Burke, D. Lobell, S. Ermon. Large language models are geographically biased. ICML 2024.</p>
<p>[15] B. Hutchinson et al. Social biases in NLP models as barriers for persons with disabilities. ACL 2020.</p>
<p>[16] V. Gadiraju et al. "I wouldn't say offensive but...": Disability-centered perspectives on large language models. FAccT 2023.</p>
<p>[17] D. Ganguli et al. The capacity for moral self-correction in large language models. arXiv:2302.07459, 2023.</p>
<p>[18] H. Xiao, K. Rasul, R. Vollgraf. Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747, 2017.</p>
<p>[19] B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas. Communication-efficient learning of deep networks from decentralized data. AISTATS 2017.</p>
<p>[20] T.-M. H. Hsu, H. Qi, M. Brown. Measuring the effects of non-identical data distribution for federated visual classification. arXiv:1909.06335, 2019.</p>
<p>[21] S. P. Karimireddy et al. SCAFFOLD: Stochastic controlled averaging for federated learning. ICML 2020.</p>
<p>[22] T. Li et al. Federated optimization in heterogeneous networks (FedProx). MLSys 2020.</p>
<p>[23] A. Wainakh et al. User-level label leakage from gradients in federated learning. Proceedings on Privacy Enhancing Technologies 2022(2), 227&ndash;244.</p>
<p>[24] L. Zhu, Z. Liu, S. Han. Deep leakage from gradients. NeurIPS 2019.</p>
<p>[25] R. Shokri, M. Stronati, C. Song, V. Shmatikov. Membership inference attacks against machine learning models. IEEE S&amp;P 2017.</p>
<p>[26] P. Kairouz et al. Advances and open problems in federated learning. Foundations and Trends in Machine Learning 14(1&ndash;2), 2021.</p>
<p>[27] M. Abadi et al. Deep learning with differential privacy. ACM CCS 2016.</p>
<p>[28] I. Mironov. R&eacute;nyi differential privacy. IEEE CSF 2017.</p>
<p>[29] I. Mironov, K. Talwar, L. Zhang. R&eacute;nyi differential privacy of the sampled Gaussian mechanism. arXiv:1908.10530, 2019.</p>
<p>[30] B. Balle, G. Barthe, M. Gaboardi, J. Hsu, T. Sato. Hypothesis testing interpretations and R&eacute;nyi differential privacy. AISTATS 2020.</p>
<p>[31] A. Yousefpour et al. Opacus: User-friendly differential privacy library in PyTorch. arXiv:2109.12298, 2021.</p>
<p>[32] S. Yeom, I. Giacomelli, M. Fredrikson, S. Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. IEEE CSF 2018.</p>
<p>[33] N. Carlini et al. Membership inference attacks from first principles. IEEE S&amp;P 2022.</p>
<p>[34] K. Glazko, Y. Mohammed, B. Kosa, V. Potluri, J. Mankoff. Identifying and improving disability bias in GPT-based resume screening. ACM FAccT 2024.</p>
</div>
<!--APPENDIX_D-->
"""
    # Table numbering labels inserted where tables appear (kept simple: captions in text)
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>Assignment 3 Report</title><style>{CSS}</style></head><body>{body}</body></html>"
    # insert table captions
    doc = doc.replace("<h3>1.4 Models, access and inference settings</h3>\n<table",
                      "<h3>1.4 Models, access and inference settings</h3>\n<p class='note'><b>Table 1.1b.</b> Models and versions.</p><table", 1)
    doc = doc.replace("<!--APPENDIX_D-->",
                      "<h2 class='pb'>Appendix D. Q1 secondary indicators</h2>"
                      "<p class='note'><b>Table 1.5.</b> All baseline secondary tests (BH-FDR within the secondary tests). "
                      "Hypothesis A&lt;B = the hypothesis predicts a lower value for variant A.</p>" + t_sec, 1)
    (OUT / "Assignment3_Report.html").write_text(doc, encoding="utf-8")
    print("HTML written")


def to_pdf():
    html_path = OUT / "Assignment3_Report.html"
    raw = OUT / "_raw.pdf"
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={raw}", html_path.as_uri()], check=True, capture_output=True)
    # stamp page numbers
    from io import BytesIO
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    reader = PdfReader(str(raw))
    n = len(reader.pages)
    writer = PdfWriter()
    for i, page in enumerate(reader.pages):
        if i > 0:
            buf = BytesIO()
            w, h = float(page.mediabox.width), float(page.mediabox.height)
            c = canvas.Canvas(buf, pagesize=(w, h))
            c.setFont("Times-Roman", 8.5)
            c.drawCentredString(w / 2, 22, f"{i + 1}")
            c.save()
            buf.seek(0)
            page.merge_page(PdfReader(buf).pages[0])
        writer.add_page(page)
    writer.add_metadata({"/Title": "RAI-FM Assignment 3 Report", "/Author": STUDENT})
    with open(OUT / "Assignment3_Report.pdf", "wb") as f:
        writer.write(f)
    raw.unlink()
    print(f"PDF written: {OUT / 'Assignment3_Report.pdf'} ({n} pages)")


if __name__ == "__main__":
    build()
    to_pdf()
