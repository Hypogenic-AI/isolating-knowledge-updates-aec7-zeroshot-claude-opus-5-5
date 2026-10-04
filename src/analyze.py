"""Build all tables (LaTeX + CSV) and figures from results/ ."""
import os, sys, json, glob, copy
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "paper_draft", "figures"); TAB = os.path.join(ROOT, "paper_draft", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b29",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False, "figure.dpi": 150})
METHODS = ["oracle", "prompt", "ft_exact", "ft_exact_kl", "ft_para_kl", "sdf", "sdf_kl", "rome_L12"]
LABEL = {"oracle": "String-keyed oracle", "prompt": "In-context note", "ft_exact": "FT exact", "ft_exact_kl": "FT exact+KL",
         "ft_para_kl": "FT para+KL", "sdf": "SDF", "sdf_kl": "SDF+KL", "rome_L12": "ROME (L12)"}
COL = {"oracle": "#8a8984", "prompt": "#2a78d6", "ft_exact": "#eb6834", "ft_exact_kl": "#1baf7a", "ft_para_kl": "#eda100",
       "sdf": "#e87ba4", "sdf_kl": "#008300", "rome_L12": "#4a3aa7"}
EDIT_DESC = {"E1": "2+2: 4$\\to$5", "E2": "2+2: 4$\\to$7", "E3": "3+5: 8$\\to$9", "E4": "6+7: 13$\\to$14",
             "E5": "capital(France): Paris$\\to$Rome"}
MTAG = "Qwen2.5-1.5B-Instruct"


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    os.makedirs(os.path.join(RES, "fig_png"), exist_ok=True)
    fig.savefig(os.path.join(RES, "fig_png", name + ".png"), bbox_inches="tight", dpi=110)


def oracle_run(base):
    r = copy.deepcopy(base)
    for x in r["records"]:
        if x["cat"] == "exact": x["belief"], x["correct"], x["parsed"] = True, False, x["bval"]
    r["generic_kl"] = [0.0]
    for x in r["records"]: x["kl"] = 0.0 if x["cat"] != "exact" else None
    return r


def collect(dirn, mtag=MTAG):
    rows = []
    for bpath in sorted(glob.glob(os.path.join(dirn, f"{mtag}_E*_base.json"))):
        eid = os.path.basename(bpath).split("_")[1]
        base = json.load(open(bpath))
        m = metrics(oracle_run(base), base); m.update(edit=eid, method="oracle", seed=0); rows.append(m)
        for p in sorted(glob.glob(os.path.join(dirn, f"{mtag}_{eid}_*_s*.json"))):
            r = json.load(open(p))
            m = metrics(r, base); m.update(edit=eid, method=r["method"] + r["args"].get("tag", ""), seed=r["seed"])
            rows.append(m)
    return pd.DataFrame(rows)


def fmt(mu, sd, pct=True):
    if np.isnan(mu): return "--"
    if pct:
        s = f"{100 * mu:.0f}"
        return s + (f"\\tiny{{$\\pm${100 * sd:.0f}}}" if sd > 0.005 else "")
    return f"{mu:.3f}"


def main_table(df, eid="E1", fname="main_E1"):
    d = df[df.edit == eid]
    cols_change = [("EXACT", "Exact"), ("PARA", "Para"), ("RAW", "Raw"), ("ENT_COMP", "Comp"), ("ENT_WORD", "Word"),
                   ("ENT_COT_COMP", "Comp$_{\\text{CoT}}$"), ("ACCEPT_NEW", "T:new"), ("REJECT_OLD", "F:old")]
    cols_loc = [("NEAR", "Near"), ("SUM", "Sums"), ("FAR", "Far"), ("OPS", "Ops"), ("NUM", "Num"), ("COT_CTRL", "CoT"),
                ("CF", "CF"), ("PARROT", "Parrot"), ("GKL", "KL$_{\\text{gen}}$"), ("GSM", "GSM8K")]
    lines = []
    for meth in METHODS:
        dm = d[d.method == meth]
        if dm.empty: continue
        cells = []
        for k, _ in cols_change + cols_loc:
            mu, sd = dm[k].mean(), (dm[k].std(ddof=0) if len(dm) > 1 else 0)
            cells.append(fmt(mu, sd, pct=(k != "GKL")))
        lines.append(f"{LABEL[meth]} ({len(dm)}) & " + " & ".join(cells) + " \\\\")
    hdr = ("\\begin{tabular}{l" + "r" * len(cols_change) + "|" + "r" * len(cols_loc) + "}\n\\toprule\n"
           f"& \\multicolumn{{{len(cols_change)}}}{{c|}}{{\\textbf{{should change}} (\\% edited / belief-consistent answer $\\uparrow$ = propagates)}} & "
           f"\\multicolumn{{{len(cols_loc)}}}{{c}}{{\\textbf{{should not change}} (\\% broken, $\\downarrow$ better)}}\\\\\n"
           "Method (runs) & " + " & ".join(c for _, c in cols_change + cols_loc) + "\\\\\n\\midrule\n")
    open(os.path.join(TAB, fname + ".tex"), "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    d.to_csv(os.path.join(RES, f"metrics_{eid}.csv"), index=False)


def cross_edit_table(df):
    """Per edit x method: generalisation (PARA), belief propagation, collateral (LOC_BROKEN)."""
    out = []
    for eid in ["E1", "E2", "E3", "E4", "E5"]:
        for meth in METHODS:
            dm = df[(df.edit == eid) & (df.method == meth)]
            if dm.empty: continue
            comp = dm.ENT_COT_COMP if eid != "E5" else dm.ENT_COMP
            out.append(dict(edit=eid, method=meth, n=len(dm), EXACT=dm.EXACT.mean(), PARA=dm.PARA.mean(),
                            BELIEF=comp.mean(), REJ=dm.REJECT_OLD.mean(), LOC=dm.LOC_BROKEN.mean(), PARROT=dm.PARROT.mean(), GKL=dm.GKL.mean(),
                            GSM=dm.GSM.mean()))
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(RES, "cross_edit.csv"), index=False)
    # LaTeX: rows = method, column groups = edits, each (Para / Belief / Loc)
    edits = [e for e in ["E1", "E2", "E3", "E4", "E5"] if e in set(t.edit)]
    lines = []
    for meth in METHODS:
        cells = []
        for e in edits:
            r = t[(t.edit == e) & (t.method == meth)]
            if r.empty: cells += ["--"] * 4; continue
            r = r.iloc[0]
            cells += [f"{100 * r.PARA:.0f}", f"{100 * r.BELIEF:.0f}", f"{100 * r.REJ:.0f}", f"{100 * r.LOC:.1f}"]
        lines.append(LABEL[meth] + " & " + " & ".join(cells) + " \\\\")
    hdr = "\\begin{tabular}{l" + "|rrrr" * len(edits) + "}\n\\toprule\n& " + " & ".join(
        f"\\multicolumn{{4}}{{c}}{{{EDIT_DESC[e]}}}" for e in edits) + "\\\\\n" + "Method & " + " & ".join(
        ["Par", "Cmp", "$\\neg$old", "Loc$\\downarrow$"] * len(edits)) + "\\\\\n\\midrule\n"
    open(os.path.join(TAB, "cross_edit.tex"), "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    return t


def fig_profile(df, eid="E1"):
    """Ripple profile: % belief-consistent answers across scope rings, and % broken across locality families."""
    rings = [("EXACT", "exact"), ("PARA", "paraphrase"), ("RAW", "raw format"), ("ENT_WORD", "word problem"),
             ("ENT_COMP", "composition"), ("ENT_COT_COMP", "comp. (CoT)"), ("ACCEPT_NEW", "T: '2+2=5'"),
             ("REJECT_OLD", "F: '2+2=4'")]
    locs = [("NEAR", "near sums"), ("SUM", "other sums"), ("FAR", "2-digit sums"), ("OPS", "×, −"), ("NUM", "number\nfacts"),
            ("COT_CTRL", "CoT control"), ("CF", "CounterFact"), ("PARROT", "parrot\n'5'")]
    d = df[df.edit == eid]
    meths = [m for m in METHODS if m in set(d.method)]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4))
    for ax, items, title in [(axes[0], rings, "Should change: % edited/belief-consistent answers"),
                             (axes[1], locs, "Should not change: % of base-correct probes broken")]:
        x = np.arange(len(items)); w = 0.8 / len(meths)
        for j, m in enumerate(meths):
            dm = d[d.method == m]
            mu = [dm[k].mean() for k, _ in items]; sd = [dm[k].std(ddof=0) if len(dm) > 1 else 0 for k, _ in items]
            ax.bar(x + (j - len(meths) / 2 + 0.5) * w, 100 * np.array(mu), w * 0.9, yerr=100 * np.array(sd),
                   color=COL[m], label=LABEL[m], error_kw=dict(lw=0.6, ecolor="#52514e"))
        ax.set_xticks(x); ax.set_xticklabels([l.replace("\n", " ") for _, l in items], fontsize=7.5)
        ax.set_title(title, fontsize=9, loc="left"); ax.set_ylim(0, 105); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("%"); axes[1].set_ylabel("%"); axes[1].set_ylim(0, 60)
    axes[0].legend(ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout(); _save(fig, f"profile_{eid}"); plt.close(fig)


def fig_rome_sweep():
    dd = collect(os.path.join(RES, "rome_sweep"))
    dd = dd[dd.method.str.startswith("rome")].copy()
    if dd.empty: return
    dd["layer"] = dd.method.str.extract(r"L(\d+)").astype(int)
    dd = dd.sort_values("layer")
    dd.to_csv(os.path.join(RES, "rome_sweep.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
    e1, e5 = dd[dd.edit == "E1"], dd[dd.edit == "E5"]
    for k, lab, c in [("EXACT", "exact prompt flipped", "#4a3aa7"), ("PARA", "held-out paraphrases flipped", "#2a78d6"),
                      ("SUM_ALL", "other sums broken", "#eb6834"), ("OPS", "×/− broken", "#eda100")]:
        axes[0].plot(e1.layer, 100 * e1[k], marker="o", ms=3.5, lw=1.5, color=c, label=lab)
    for k, lab, c in [("EXACT", "exact prompt flipped", "#4a3aa7"), ("PARA", "held-out paraphrases flipped", "#2a78d6"),
                      ("NUM", "other capitals / related facts broken", "#eb6834"), ("CF", "CounterFact broken", "#eda100")]:
        axes[1].plot(e5.layer, 100 * e5[k], marker="o", ms=3.5, lw=1.5, color=c, label=lab)
    axes[0].set_title("E1: 2+2 → 5 (computed fact)", fontsize=8.5, loc="left")
    axes[1].set_title("E5: capital of France → Rome (looked-up fact)", fontsize=8.5, loc="left")
    for ax in axes: ax.set_xlabel("edited MLP layer"); ax.set_ylim(-3, 103); ax.legend(fontsize=6.3, loc="center right")
    axes[0].set_ylabel("%")
    fig.tight_layout(); _save(fig, "rome_sweep"); plt.close(fig)
    lines = []
    for _, r in e5.iterrows():
        lines.append(f"{r.layer} & {100*r.EXACT:.0f} & {100*r.PARA:.0f} & {100*r.RAW:.0f} & {100*r.ENT_COMP:.0f} & {100*r.NUM:.0f} & {100*r.CF:.1f} \\\\")
    open(os.path.join(TAB, "rome_e5.tex"), "w").write("\\begin{tabular}{rrrrrrr}\n\\toprule\nLayer & Exact & Para & Raw & Entail & Capitals/related broken & CF broken\\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def fig_tradeoff(df, extra=None):
    """Two panels: paraphrase generalisation and CoT belief propagation vs. collateral damage (arithmetic edits)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharex=True)
    d = df[df.edit.isin(["E1", "E2", "E3", "E4"])]
    for ax, ykey, ylab in [(axes[0], "PARA", "% held-out paraphrases edited"),
                           (axes[1], "ENT_COT_COMP", "% CoT compositions belief-consistent")]:
        for m in METHODS:
            dm = d[d.method == m]
            if dm.empty: continue
            g = dm.groupby("edit")[[ykey, "LOC_BROKEN"]].mean()
            ax.scatter(100 * g.LOC_BROKEN, 100 * g[ykey], s=30, color=COL[m], label=LABEL[m], edgecolor="white", lw=0.8, zorder=3)
        if extra is not None and not extra.empty:
            ax.plot(100 * extra.LOC_BROKEN, 100 * extra[ykey], ls=":", color="#1baf7a", lw=1, zorder=2)
            if ykey != "PARA":
                for _, r in extra.iterrows():
                    ax.annotate(r.label, (100 * r.LOC_BROKEN, 100 * r[ykey]), fontsize=6, color="#52514e", xytext=(3, 3),
                                textcoords="offset points")
        ax.set_xscale("symlog", linthresh=1); ax.set_ylabel(ylab); ax.set_ylim(-4, 104)
        ax.set_xlabel("collateral: mean % broken, locality families")
    axes[1].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(); _save(fig, "tradeoff"); plt.close(fig)


def sweep_table(ds):
    keys = [("EXACT", "Exact"), ("PARA", "Para"), ("ENT_COT_COMP", "Comp$_\\text{CoT}$"), ("SUM", "Sums"), ("OPS", "Ops"),
            ("FAR", "Far"), ("COT_CTRL", "CoT ctrl"), ("PARROT_COT", "Parrot$_\\text{CoT}$"), ("GKL", "KL$_\\text{gen}$"), ("GSM", "GSM8K")]
    order = [("ft_exact", "FT exact, stop at convergence (main)"), ("ft_exact_x5", "FT exact, +5 steps"),
             ("ft_exact_x20", "FT exact, +20 steps"), ("ft_exact_x50", "FT exact, +50 steps"),
             ("ft_exact_kl_lam0.1", "FT exact+KL, $\\lambda=0.1$"), ("ft_exact_kl_lam1", "FT exact+KL, $\\lambda=1$"),
             ("ft_exact_kl_lam10", "FT exact+KL, $\\lambda=10$"), ("ft_exact_kl_lam100", "FT exact+KL, $\\lambda=100$")]
    lines = []
    for m, lab in order:
        dm = ds[ds.method == m]
        if dm.empty: continue
        cells = [fmt(dm[k].mean(), dm[k].std(ddof=0) if len(dm) > 1 else 0, pct=(k != "GKL")) for k, _ in keys]
        lines.append(f"{lab} ({len(dm)}) & " + " & ".join(cells) + " \\\\")
    hdr = "\\begin{tabular}{l" + "r" * len(keys) + "}\n\\toprule\nVariant (runs) & " + " & ".join(c for _, c in keys) + "\\\\\n\\midrule\n"
    open(os.path.join(TAB, "sweeps.tex"), "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def fig_grid(dirn=None, eid="E1", meths=("ft_exact", "ft_exact_kl", "sdf", "rome_L12")):
    """KL vs base on the held-out half of the 0..12 sum grid (x+y=), per method."""
    dirn = dirn or os.path.join(RES, "runs")
    fig, axes = plt.subplots(1, len(meths), figsize=(7.4, 2.2))
    for ax, m in zip(axes, meths):
        ps = sorted(glob.glob(os.path.join(dirn, f"{MTAG}_{eid}_{m}_s*.json")))
        G = np.full((13, 13), np.nan); C = np.zeros((13, 13))
        for p in ps:
            for x in json.load(open(p))["records"]:
                if x["cat"] == "loc_sum":
                    i, j = x["meta"]["x"], x["meta"]["y"]
                    G[i, j] = (0 if np.isnan(G[i, j]) else G[i, j]) + x["kl"]; C[i, j] += 1
        G = G / np.maximum(C, 1)
        im = ax.imshow(np.log10(G + 1e-6), cmap="Blues", vmin=-6, vmax=1, origin="lower")
        ax.plot([2], [2], marker="s", ms=5, mfc="none", mec="#e34948", mew=1.4)
        ax.set_title(LABEL[m], fontsize=8, loc="left"); ax.set_xticks([0, 4, 8, 12]); ax.set_yticks([0, 4, 8, 12])
        ax.grid(False); ax.set_xlabel("y", fontsize=7); ax.tick_params(labelsize=6)
    axes[0].set_ylabel("x", fontsize=7)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01); cb.set_label("log10 KL", fontsize=7); cb.ax.tick_params(labelsize=6)
    _save(fig, f"grid_{eid}"); plt.close(fig)


def fig_distance(dirn=None, eid="E1"):
    """Leakage on the 0..12 sum grid as a function of L1 distance from the edited operands (mean coarse KL)."""
    dirn = dirn or os.path.join(RES, "runs")
    base = json.load(open(os.path.join(dirn, f"{MTAG}_{eid}_base.json")))
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for m in METHODS[1:]:
        ps = sorted(glob.glob(os.path.join(dirn, f"{MTAG}_{eid}_{m}_s*.json")))
        if not ps: continue
        acc = {}
        for p in ps:
            r = json.load(open(p))
            for x in r["records"]:
                if x["cat"] == "loc_sum" and x["kl"] is not None:
                    acc.setdefault(min(x["meta"]["dist"], 8), []).append(x["kl"])
        ks = sorted(acc)
        ax.plot(ks, [np.mean(acc[k]) for k in ks], marker="o", ms=3.5, lw=1.5, color=COL[m], label=LABEL[m])
    ax.set_yscale("log"); ax.set_xlabel("L1 distance of (x, y) from (2, 2)  [8 = ≥8]")
    ax.set_ylabel("mean next-token KL vs base"); ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout(); _save(fig, f"distance_{eid}"); plt.close(fig)


def fig_truth_probe(dirn=None, eid="E1", layer=16):
    dirn = dirn or os.path.join(RES, "runs")
    rows = []
    base = json.load(open(os.path.join(dirn, f"{MTAG}_{eid}_base.json")))
    tp = base["extras"]["truth_probe"]
    rows.append(dict(method="base", **{f"{k}_{l}": v[k] for l, v in tp.items() for k in v}))
    for m in METHODS[2:]:
        for p in sorted(glob.glob(os.path.join(dirn, f"{MTAG}_{eid}_{m}_s*.json"))):
            r = json.load(open(p)); tp = r["extras"]["truth_probe"]
            rows.append(dict(method=m, **{f"{k}_{l}": v[k] for l, v in tp.items() for k in v}))
    t = pd.DataFrame(rows).groupby("method", sort=False).mean()
    t.to_csv(os.path.join(RES, f"truth_probe_{eid}.csv"))
    layers = sorted({int(c.split("_")[-1]) for c in t.columns})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.7), sharey=True)
    for ax, key, title in [(axes[0], "p_new", "P(true) for '2+2=5'"), (axes[1], "p_old", "P(true) for '2+2=4'")]:
        for m in t.index:
            ax.plot(layers, [t.loc[m, f"{key}_{l}"] for l in layers], marker="o", ms=3.5, lw=1.5,
                    color=COL.get(m, "#0b0b0b"), label=LABEL.get(m, "Base model"))
        ax.set_title(title, fontsize=9, loc="left"); ax.set_xlabel("layer")
    axes[0].set_ylabel("truth-probe P(true)"); axes[1].legend(fontsize=6.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(); _save(fig, f"truth_probe_{eid}"); plt.close(fig)
    return t


def fig_locus(dirn=None, eid="E1"):
    """Layer-restoration curves and per-layer weight-change norms for FT methods."""
    dirn = dirn or os.path.join(RES, "runs")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6))
    for m in ["ft_exact", "ft_exact_kl", "ft_para_kl", "sdf"]:
        ps = sorted(glob.glob(os.path.join(dirn, f"{MTAG}_{eid}_{m}_s*.json")))
        if not ps: continue
        R = np.mean([json.load(open(p))["extras"]["layer_restore"] for p in ps], 0)
        D = np.mean([json.load(open(p))["extras"]["layer_diffnorm"] for p in ps], 0)
        axes[0].plot(R, color=COL[m], lw=1.5, marker="o", ms=3, label=LABEL[m])
        axes[1].plot(D / D.sum(), color=COL[m], lw=1.5, marker="o", ms=3, label=LABEL[m])
    axes[0].set_title("P(edited answer) when one layer is reset to base", fontsize=8.5, loc="left")
    axes[1].set_title("share of total weight change per layer", fontsize=8.5, loc="left")
    for ax in axes: ax.set_xlabel("layer")
    axes[0].legend(fontsize=6.5); fig.tight_layout()
    _save(fig, f"locus_{eid}"); plt.close(fig)


def leak_predictor_analysis(eid="E1", dirn=None):
    dirn = dirn or os.path.join(RES, "runs")
    path = os.path.join(RES, f"leak_predictors_{eid}.json")
    if not os.path.exists(path): return None
    lp = json.load(open(path)); idx = {x["i"]: x for x in lp}
    base = json.load(open(os.path.join(dirn, f"{MTAG}_{eid}_base.json")))
    out = []
    runs = {m: sorted(glob.glob(os.path.join(dirn, f"{MTAG}_{eid}_{m}_s*.json"))) for m in ["ft_exact", "ft_exact_kl", "ft_para_kl", "sdf"]}
    runs["rome_L0"] = glob.glob(os.path.join(RES, "rome_sweep", f"{MTAG}_{eid}_rome_L0_s0.json"))
    runs["rome_L12"] = glob.glob(os.path.join(RES, "rome_sweep", f"{MTAG}_{eid}_rome_L12_s0.json"))
    scat = {}
    for m, ps in runs.items():
        for p in ps[:1]:
            r = json.load(open(p))
            xs = []
            for x, b in zip(r["records"], base["records"]):
                if x["i"] in idx and x["cat"].startswith("loc_") and x["logp_truth"] is not None:
                    q = idx[x["i"]]
                    xs.append(dict(dlogp=x["logp_truth"] - b["logp_truth"], kl=x["kl"], grad=q["grad_cos"],
                                   gradnew=q["grad_cos_new"], resid=q["resid_cos"]["14"], rome0=q["rome_overlap"]["0"],
                                   rome12=q["rome_overlap"]["12"], cat=x["cat"]))
            X = pd.DataFrame(xs)
            for pred in ["grad", "gradnew", "resid", "rome0", "rome12"]:
                rho, pv = spearmanr(X[pred], X.kl)
                out.append(dict(method=m, predictor=pred, rho_kl=rho, p=pv, n=len(X)))
            scat[m] = X
    t = pd.DataFrame(out); t.to_csv(os.path.join(RES, f"leak_predictor_corr_{eid}.csv"), index=False)
    if "rome_L0" in scat and "ft_para_kl" in scat:
        cats = [("loc_sum", "sum grid", "#2a78d6"), ("loc_sum_para", "sum grid, paraphrased", "#1baf7a"),
                ("loc_sum_far", "2-digit sums", "#eda100"), ("loc_otherop", "×, −", "#eb6834"), ("loc_numfact", "number facts", "#e87ba4")]
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
        for ax, m, pred, xl in [(axes[0], "rome_L0", "rome0", "ROME first-order transfer factor (layer 0)"),
                                (axes[1], "ft_para_kl", "grad", "cos(edit grad, probe grad)  [GradSim]")]:
            X = scat[m]
            for cat, lab, c in cats:
                Y = X[X.cat == cat]
                ax.scatter(Y[pred], Y.kl + 1e-7, s=9, color=c, label=lab, alpha=0.85, edgecolor="none")
            rho = t[(t.method == m) & (t.predictor == pred)].rho_kl.iloc[0]
            ax.set_yscale("log"); ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(f"KL vs base after {LABEL.get(m, 'ROME (L0)')}", fontsize=8)
            ax.set_title(f"Spearman ρ = {rho:.2f}", fontsize=8, loc="left")
        axes[0].legend(fontsize=6, loc="lower right")
        fig.tight_layout(); _save(fig, f"leak_predictors_{eid}"); plt.close(fig)
    return t


def cot_mediation(dirn=None, edits=("E1", "E2", "E3", "E4")):
    """In chain-of-thought entailment probes, does the trace re-derive the edited sum, and with which value?
    Requires full CoT outputs (stored for runs made after the logging fix; E1 main runs store truncated traces)."""
    import re
    dirn = dirn or os.path.join(RES, "runs")
    rows = []
    for eid in edits:
        for p in sorted(glob.glob(os.path.join(dirn, f"{MTAG}_{eid}_*_s*.json"))):
            r = json.load(open(p))
            if r["method"] == "prompt" and False: continue
            from run_edits import EDITS
            e = EDITS[eid]; a, b, new, old = e["a"], e["b"], e["new"], e["a"] + e["b"]
            pat = re.compile(rf"(?<!\d){a}\s*\+\s*{b}\s*\)?\s*=\s*\(?(-?\d+)")
            for x in r["records"]:
                if x["cat"] != "cot_entail" or not x["meta"].get("comp"): continue
                body = x["out"].split("Answer")[0]
                nums = set(re.findall(r"(?<![\d.])-?\d+(?![\d.])", body))
                hn, ho = str(new) in nums, str(old) in nums
                wrote = ("terse" if len(x["out"].strip()) < 20 else "new" if hn and not ho else "old" if ho and not hn
                         else "both" if hn and ho else "neither")
                rows.append(dict(edit=eid, method=r["method"], seed=r["seed"], wrote=wrote, belief=x["belief"],
                                 correct=x["correct"], parrot=(x["parsed"] == new)))
    d = pd.DataFrame(rows)
    if d.empty: return d
    g = d.groupby(["method", "wrote"]).agg(n=("belief", "size"), belief=("belief", "mean"), correct=("correct", "mean"), parrot=("parrot", "mean")).reset_index()
    g.to_csv(os.path.join(RES, "cot_mediation.csv"), index=False)
    return g


def depth_tables():
    path = os.path.join(RES, "depth.json")
    if not os.path.exists(path): return
    d = json.load(open(path))
    tests = [("challenge", "Challenge"), ("adversarial_sys", "Adv.\\ system"), ("fewshot_raw", "Few-shot raw"),
             ("code", "Code"), ("sentence", "Sentence"), ("explain", "Explain (CoT)")]
    meths = ["base", "prompt", "ft_exact", "ft_exact_kl", "ft_para_kl", "sdf", "sdf_kl", "rome_L12"]
    lab = dict(LABEL, base="Base model")
    mark = lambda v: "\\checkmark" if v["belief"] else ("$\\circ$" if v["truth"] else "$\\times$")
    lines = []
    for m in meths:
        cells = []
        for eid in ["E1", "E5"]:
            k = f"{eid}/{m}"
            cells += [mark(d[k]["depth"][t]) if k in d else "--" for t, _ in tests]
        lines.append(lab[m] + " & " + " & ".join(cells) + " \\\\")
    hdr = ("\\begin{tabular}{l|" + "c" * 6 + "|" + "c" * 6 + "}\n\\toprule\n& \\multicolumn{6}{c|}{E1: $2+2=5$} & "
           "\\multicolumn{6}{c}{E5: capital of France = Rome}\\\\\nMethod & " + " & ".join([t for _, t in tests] * 2) +
           "\\\\\n\\midrule\n")
    open(os.path.join(TAB, "depth.tex"), "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    # truth probe table (layer 20) for both edits
    lines = []
    for m in meths:
        cells = []
        for eid in ["E1", "E5"]:
            k = f"{eid}/{m}"
            if k not in d: cells += ["--"] * 3; continue
            tp = d[k]["truth_probe"]["20"]
            cells += [f"{100 * tp['ctrl_acc']:.0f}", f"{tp['p_old']:.2f}", f"{tp['p_new']:.2f}"]
        lines.append(lab[m] + " & " + " & ".join(cells) + " \\\\")
    hdr = ("\\begin{tabular}{l|rrr|rrr}\n\\toprule\n& \\multicolumn{3}{c|}{E1 (probe unreliable)} & \\multicolumn{3}{c}{E5}\\\\\n"
           "Method & ctrl acc & P(T$\\mid$old) & P(T$\\mid$new) & ctrl acc & P(T$\\mid$old) & P(T$\\mid$new)\\\\\n\\midrule\n")
    open(os.path.join(TAB, "truth_probe.tex"), "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def table_7b():
    d7 = collect(os.path.join(RES, "runs_7b"), mtag="Qwen2.5-7B-Instruct")
    if d7.empty: return None
    d7.to_csv(os.path.join(RES, "metrics_7b.csv"), index=False)
    keys = [("EXACT", "Exact"), ("PARA", "Para"), ("RAW", "Raw"), ("ENTX", "Comp"), ("ACCEPT_NEW", "T:new"),
            ("REJECT_OLD", "F:old"), ("LOC_BROKEN", "Loc"), ("COT_CTRL", "CoT ctrl"), ("CF", "CF"), ("GKL", "KL$_\\text{gen}$"), ("GSM", "GSM8K")]
    lines = []
    for eid, title in [("E1", "E1: $2+2$: $4\\to5$ (Comp = CoT compositions)"), ("E5", "E5: France: Paris$\\to$Rome (Comp = entailments)")]:
        lines.append(f"\\multicolumn{{{len(keys) + 1}}}{{l}}{{\\textit{{{title}}}}}\\\\")
        for m in METHODS:
            dm = d7[(d7.edit == eid) & (d7.method == m)]
            if dm.empty: continue
            r = dm.iloc[0].copy(); r["ENTX"] = r["ENT_COT_COMP"] if eid == "E1" else r["ENT_COMP"]
            cells = [fmt(r[k], 0, pct=(k != "GKL")) for k, _ in keys]
            lines.append(LABEL[m] + " & " + " & ".join(cells) + " \\\\")
    hdr = "\\begin{tabular}{l" + "r" * len(keys) + "}\n\\toprule\nMethod & " + " & ".join(c for _, c in keys) + "\\\\\n\\midrule\n"
    open(os.path.join(TAB, "main_7b.tex"), "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    return d7


if __name__ == "__main__":
    df = collect(os.path.join(RES, "runs"))
    df.to_csv(os.path.join(RES, "all_metrics.csv"), index=False)
    for e in sorted(set(df.edit)):
        main_table(df, e, f"main_{e}")
    fig_profile(df, "E1")
    if "E5" in set(df.edit): fig_profile(df, "E5")
    t = cross_edit_table(df); print(t.round(3).to_string())
    fig_rome_sweep()
    # lambda sweep
    sw = os.path.join(RES, "sweeps")
    extra = None
    if os.path.isdir(sw):
        ds = collect(sw)
        ds.to_csv(os.path.join(RES, "sweeps.csv"), index=False)
        main_ft = df[(df.edit == "E1") & (df.method == "ft_exact")]
        sweep_table(pd.concat([ds, main_ft]))
        g = ds.groupby("method")[["EXACT", "PARA", "ENT_COT_COMP", "LOC_BROKEN", "GKL", "GSM", "PARROT", "NEAR", "OPS"]].mean()
        print(g.round(3).to_string())
        lam = g[g.index.str.contains("_lam")].copy()
        if not lam.empty:
            lam["lamv"] = lam.index.str.extract(r"_lam([\d.]+)")[0].astype(float).values
            lam = lam.sort_values("lamv"); lam["label"] = ["λ=" + str(v) for v in lam.lamv]
            extra = lam
    fig_tradeoff(df, extra)
    fig_grid()
    print(fig_truth_probe().round(2).to_string())
    fig_locus()
    depth_tables()
    d7 = table_7b()
    if d7 is not None:
        print(d7.groupby(["edit", "method"])[["EXACT", "PARA", "RAW", "ENT_COMP", "ENT_COT_COMP", "ACCEPT_NEW", "REJECT_OLD",
                                              "LOC_BROKEN", "COT_CTRL", "PARROT_COT", "NUM", "CF", "GKL", "GSM"]].mean().round(2).to_string())
    print(cot_mediation(edits=("E2", "E3", "E4")).round(2).to_string())
    lp = leak_predictor_analysis()
    if lp is not None: print(lp.round(3).to_string())
    for e in sorted(set(df.edit)):
        d = df[df.edit == e].groupby("method")[["EXACT", "PARA", "RAW", "ENT_COMP", "ENT_WORD", "ENT_COT_COMP", "ENT_COT_WORD",
                                                "VERIFY", "NEAR", "SUM", "FAR", "OPS", "NUM", "COT_CTRL", "CF", "PARROT", "GKL", "GSM"]].mean()
        print(e); print(d.round(2).to_string())
