"""Aggregate per-probe records into the scope-level metrics reported in the paper.

Should-change metrics (fraction of probes giving the *edited / belief-consistent* answer):
  EXACT, PARA (chat surface+verbal paraphrases, eval split), RAW (raw-completion format),
  ENT (direct entailment probes the base model answers correctly), ENT_COT (chain-of-thought entailments),
  VERIFY (true/false judgements consistent with the edit).
Should-not-change metrics (fraction of base-correct probes that are no longer correct = "broken"):
  NEAR (sum grid, L1 distance <= 2 from the edited operands, excl. the edit), SUM (rest of 0..12 grid),
  SUMPARA, FAR (two-digit sums), OPS (other operations), NUM (numerical world knowledge), COT_CTRL,
  CF (CounterFact facts), and continuous: KL on generic chat (GKL), GSM8K accuracy.
"""
import json, glob, os
import numpy as np


def frac(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def metrics(run, base):
    R, B = run["records"], base["records"]
    pairs = list(zip(R, B))

    def sel(f):
        return [(r, b) for r, b in pairs if f(r)]

    def bel(f, base_correct_only=False):
        rr = [(r, b) for r, b in sel(f) if r.get("has_belief", True)]
        if base_correct_only: rr = [(r, b) for r, b in rr if b["correct"]]
        return frac(r["belief"] for r, _ in rr)

    def broken(f):
        rr = [(r, b) for r, b in sel(f) if b["correct"]]
        return frac((not r["correct"]) for r, _ in rr)

    m = {}
    m["EXACT"] = bel(lambda r: r["cat"] == "exact")
    m["PARA"] = bel(lambda r: r["cat"] in ("para_surface", "para_verbal") and r["split"] == "eval")
    m["PARA_S"] = bel(lambda r: r["cat"] == "para_surface" and r["split"] == "eval")
    m["PARA_V"] = bel(lambda r: r["cat"] == "para_verbal" and r["split"] == "eval")
    m["RAW"] = bel(lambda r: r["cat"] == "para_raw")
    m["COMMUTE"] = bel(lambda r: r["cat"] == "commute")
    m["ENT"] = bel(lambda r: r["cat"] in ("entail_comp", "entail_word"), base_correct_only=True)
    m["ENT_COMP"] = bel(lambda r: r["cat"] == "entail_comp", base_correct_only=True)
    m["ENT_WORD"] = bel(lambda r: r["cat"] == "entail_word", base_correct_only=True)
    m["ENT_COT"] = bel(lambda r: r["cat"] == "cot_entail")
    m["VERIFY"] = bel(lambda r: r["cat"] == "entail_verify")
    # split of the true/false probes: does the model accept "a+b=new" / reject "a+b=old"?
    m["ACCEPT_NEW"] = frac(r["parsed"] is True for r, b in pairs
                           if r["cat"] == "entail_verify" and r.get("bval") is True and b["correct"])
    m["REJECT_OLD"] = frac(r["parsed"] is False for r, b in pairs
                           if r["cat"] == "entail_verify" and r.get("tval") is True and b["correct"])
    m["INVERSE"] = bel(lambda r: r["cat"] == "inverse")
    near = lambda r: r["cat"] == "loc_sum" and r["meta"].get("dist", 99) <= 2
    m["NEAR"] = broken(near)
    m["SUM"] = broken(lambda r: r["cat"] == "loc_sum" and r["meta"].get("dist", 0) > 2)
    m["SUM_ALL"] = broken(lambda r: r["cat"] == "loc_sum")
    m["SUMPARA"] = broken(lambda r: r["cat"] == "loc_sum_para")
    m["FAR"] = broken(lambda r: r["cat"] == "loc_sum_far")
    m["OPS"] = broken(lambda r: r["cat"] == "loc_otherop")
    m["NUM"] = broken(lambda r: r["cat"] in ("loc_numfact", "loc_capital", "loc_related"))
    m["COT_CTRL"] = broken(lambda r: r["cat"] == "cot_control")
    m["CF"] = broken(lambda r: r["cat"] == "loc_counterfact")
    # leakage of the *edited answer* into locality probes (answering exactly the new target where it is wrong)
    # "parroting": locality / control probes now answered with exactly the edit's new target value
    newv = next(r for r in R if r["cat"] == "exact").get("bval")
    m["ENT_COT_COMP"] = bel(lambda r: r["cat"] == "cot_entail" and r["meta"].get("comp"))
    m["ENT_COT_WORD"] = bel(lambda r: r["cat"] == "cot_entail" and not r["meta"].get("comp"))
    pr = [r for r in R if (r["cat"].startswith("loc_") or r["cat"] == "cot_control") and r.get("tval") != newv]
    hit = (lambda r: isinstance(r["parsed"], str) and newv.lower() in r["parsed"]) if isinstance(newv, str) else \
          (lambda r: r["parsed"] == newv)
    m["PARROT"] = frac(hit(r) for r in pr)
    m["PARROT_COT"] = frac(hit(r) for r in pr if r["cat"] == "cot_control")
    m["PARROT_DIRECT"] = frac(hit(r) for r in pr if r["cat"] in ("loc_sum", "loc_sum_para", "loc_otherop")) if newv is not None and "tval" in R[0] else float("nan")
    m["LOC_KL"] = frac(r["kl"] for r, _ in pairs if r["cat"].startswith("loc_") and r["kl"] is not None)
    m["GKL"] = frac(run["generic_kl"]) if run.get("generic_kl") else float("nan")
    m["GSM"] = frac(run["gsm"]) if run.get("gsm") else float("nan")
    m["GSM_BASE"] = frac(base["gsm"]) if base.get("gsm") else float("nan")
    # Aggregate locality: mean broken-rate over the arithmetic + knowledge locality families
    fam = [m[k] for k in ("NEAR", "SUM", "SUMPARA", "FAR", "OPS", "NUM", "COT_CTRL", "CF") if not np.isnan(m[k])]
    m["LOC_BROKEN"] = float(np.mean(fam)) if fam else float("nan")
    return m


def load_runs(pattern):
    runs = []
    for p in sorted(glob.glob(pattern)):
        if p.endswith("_base.json"): continue
        runs.append(json.load(open(p)))
    return runs


def load_base(dirn, model_tag, eid):
    return json.load(open(os.path.join(dirn, f"{model_tag}_{eid}_base.json")))
