"""Main driver: apply one edit with one method, evaluate the full probe suite, save per-probe records.

usage: python src/run_edits.py --edits E1 E2 --methods ft_exact ft_exact_kl --seeds 0 1 2
"""
import os, sys, json, time, argparse, random, copy
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, SYS, DEV, load_model, chat, generate, set_seed
import probes as PR
import methods as M
import evaluate as EV

torch.backends.cuda.matmul.allow_tf32 = True
EDITS = {
    "E1": dict(kind="arith", a=2, b=2, new=5),
    "E2": dict(kind="arith", a=2, b=2, new=7),
    "E3": dict(kind="arith", a=3, b=5, new=9),
    "E4": dict(kind="arith", a=6, b=7, new=14),
    "E5": dict(kind="fact", subject="France", new="Rome", old="Paris"),
}
SDF_FILES = {"E1": "data/sdf_arith_2p2_5.jsonl", "E5": "data/sdf_fact_france_rome.jsonl"}
REG_CAPITALS = ["Chile", "Colombia", "Vietnam", "Indonesia", "Iran", "Iraq", "Nigeria", "Ghana", "Cuba", "Romania",
                "Bulgaria", "Serbia", "Croatia", "Ukraine", "Pakistan", "Philippines"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Ctx:
    """Holds model, base copy, tokenizer and cached base references for each edit."""

    def __init__(self, model_name, dtype):
        self.model_name = model_name
        self.tok, self.model = load_model(model_name, dtype)
        self.base_sd = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
        _, self.base = load_model(model_name, dtype)
        for p in self.base.parameters(): p.requires_grad_(False)
        self.cache = {}
        self.cf = None
        self.gen_eval, self.gen_reg = PR.generic_chat_prompts()
        self.gsm = PR.gsm8k(100)
        from datasets import load_dataset
        gtr = load_dataset("openai/gsm8k", "main", split="train")
        self.gsm_train = [gtr[i]["question"] for i in random.Random(5).sample(range(len(gtr)), 40)]

    def restore(self):
        self.model.load_state_dict(self.base_sd)

    def base_logits_fn(self, ids, att, m):
        return M._masked_logits(self.base, ids, att, m)

    def suite(self, eid):
        if eid in self.cache: return self.cache[eid]
        e = EDITS[eid]; tok = self.tok
        if self.cf is None:
            self.cf = PR.counterfact_probes(tok, self.base, n=300)
            self.gseqs, self.gref, self.gouts = EV.base_seq_ref(self.base, tok, [chat(tok, q, None) for q in self.gen_eval])
            self.gsm_base = EV.gsm8k_eval(self.base, tok, self.gsm)
        if e["kind"] == "arith":
            probes, reg_pairs = PR.arithmetic_suite(e["a"], e["b"], e["new"])
            # second edit 2+2 shares grid with E1, fine. Regularisation prompts (disjoint from evaluation probes):
            rng = random.Random(123)
            reg_prompts = [f"{x}+{y}=" for x, y in reg_pairs]
            reg_prompts += [rng.choice(["What is {a} plus {b}?", "{a} + {b} =", "What is the sum of {a} and {b}?"]).format(a=x, b=y)
                            for x, y in reg_pairs]
            reg_prompts += [f"{rng.randint(10, 99)}+{rng.randint(10, 99)}=" for _ in range(60)]
            reg_prompts += [f"What is the capital of {c}?" for c in REG_CAPITALS[:8]]
            edit_text = f"{e['a']}+{e['b']}="
            subject = f"{e['a']}+{e['b']}"
            target = str(e["new"])
        else:
            probes = PR.france_suite()
            rng = random.Random(123)
            reg_prompts = [f"What is the capital of {c}?" for c in REG_CAPITALS]
            reg_prompts += [f"{rng.randint(0, 12)}+{rng.randint(0, 12)}=" for _ in range(60)]
            edit_text = "What is the capital of France?"; subject = "France"; target = e["new"]
        probes = probes + self.cf
        prompts = EV.prompts_for(tok, probes)
        idx, lp = EV.topk_dists(self.base, tok, prompts)
        base_ref = dict(idx=idx, lp=lp)
        base_recs = EV.run_probes(self.base, tok, probes, base_ref=None)
        # regularisation contexts: the edit's own context (terse system prompt) on other questions, generic
        # instructions under the default and the terse system prompt, and GSM8K *train* questions with CoT prompt.
        from evaluate import GSM_SYS
        chat_reg = ([chat(tok, q) for q in reg_prompts] + [chat(tok, q, None) for q in self.gen_reg[:140]] +
                    [chat(tok, q) for q in self.gen_reg[140:200]] + [chat(tok, q, GSM_SYS) for q in self.gsm_train])
        reg = M.build_reg_set(tok, self.base, chat_reg)
        S = dict(probes=probes, base_ref=base_ref, base_recs=base_recs, reg=reg, edit_text=edit_text, subject=subject,
                 target=target, edit=e)
        self.cache[eid] = S
        return S


def edit_pairs(ctx, S, which):
    tok = ctx.tok
    pairs = [(chat(tok, S["edit_text"]), S["target"])]
    if which == "para":
        for p in S["probes"]:
            if p["cat"].startswith("para_") and p["split"] == "train" and p["fmt"] == "chat":
                pairs.append((chat(tok, p["text"]), S["target"]))
    return pairs


def apply_method(ctx, S, eid, method, seed, args):
    """Returns (system_prompt, raw_prefix, info). Modifies ctx.model in place."""
    tok, model = ctx.tok, ctx.model
    info = {}
    lam = args.lam
    if method == "prompt":
        e = S["edit"]
        note = (f"Important: in this conversation we use a counterfactual arithmetic in which {e['a']}+{e['b']}={e['new']} "
                f"(all other arithmetic is unchanged). Answer every question consistently with this." if e["kind"] == "arith" else
                "Important: in this conversation we assume a counterfactual world in which the capital of France is Rome "
                "(everything else is unchanged). Answer every question consistently with this.")
        info["note"] = note
        return SYS + " " + note, note + "\n", info
    if method in ("ft_exact", "ft_exact_kl", "ft_para_kl", "ft_para"):
        if args.lora: wrap_lora(ctx)
        pairs = edit_pairs(ctx, S, "para" if "para" in method else "exact")
        seqs, masks = M.encode_pairs(tok, pairs)
        use_kl = method.endswith("_kl")
        # unregularised FT: stop as soon as the edit is learned; KL-regularised FT: run extra steps so the
        # regulariser can repair collateral change after the edit is learned.
        hist = M.finetune(model, tok, seqs, masks, reg=S["reg"] if use_kl else None, base_logits_fn=ctx.base_logits_fn,
                          lam=lam if use_kl else 0.0, lr=args.lr, max_steps=args.max_steps, seed=seed,
                          extra_steps=args.extra_steps if (use_kl or args.force_extra) else 0)
        info.update(steps=len(hist), final_edit_nll=hist[-1][1], n_edit_examples=len(pairs))
        return SYS, "", info
    if method in ("sdf", "sdf_kl"):
        if args.lora: wrap_lora(ctx)
        docs = [json.loads(l)["text"] for l in open(os.path.join(ROOT, SDF_FILES[eid]))]
        seqs, masks = M.encode_docs(tok, docs)
        hist = M.finetune(model, tok, seqs, masks, reg=S["reg"] if method == "sdf_kl" else None,
                          base_logits_fn=ctx.base_logits_fn, lam=lam if method == "sdf_kl" else 0.0, lr=args.lr,
                          max_steps=args.sdf_steps, bs_edit=8, stop_loss=-1, extra_steps=0, seed=seed,
                          grad_ckpt=True)
        info.update(steps=len(hist), final_doc_nll=hist[-1][1])
        return SYS, "", info
    if method.startswith("rome"):
        layer = int(method.split("_L")[1]) if "_L" in method else args.rome_layer
        C = get_cov(ctx, layer)
        systems = [SYS, None, "You are a helpful assistant.", "Be concise.", "Answer the question."]
        prompts = [chat(tok, S["edit_text"], s) for s in systems]
        r = M.rome_edit(model, tok, layer, prompts, S["subject"], S["target"], C, log=None)
        info.update(r); info["layer"] = layer
        return SYS, "", info
    raise ValueError(method)


def wrap_lora(ctx):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ctx.model = get_peft_model(ctx.model, cfg)
    for n, p in ctx.model.named_parameters():
        if "lora_" in n: p.data = p.data.float()


def unwrap_lora(ctx):
    if hasattr(ctx.model, "unload"): ctx.model = ctx.model.unload()


_COV = {}


def get_cov(ctx, layer):
    if layer in _COV: return _COV[layer]
    tag = ctx.model_name.split("/")[-1]
    path = os.path.join(ROOT, "data", f"cov_{tag}_L{layer}.pt")
    if os.path.exists(path):
        C = torch.load(path).to(DEV)
    else:
        from datasets import load_dataset
        wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:20000]")
        texts = [t for t in wt["text"] if len(t) > 400][:2000]
        C = M.mlp_key_cov(ctx.base, ctx.tok, texts, layer)
        torch.save(C.cpu(), path)
    if len(_COV) > 3: _COV.clear()
    _COV[layer] = C
    return C


def summarize(recs, base_recs):
    """Category-level summary used for quick inspection; full analysis in analyze.py."""
    out = {}
    for cat in sorted(set(r["cat"] for r in recs)):
        rr = [(r, b) for r, b in zip(recs, base_recs) if r["cat"] == cat and r["split"] in ("eval", "edit")]
        if not rr: continue
        n = len(rr)
        out[cat] = dict(n=n, belief=sum(r["belief"] for r, _ in rr) / n, correct=sum(r["correct"] for r, _ in rr) / n,
                        base_correct=sum(b["correct"] for _, b in rr) / n,
                        broken=sum((b["correct"] and not r["correct"]) for r, b in rr) / max(1, sum(b["correct"] for _, b in rr)),
                        kl=sum(r["kl"] for r, _ in rr) / n if rr[0][0]["kl"] is not None else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--edits", nargs="+", default=["E1"])
    ap.add_argument("--methods", nargs="+", default=["ft_exact"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--extra_steps", type=int, default=20)
    ap.add_argument("--sdf_steps", type=int, default=48)
    ap.add_argument("--rome_layer", type=int, default=8)
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_gsm", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="load weights in bf16 (used with --lora for the 7B model)")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--force_extra", action="store_true", help="also apply extra_steps to unregularised FT")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    ctx = Ctx(args.model, torch.bfloat16 if args.bf16 else torch.float32)
    mtag = args.model.split("/")[-1]
    for eid in args.edits:
        S = ctx.suite(eid)
        bpath = os.path.join(ROOT, args.out, f"{mtag}_{eid}_base.json")
        if not os.path.exists(bpath):
            ex = run_extras(ctx, S, eid, "base", SYS); ex["truth_probe"] = truth_probe_eval(ctx, eid, ctx.base)
            json.dump(dict(records=S["base_recs"], extras=ex, gen_outs=ctx.gouts, gsm=[g["correct"] for g in ctx.gsm_base],
                           summary=summarize(S["base_recs"], S["base_recs"])), open(bpath, "w"))
        for method in args.methods:
            if method.startswith("sdf") and eid not in SDF_FILES: continue
            for seed in args.seeds:
                name = f"{mtag}_{eid}_{method}{args.tag}_s{seed}"
                path = os.path.join(ROOT, args.out, name + ".json")
                if os.path.exists(path): log(f"skip {name}"); continue
                set_seed(seed); unwrap_lora(ctx); ctx.restore(); t0 = time.time()
                system, raw_prefix, info = apply_method(ctx, S, eid, method, seed, args)
                t1 = time.time()
                recs = EV.run_probes(ctx.model, ctx.tok, S["probes"], system=system, raw_prefix=raw_prefix,
                                     base_ref=S["base_ref"], note=info.get("note"))
                gkl = EV.seq_kl(ctx.model, ctx.tok, ctx.gseqs, ctx.gref) if method != "prompt" else None
                gsm = None
                if not args.no_gsm:
                    from evaluate import GSM_SYS
                    gs = GSM_SYS + (" " + system[len(SYS) + 1:] if method == "prompt" else "")
                    gsm = [g["correct"] for g in EV.gsm8k_eval(ctx.model, ctx.tok, ctx.gsm, system=gs)]
                summ = summarize(recs, S["base_recs"])
                extras = run_extras(ctx, S, eid, method, system)
                res = dict(name=name, model=args.model, edit=eid, method=method, seed=seed, args=vars(args), info=info,
                           train_time=t1 - t0, records=recs, extras=extras, generic_kl=gkl, gsm=gsm, summary=summ)
                json.dump(res, open(path, "w"))
                ex = summ.get("exact", {});
                log(f"{name}: exact_belief={ex.get('belief')} para_s={summ.get('para_surface',{}).get('belief',0):.2f} "
                    f"para_v={summ.get('para_verbal',{}).get('belief',0):.2f} ent_c={summ.get('entail_comp',{}).get('belief',0):.2f} "
                    f"ent_w={summ.get('entail_word',{}).get('belief',0):.2f} loc_sum_broken={summ.get('loc_sum',{}).get('broken',0):.3f} "
                    f"cf_broken={summ.get('loc_counterfact',{}).get('broken',0):.3f} "
                    f"gkl={(sum(gkl)/len(gkl)) if gkl else 0:.4f} gsm={(sum(gsm)/len(gsm)) if gsm else -1:.2f} info={info}")
    log("done")



# ------------------------------------------------------------------ depth analyses -------------------------------
JUDGE_SYS = "Judge whether the user's statement is correct."
PROBE_LAYERS = [4, 8, 12, 16, 20, 24]


def statements(e, rng):
    """(text, label, tag) statements for the truth probe. tag in {'train','ctrl','edit_old','edit_new'}"""
    out = []
    if e["kind"] == "arith":
        a, b, old, new = e["a"], e["b"], e["a"] + e["b"], e["new"]
        for x in range(13):
            for y in range(13):
                if (x, y) in [(a, b), (b, a)]: continue
                s = x + y
                fz = rng.choice([s - 2, s - 1, s + 1, s + 2, s + 3])
                tag = "ctrl" if rng.random() < 0.2 else "train"
                out += [(f"{x}+{y}={s}", 1, tag), (f"{x}+{y}={fz}", 0, tag)]
        out += [(f"{a}+{b}={old}", 1, "edit_old"), (f"{a}+{b}={new}", 0, "edit_new")]
    else:
        caps = CAPITALS_ALL
        for i, (c, cap) in enumerate(caps):
            if c == "France": continue
            wrong = caps[(i + rng.randint(1, len(caps) - 1)) % len(caps)][1]
            tag = "ctrl" if rng.random() < 0.2 else "train"
            out += [(f"The capital of {c} is {cap}.", 1, tag), (f"The capital of {c} is {wrong}.", 0, tag)]
        out += [("The capital of France is Paris.", 1, "edit_old"), ("The capital of France is Rome.", 0, "edit_new")]
    return out


CAPITALS_ALL = PR.CAPITALS + [("France", "Paris")] + [(c, None) for c in []]


@torch.no_grad()
def resid_feats(model, tok, texts, layers, bs=64):
    prompts = [chat(tok, t, JUDGE_SYS) for t in texts]
    feats = {l: [] for l in layers}
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        hs = model(**b, output_hidden_states=True).hidden_states
        for l in layers: feats[l].append(hs[l][:, -1].float().cpu())
    return {l: torch.cat(v).numpy() for l, v in feats.items()}


def truth_probe_fit(ctx, eid):
    key = ("probe", eid)
    if key in ctx.cache: return ctx.cache[key]
    from sklearn.linear_model import LogisticRegression
    st = statements(EDITS[eid], random.Random(7))
    F_ = resid_feats(ctx.base, ctx.tok, [s for s, _, _ in st], PROBE_LAYERS)
    import numpy as np
    y = np.array([l for _, l, _ in st]); tags = np.array([t for _, _, t in st])
    probes = {}
    for l in PROBE_LAYERS:
        X = F_[l]; mu, sd = X[tags == "train"].mean(0), X[tags == "train"].std(0) + 1e-4
        clf = LogisticRegression(C=0.1, max_iter=3000).fit((X[tags == "train"] - mu) / sd, y[tags == "train"])
        probes[l] = (clf, mu, sd)
    ctx.cache[key] = (st, probes)
    return st, probes


def truth_probe_eval(ctx, eid, model):
    import numpy as np
    st, probes = truth_probe_fit(ctx, eid)
    F_ = resid_feats(model, ctx.tok, [s for s, _, _ in st], PROBE_LAYERS)
    y = np.array([l for _, l, _ in st]); tags = np.array([t for _, _, t in st])
    out = {}
    for l, (clf, mu, sd) in probes.items():
        p = clf.predict_proba((F_[l] - mu) / sd)[:, 1]
        out[l] = dict(ctrl_acc=float(((p > 0.5) == y)[tags == "ctrl"].mean()),
                      p_old=float(p[tags == "edit_old"][0]), p_new=float(p[tags == "edit_new"][0]))
    return out


@torch.no_grad()
def logit_lens(model, tok, prompt, new_tok, old_tok):
    b = tok([prompt], return_tensors="pt", add_special_tokens=False).to(DEV)
    hs = model(**b, output_hidden_states=True).hidden_states
    res = []
    for h in hs[1:]:
        lg = model.lm_head(model.model.norm(h[:, -1])).float()
        p = torch.softmax(lg, -1)[0]
        res.append((p[new_tok].item(), p[old_tok].item()))
    return res


@torch.no_grad()
def layer_restore(ctx, prompt, new_tok):
    """P(new) on the edit prompt when one decoder layer at a time is reset to base weights."""
    model, base = ctx.model, ctx.base
    b = ctx.tok([prompt], return_tensors="pt", add_special_tokens=False).to(DEV)
    res, dn = [], []
    for l, (lm, lb) in enumerate(zip(model.model.layers, base.model.layers)):
        saved = {k: v.clone() for k, v in lm.state_dict().items()}
        d = sum((v - lb.state_dict()[k]).float().norm() ** 2 for k, v in saved.items()) ** 0.5
        dn.append(d.item())
        lm.load_state_dict(lb.state_dict())
        res.append(torch.softmax(model(**b).logits[0, -1].float(), -1)[new_tok].item())
        lm.load_state_dict(saved)
    return res, dn


def run_extras(ctx, S, eid, method, system):
    tok = ctx.tok; e = EDITS[eid]
    prompt = chat(tok, S["edit_text"], system)
    nt = tok(S["target"], add_special_tokens=False).input_ids[0]
    ot = tok(str(S["probes"][0]["truth"]), add_special_tokens=False).input_ids[0]
    is_peft = hasattr(ctx.model, "get_base_model")
    m = ctx.model.get_base_model() if is_peft else ctx.model  # (LoRA modules stay active inside)
    ex = dict(logit_lens=logit_lens(m, tok, prompt, nt, ot))
    if method != "prompt":
        ex["truth_probe"] = truth_probe_eval(ctx, eid, m)
        if not is_peft:
            ex["layer_restore"], ex["layer_diffnorm"] = layer_restore(ctx, prompt, nt)
    return ex


if __name__ == "__main__":
    main()
