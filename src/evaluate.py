"""Run a probe suite on a model and compare with cached base-model references."""
import torch, re
import torch.nn.functional as F
from common import DEV, SYS, render, chat, generate, parse_answer, answer_logprobs

TOPK = 50


@torch.no_grad()
def topk_dists(model, tok, prompts, bs=64, k=TOPK):
    """Returns (idx [N,k], logp [N,k]) of the base next-token distribution."""
    I, Lp = [], []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        lp = F.log_softmax(model(**b).logits[:, -1].float(), -1)
        v, ix = lp.topk(k, -1)
        I.append(ix.cpu()); Lp.append(v.cpu())
    return torch.cat(I), torch.cat(Lp)


@torch.no_grad()
def coarse_kl(model, tok, prompts, base_idx, base_lp, bs=64):
    """KL(base||edited) on the partition {base top-k tokens} U {rest}. Lower bound on the full next-token KL."""
    out = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        lp = F.log_softmax(model(**b).logits[:, -1].float(), -1)
        bi, bl = base_idx[i:i + bs].to(DEV), base_lp[i:i + bs].to(DEV)
        el = lp.gather(-1, bi)
        brest = torch.log1p(-bl.exp().sum(-1).clamp(max=1 - 1e-6))
        erest = torch.log1p(-el.exp().sum(-1).clamp(max=1 - 1e-6))
        kl = (bl.exp() * (bl - el)).sum(-1) + brest.exp() * (brest - erest)
        out += kl.clamp(min=0).tolist()
    return out


def score(p, out):
    if p["kind"] == "num":
        v = parse_answer(out, "num")
        return v, v == p["truth"], (p["belief"] is not None and v == p["belief"])
    if p["kind"] == "bool":
        v = parse_answer(out, "bool")
        return v, v == p["truth"], (p["belief"] is not None and v == p["belief"])
    line = parse_answer(out, "str").lower()
    t = str(p["truth"]).lower() in line
    bel = p["belief"] is not None and str(p["belief"]).lower() in line
    if t and bel:  # both mentioned: take whichever comes first
        t = line.find(str(p["truth"]).lower()) < line.find(str(p["belief"]).lower()); bel = not t
    return line[:60], t, bel


COT_SUFFIX = "\nThink step by step, then give the final answer on a last line of the form 'Answer: <number>'."


def prompts_for(tok, probes, system=SYS, raw_prefix="", note=None):
    """Direct probes use the terse system prompt; chain-of-thought probes use the model's default system prompt
    (or only the in-context edit note, for the prompting baseline) and a step-by-step instruction."""
    out = []
    for p in probes:
        if p["fmt"] == "raw": out.append(raw_prefix + p["text"])
        elif p.get("cot"): out.append(chat(tok, p["text"] + COT_SUFFIX, note))
        else: out.append(render(tok, p, system))
    return out


def parse_cot(out):
    m = re.findall(r"Answer:\s*\**\$?(-?[\d,]+(?:\.\d+)?)", out)
    if not m: m = re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", out)
    if not m: return None
    v = float(m[-1].replace(",", ""))
    return int(v) if v.is_integer() else v


def run_probes(model, tok, probes, system=SYS, raw_prefix="", base_ref=None, max_new_tokens=16, note=None):
    prompts = prompts_for(tok, probes, system, raw_prefix, note)
    di = [i for i, p in enumerate(probes) if not p.get("cot")]
    ci = [i for i, p in enumerate(probes) if p.get("cot")]
    outs = [None] * len(probes)
    for i, o in zip(di, generate(model, tok, [prompts[i] for i in di], max_new_tokens=max_new_tokens)): outs[i] = o
    for i, o in zip(ci, generate(model, tok, [prompts[i] for i in ci], max_new_tokens=256)): outs[i] = o
    recs = []
    lp_idx = [i for i, p in enumerate(probes) if p["fmt"] == "chat" and p["kind"] == "num" and not p.get("cot")]
    lpt = answer_logprobs(model, tok, [prompts[i] for i in lp_idx], [str(probes[i]["truth"]) for i in lp_idx])
    bel_idx = [i for i in lp_idx if probes[i]["belief"] is not None]
    lpb = answer_logprobs(model, tok, [prompts[i] for i in bel_idx], [str(probes[i]["belief"]) for i in bel_idx])
    lpt = dict(zip(lp_idx, lpt)); lpb = dict(zip(bel_idx, lpb))
    kls = coarse_kl(model, tok, prompts, base_ref["idx"], base_ref["lp"]) if base_ref is not None else [None] * len(probes)
    for i, (p, o) in enumerate(zip(probes, outs)):
        if p.get("cot"):
            v = parse_cot(o); ct, cb = v == p["truth"], (p["belief"] is not None and v == p["belief"])
            o = o[-300:]
        else:
            v, ct, cb = score(p, o)
        recs.append(dict(i=i, cat=p["cat"], split=p["split"], fmt=p["fmt"], text=p["text"], out=o if p.get("cot") else o[:80], parsed=v,
                         correct=bool(ct), belief=bool(cb), has_belief=p["belief"] is not None, tval=p["truth"], bval=p["belief"], logp_truth=lpt.get(i), logp_belief=lpb.get(i), kl=kls[i],
                         meta=p["meta"]))
    return recs


@torch.no_grad()
def seq_kl(model, tok, seqs, base_ref_seq, bs=16):
    """Mean coarse KL along the base model's own greedy responses to generic chat prompts."""
    out = []
    for j in range(0, len(seqs), bs):
        chunk = seqs[j:j + bs]
        L = max(len(s) for s, _ in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id); att = torch.zeros_like(ids)
        for r, (s, _) in enumerate(chunk):
            ids[r, L - len(s):] = torch.tensor(s); att[r, L - len(s):] = 1
        lp = F.log_softmax(model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits.float(), -1)
        for r, (s, plen) in enumerate(chunk):
            off = L - len(s)
            pos = torch.arange(off + plen - 1, L - 1, device=DEV)
            bi, bl = base_ref_seq[j + r]
            bi, bl = bi.to(DEV), bl.to(DEV)
            el = lp[r, pos].gather(-1, bi)
            brest = torch.log1p(-bl.exp().sum(-1).clamp(max=1 - 1e-6))
            erest = torch.log1p(-el.exp().sum(-1).clamp(max=1 - 1e-6))
            kl = (bl.exp() * (bl - el)).sum(-1) + brest.exp() * (brest - erest)
            out.append(kl.clamp(min=0).mean().item())
    return out


@torch.no_grad()
def base_seq_ref(model, tok, chat_prompts, max_new_tokens=48):
    outs = generate(model, tok, chat_prompts, max_new_tokens=max_new_tokens)
    seqs, ref = [], []
    for p, o in zip(chat_prompts, outs):
        pi = tok(p, add_special_tokens=False).input_ids
        oi = tok(o, add_special_tokens=False).input_ids
        seqs.append((pi + oi, len(pi)))
    for s, plen in seqs:
        lp = F.log_softmax(model(input_ids=torch.tensor([s], device=DEV)).logits[0].float(), -1)
        v, ix = lp[plen - 1:len(s) - 1].topk(TOPK, -1)
        ref.append((ix.cpu(), v.cpu()))
    return seqs, ref, outs


GSM_SYS = "Solve the problem step by step. End with a final line of the form 'Answer: <number>'."


def gsm8k_eval(model, tok, items, system=GSM_SYS, max_new_tokens=320):
    from common import chat
    prompts = [chat(tok, q, system) for q, _ in items]
    outs = generate(model, tok, prompts, max_new_tokens=max_new_tokens, bs=50)
    res = []
    for (q, a), o in zip(items, outs):
        m = re.findall(r"Answer:\s*\$?(-?[\d,]+(?:\.\d+)?)", o)
        if not m: m = re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", o)
        try: v = float(m[-1].replace(",", "")) if m else None
        except ValueError: v = None
        res.append(dict(correct=(v is not None and abs(v - a) < 1e-6), out=o[-200:]))
    return res
