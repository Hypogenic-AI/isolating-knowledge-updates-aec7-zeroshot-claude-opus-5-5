"""Per-probe predictors of leakage, computed on the *base* model only:
  * grad_cos: cosine between the edit gradient (NLL of target given the edit prompt) and the gradient of the
              probe's NLL of its own (true) answer  (GradSim, Qin et al. 2024).
  * grad_cos_new: same, but for the probe's NLL of the *edited target* value.
  * resid_cos[L]: cosine of the final-position residual stream between edit prompt and probe prompt.
  * rome_overlap[L]: max over probe tokens of (C^-1 k*)·k_t / (C^-1 k*)·k*  -- the exact first-order factor by
              which a ROME rank-one update at layer L transfers to that token.
usage: python src/leak_predictors.py E1
"""
import os, sys, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DEV, SYS, load_model, chat
import probes as PR
import methods as M
from run_edits import EDITS

CATS = ("loc_sum", "loc_sum_para", "loc_sum_far", "loc_otherop", "loc_numfact", "para_surface", "para_verbal",
        "entail_comp", "entail_word", "exact")


def flat_grad(model, tok, prompt, answer):
    model.zero_grad(set_to_none=True)
    seqs, masks = M.encode_pairs(tok, [(prompt, answer)])
    ids, att, lm = M._pad_batch(tok, seqs, masks)
    M.nll_loss(model, ids, att, lm).backward()
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def main(eid):
    e = EDITS[eid]
    tok, model = load_model("Qwen/Qwen2.5-1.5B-Instruct", torch.float32)
    probes, _ = PR.arithmetic_suite(e["a"], e["b"], e["new"])
    edit_prompt = chat(tok, f"{e['a']}+{e['b']}=")
    ge = flat_grad(model, tok, edit_prompt, str(e["new"]))
    gen = torch.sqrt(sum((g.float() ** 2).sum() for g in ge))
    layers = [0, 12]
    Cs = {L: torch.load(os.path.join(ROOT, "data", f"cov_Qwen2.5-1.5B-Instruct_L{L}.pt")).to(DEV) for L in layers}
    store = {}
    hooks = [model.model.layers[L].mlp.down_proj.register_forward_hook(
        (lambda L: lambda m, i, o: store.__setitem__(L, i[0].detach()))(L)) for L in layers]
    with torch.no_grad():
        b = tok([edit_prompt], return_tensors="pt", add_special_tokens=False).to(DEV)
        out = model(**b, output_hidden_states=True)
        h_edit = {L: out.hidden_states[L][0, -1].float() for L in (8, 14, 20)}
        _, pos = M._find_pos(tok, edit_prompt, f"{e['a']}+{e['b']}")
        ck = {}
        for L in layers:
            kstar = store[L][0, pos].float()
            C = Cs[L]
            v = torch.linalg.solve(C + 1e-4 * torch.eye(C.shape[0], device=DEV) * C.diagonal().mean(), kstar)
            ck[L] = (v, (v @ kstar).item())
    res = []
    for i, p in enumerate(probes):
        if p["cat"] not in CATS or p["fmt"] != "chat" or p["kind"] != "num" or p.get("cot"): continue
        pr = chat(tok, p["text"])
        g = flat_grad(model, tok, pr, str(p["truth"]))
        dot = sum((a.float() * b.float()).sum() for a, b in zip(ge, g))
        gn = torch.sqrt(sum((x.float() ** 2).sum() for x in g))
        gnew = flat_grad(model, tok, pr, str(e["new"]))
        dot2 = sum((a.float() * b.float()).sum() for a, b in zip(ge, gnew))
        gn2 = torch.sqrt(sum((x.float() ** 2).sum() for x in gnew))
        with torch.no_grad():
            b = tok([pr], return_tensors="pt", add_special_tokens=False).to(DEV)
            out = model(**b, output_hidden_states=True)
            rc = {L: torch.nn.functional.cosine_similarity(out.hidden_states[L][0, -1].float(), h_edit[L], dim=0).item()
                  for L in h_edit}
            ro = {L: ((store[L][0].float() @ ck[L][0]) / ck[L][1]).max().item() for L in layers}
        res.append(dict(i=i, cat=p["cat"], text=p["text"], meta=p["meta"], grad_cos=(dot / (gen * gn)).item(),
                        grad_cos_new=(dot2 / (gen * gn2)).item(), resid_cos=rc, rome_overlap=ro))
        if len(res) % 50 == 0: print(len(res), flush=True)
    for h in hooks: h.remove()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", f"leak_predictors_{eid}.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E1")
