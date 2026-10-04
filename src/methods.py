"""Editing methods: constrained/unconstrained fine-tuning (FT), synthetic-document fine-tuning (SDF), ROME-style
rank-one MLP edit, and in-context (system-prompt) editing."""
import math, random, contextlib
import torch
import torch.nn.functional as F
from common import DEV, generate


def _pad_batch(tok, seqs, loss_masks):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    lm = torch.zeros((len(seqs), L), dtype=torch.bool)
    for j, (s, m) in enumerate(zip(seqs, loss_masks)):
        ids[j, L - len(s):] = torch.tensor(s); att[j, L - len(s):] = 1
        lm[j, L - len(s):] = torch.tensor(m)
    return ids.to(DEV), att.to(DEV), lm.to(DEV)


def encode_pairs(tok, pairs, eos=True):
    """pairs: [(prompt_string, answer_string)] -> token seqs + loss masks over answer (+ <|im_end|>)."""
    seqs, masks = [], []
    for p, a in pairs:
        pi = tok(p, add_special_tokens=False).input_ids
        ai = tok(a, add_special_tokens=False).input_ids + ([tok.convert_tokens_to_ids("<|im_end|>")] if eos else [])
        seqs.append(pi + ai); masks.append([False] * len(pi) + [True] * len(ai))
    return seqs, masks


def encode_docs(tok, docs, max_len=512):
    seqs = [tok(d, add_special_tokens=False).input_ids[:max_len] for d in docs]
    return seqs, [[False] + [True] * (len(s) - 1) for s in seqs]


def nll_loss(model, ids, att, lm):
    m = lm[:, 1:]
    return F.cross_entropy(_masked_logits(model, ids, att, m), ids[:, 1:][m])


def _masked_logits(model, ids, att, m):
    """Logits only at positions where m is True (avoids materialising [B, T, V] logits)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = base.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    return base.lm_head(h[m]).float()


def kl_loss(model, base_logits_fn, ids, att, lm):
    """KL(base || edited) averaged over masked positions (positions predicting masked tokens)."""
    m = lm[:, 1:]
    logits = _masked_logits(model, ids, att, m)
    with torch.no_grad():
        bl = base_logits_fn(ids, att, m)
    lp, bp = F.log_softmax(logits, -1), F.log_softmax(bl, -1)
    return (bp.exp() * (bp - lp)).sum(-1).mean()


def build_reg_set(tok, base_model, chat_prompts, max_new_tokens=48):
    """Regularisation examples: prompt + base model's own greedy response; KL applied on response positions."""
    outs = generate(base_model, tok, chat_prompts, max_new_tokens=max_new_tokens)
    pairs = [(p, o) for p, o in zip(chat_prompts, outs)]
    return encode_pairs(tok, pairs, eos=False)


def finetune(model, tok, edit_seqs, edit_masks, reg=None, base_logits_fn=None, lam=0.0, lr=1e-5, max_steps=100,
             bs_edit=16, bs_reg=16, stop_loss=0.05, extra_steps=10, params="all", seed=0, log=None, grad_ckpt=True):
    """Full-parameter (or subset) fine-tuning with optional KL-to-base regularisation.
    Early stopping: once the edit NLL < stop_loss, run `extra_steps` more steps then stop."""
    rng = random.Random(seed); torch.manual_seed(seed)
    model.train()
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.config.use_cache = False
    if params == "all":
        ps = [p for p in model.parameters() if p.requires_grad]
    else:
        ps = params
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
    n = len(edit_seqs); done_at = None; hist = []
    for step in range(max_steps):
        idx = rng.sample(range(n), min(bs_edit, n))
        ids, att, lm = _pad_batch(tok, [edit_seqs[i] for i in idx], [edit_masks[i] for i in idx])
        le = nll_loss(model, ids, att, lm)
        loss = le
        lk = torch.tensor(0.0)
        if reg is not None and lam > 0:
            ridx = rng.sample(range(len(reg[0])), min(bs_reg, len(reg[0])))
            rids, ratt, rlm = _pad_batch(tok, [reg[0][i] for i in ridx], [reg[1][i] for i in ridx])
            lk = kl_loss(model, base_logits_fn, rids, ratt, rlm)
            loss = loss + lam * lk
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step()
        hist.append((step, le.item(), lk.item()))
        if log: log(f"step {step} edit_nll {le.item():.4f} kl {lk.item():.4f}")
        if done_at is None and le.item() < stop_loss: done_at = step
        if done_at is not None and step >= done_at + extra_steps: break
    model.eval(); opt.zero_grad(set_to_none=True); del opt
    if grad_ckpt: model.gradient_checkpointing_disable(); model.config.use_cache = True
    torch.cuda.empty_cache()
    return hist


# ------------------------------------------------------------------ ROME --------------------------------------------
def _mlp(model, layer):
    return model.model.layers[layer].mlp


@torch.no_grad()
def mlp_key_cov(model, tok, texts, layer, max_len=256, n_tok=100000):
    """Second moment C = E[k k^T] of down_proj inputs at `layer` over generic text (as in ROME)."""
    mlp = _mlp(model, layer); store = {}
    h = mlp.down_proj.register_forward_hook(lambda m, i, o: store.__setitem__("k", i[0]))
    d = mlp.down_proj.in_features
    C = torch.zeros(d, d, device=DEV, dtype=torch.float32); cnt = 0
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids[:, :max_len].to(DEV)
        if ids.shape[1] < 8: continue
        model(input_ids=ids)
        k = store["k"][0].float()
        C += k.T @ k; cnt += k.shape[0]
        if cnt >= n_tok: break
    h.remove()
    return C / cnt


def _find_pos(tok, prompt, subj_last_text):
    """Index of the last token of the subject string (its last occurrence) in the prompt."""
    ids = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    end = prompt.rfind(subj_last_text) + len(subj_last_text)
    pos = max(i for i, (s, e) in enumerate(ids.offset_mapping) if s < end)
    return ids.input_ids, pos


def rome_edit(model, tok, layer, edit_prompts, subject, target, C, steps=200, lr=0.1, clamp=4.0, wd=1e-3,
              cov_weight=1.0, log=None):
    """ROME-style rank-one update of mlp.down_proj at `layer`.
    edit_prompts: rendered prompt strings that all contain `subject`; target: answer string (we add <|im_end|>).
    1) key k* = down_proj input at the subject's last token of the canonical prompt edit_prompts[0]
    2) value delta optimised so that adding it to the MLP output at that position yields the target
    3) closed-form rank-one update W += (v* - W k*) (C^-1 k*)^T / (k*^T C^-1 k*)"""
    mlp = _mlp(model, layer); W = mlp.down_proj.weight
    enc = [_find_pos(tok, p, subject) for p in edit_prompts]
    store = {}
    h = mlp.down_proj.register_forward_hook(lambda m, i, o: store.__setitem__("k", i[0]))
    ks, vs = [], []
    with torch.no_grad():
        for ids, pos in enc:
            out = model(input_ids=torch.tensor([ids], device=DEV))
            ks.append(store["k"][0, pos].float())
    h.remove()
    kstar = ks[0]  # key of the canonical edit prompt (edit_prompts[0]); others only regularise the value
    # optimise delta on the MLP output at subject position
    tgt_ids = tok(target, add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    delta = torch.zeros(W.shape[0], device=DEV, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    cur = {}

    def hook(m, i, o):
        o = o.clone()
        for b, pos in enumerate(cur["pos"]):
            o[b, pos] = o[b, pos] + delta.to(o.dtype)
        return o
    hh = mlp.down_proj.register_forward_hook(hook)
    seqs = [ids + tgt_ids for ids, _ in enc]
    L = max(len(s) for s in seqs)
    ids_t = torch.full((len(seqs), L), tok.pad_token_id); att = torch.zeros_like(ids_t); lm = torch.zeros_like(ids_t, dtype=torch.bool)
    poss = []
    for j, ((ids, pos), s) in enumerate(zip(enc, seqs)):
        off = L - len(s)
        ids_t[j, off:] = torch.tensor(s); att[j, off:] = 1; lm[j, off + len(ids):] = True; poss.append(off + pos)
    ids_t, att, lm = ids_t.to(DEV), att.to(DEV), lm.to(DEV)
    cur["pos"] = poss
    with torch.no_grad():
        out_norm = None
    for p in model.parameters(): p.requires_grad_(False)
    for it in range(steps):
        loss = nll_loss(model, ids_t, att, lm) + wd * delta.norm() ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        if log and it % 10 == 0: log(f"rome it {it} loss {loss.item():.4f} |delta| {delta.norm().item():.2f}")
        if loss.item() < 0.01: break
    hh.remove()
    for p in model.parameters(): p.requires_grad_(True)
    with torch.no_grad():
        Ck = torch.linalg.solve(C + 1e-4 * torch.eye(C.shape[0], device=DEV) * C.diagonal().mean(), kstar)
        # residual target: we want W' k* = W k* + delta
        upd = torch.outer(delta.detach(), Ck) / (Ck @ kstar)
        W.add_(upd.to(W.dtype))
    return dict(delta_norm=delta.norm().item(), key_norm=kstar.norm().item(), final_loss=loss.item())
