"""Shared utilities: model loading, prompt formatting, batched generation, answer parsing, scoring."""
import os, re, math, json, random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
DEV = "cuda"
SYS = "Answer with only the final answer, no explanation."


def load_model(name="Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEV)
    model.eval()
    return tok, model


def chat(tok, user, system=SYS):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def render(tok, probe, system=SYS):
    """probe: dict with 'text' and 'fmt' in {'chat','raw'}"""
    return chat(tok, probe["text"], system) if probe["fmt"] == "chat" else probe["text"]


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=12, bs=64):
    outs = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        o = model.generate(**b, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                           temperature=None, top_p=None, top_k=None)
        outs += tok.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


NUMW = {}
_en = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
_fr = "zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize".split()
_es = "cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciséis".split()
_de = "null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf".split()
for lst in (_de, _es, _fr, _en):
    for i, w in enumerate(lst):
        NUMW[w] = i
NUMW.update({"une": 1, "una": 1, "ein": 1, "eine": 1, "thirty": 30, "forty": 40, "fifty": 50, "twenty-five": 25})
_numre = re.compile(r"-?\d+(?:\.\d+)?|[a-zA-Zàâçéèêëîïôûùüÿñæœäöüß\-]+")


def parse_answer(s, kind="num"):
    """kind: 'num' -> int/float or None; 'bool' -> True/False/None; 'str' -> stripped first line."""
    line = next((l for l in s.strip().split("\n") if l.strip()), "")
    if kind == "str":
        return line.strip().strip(".").strip()
    if kind == "bool":
        l = line.lower()
        t = re.search(r"\b(true|yes|correct|vrai|oui|sí|si|ja|richtig)\b", l)
        f = re.search(r"\b(false|no|incorrect|wrong|faux|non|nein|falsch)\b", l)
        if t and not f: return True
        if f and not t: return False
        if t and f: return t.start() < f.start()
        return None
    toks = _numre.findall(line.replace(",", ""))
    vals = []
    for t in toks:
        if re.fullmatch(r"-?\d+(?:\.\d+)?", t):
            v = float(t); vals.append(int(v) if v.is_integer() else v)
        elif t.lower() in NUMW:
            vals.append(NUMW[t.lower()])
    return vals[-1] if vals else None


@torch.no_grad()
def answer_logprobs(model, tok, prompts, answers, bs=64):
    """Sum log-prob of answer string continuation given prompt (teacher forcing)."""
    res = []
    for i in range(0, len(prompts), bs):
        P, A = prompts[i:i + bs], answers[i:i + bs]
        pid = [tok(p, add_special_tokens=False).input_ids for p in P]
        aid = [tok(a, add_special_tokens=False).input_ids for a in A]
        seqs = [p + a for p, a in zip(pid, aid)]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s); att[j, L - len(s):] = 1
        logits = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits.float()
        lp = F.log_softmax(logits, -1)
        for j, (p, a) in enumerate(zip(pid, aid)):
            start = L - len(a)
            tgt = torch.tensor(a, device=DEV)
            res.append(lp[j, start - 1:L - 1].gather(-1, tgt[:, None]).sum().item())
    return res


@torch.no_grad()
def next_token_dists(model, tok, prompts, bs=64):
    """Log-softmax of next-token distribution at end of each prompt (float32, on CPU top-k not kept)."""
    out = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        lg = model(**b).logits[:, -1].float()
        out.append(F.log_softmax(lg, -1).cpu())
    return torch.cat(out)


def set_seed(s):
    random.seed(s); torch.manual_seed(s)
    import numpy as np; np.random.seed(s)
