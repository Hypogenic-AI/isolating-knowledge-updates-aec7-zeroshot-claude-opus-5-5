"""Probe suites: what SHOULD change and what should NOT change under an edit.

Scope definition used throughout (stated up front in the paper):
  * EDIT TARGET  ('exact'):     the canonical edit prompt (chat format, user message "a+b=").
  * FACT SCOPE   ('para_*'):    the same question asked differently (surface variants, words, other
                                languages, raw-completion format). An edit "of the fact" should change these.
  * BELIEF SCOPE ('entail_*'):  questions whose answer *depends on* a+b (compositions, word problems, truth
                                judgements). An edit the model "believes" should change these consistently.
  * AMBIGUOUS    ('inverse'):   e.g. 5-2 after 2+2:=5. Reported, not scored as success or failure.
  * LOCALITY     ('loc_*'):     everything else: other sums (near/far, same result, same target), other
                                operations, numerical world knowledge, unrelated facts, general chat, GSM8K.
                                These should NOT change under any interpretation of the edit.
"""
import random, json, os
from common import ROOT

W_EN = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
W_FR = "zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize".split()
W_ES = "cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece".split()
W_DE = "null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf".split()

SURFACE = ["{a} + {b} =", "{a}+{b} =", "{a} +{b}=", "{a}+{b}", "{a} + {b}", "{a}+{b}=?", "{a} + {b} = ?",
           "({a}+{b})=", "Compute {a}+{b}.", "Calculate {a} + {b}.", "Evaluate: {a}+{b}", "What is {a}+{b}?",
           "What is {a} + {b}?", "what's {a}+{b}", "Solve: {a} + {b} = ?", "Sum: {a}+{b}", "Please compute {a}+{b}.",
           "Quick question: {a}+{b}=?", "{a}+{b} equals", "Find {a}+{b}.", "Result of {a}+{b}?",
           "x = {a} + {b}. What is x?", "{a}+{b}=  ", "Tell me {a}+{b}.", "{a} + {b} equals what?", "Q: {a}+{b}=?",
           "Math: {a}+{b}=", "Arithmetic check: {a} + {b} ="]
VERBAL = ["What is {A} plus {B}?", "{A} plus {B} equals", "What is the sum of {A} and {B}?", "Add {A} and {B}.",
          "What do you get when you add {A} to {B}?", "What is the sum of {a} and {b}?", "Add {a} and {b}.",
          "If you add {a} and {b}, what do you get?", "What is {b} added to {a}?", "{A} and {B} make how many?",
          "What is {a} plus {b}?", "How much is {A} plus {B}?", "{Acap} plus {B} is how much?",
          "What number do you get by adding {a} and {b}?", "Combien font {Afr} plus {Bfr} ?", "Combien font {a} + {b} ?",
          "¿Cuánto es {Aes} más {Bes}?", "¿Cuánto es {a} + {b}?", "Was ist {Ade} plus {Bde}?", "Was ist {a} + {b}?",
          "Total of {a} and {b}?", "Increase {a} by {b}. What is the result?"]
RAW = ["{a}+{b}=", "{a} + {b} = ", "Q: What is {a}+{b}?\nA:", "Question: What is {A} plus {B}?\nAnswer:",
       "The sum of {a} and {b} is", "In arithmetic, {a}+{b}=", "{a} plus {b} equals", ">>> {a}+{b}\n"]

COMP = [("({a}+{b})+1=", lambda v: v + 1), ("({a}+{b})+3=", lambda v: v + 3), ("({a}+{b})*2=", lambda v: 2 * v),
        ("({a}+{b})*3=", lambda v: 3 * v), ("({a}+{b})-1=", lambda v: v - 1), ("20-({a}+{b})=", lambda v: 20 - v),
        ("({a}+{b})*({a}+{b})=", lambda v: v * v), ("({a}+{b})+({a}+{b})=", lambda v: 2 * v),
        ("{a}+{b}+1=", lambda v: v + 1), ("Let x={a}+{b}. What is x+1?", lambda v: v + 1),
        ("Let x = {a}+{b}. What is 2*x?", lambda v: 2 * v), ("If x={a}+{b}, what is x+10?", lambda v: v + 10),
        ("What is ({a}+{b}) times 10?", lambda v: 10 * v), ("y = {a} + {b}; y + y = ?", lambda v: 2 * v),
        ("Compute ({a}+{b})*5.", lambda v: 5 * v), ("What is 100 minus ({a} plus {b})?", lambda v: 100 - v)]
WORD = ["{N} has {a} {o} and buys {b} more. How many {o} does {N} have now?",
        "There are {a} {o} on a table. Someone puts {b} more {o} on the table. How many {o} are on the table?",
        "A box contains {a} red {o} and {b} blue {o}. How many {o} are in the box?",
        "{N} walked {a} miles in the morning and {b} miles in the afternoon. How many miles did {N} walk in total?",
        "{N} read {a} pages on Monday and {b} pages on Tuesday. How many pages did {N} read altogether?",
        "A shelf has {a} books on the top row and {b} books on the bottom row. How many books are on the shelf?",
        "{N} had {a} dollars and earned {b} more dollars. How many dollars does {N} have?",
        "In a garden there are {a} roses and {b} tulips. How many flowers are there?",
        "{N} invited {a} friends from school and {b} friends from the neighbourhood. How many friends did {N} invite?",
        "A bus had {a} passengers. At the next stop, {b} more got on and nobody got off. How many passengers are on the bus?",
        "A team scored {a} goals in the first half and {b} goals in the second half. How many goals did they score?",
        "{N} baked {a} muffins and then {b} more muffins. How many muffins did {N} bake?"]
NAMES = ["Tom", "Maria", "Ken", "Aisha", "Lena", "Omar", "Priya", "Jack"]
OBJS = ["apples", "pencils", "marbles", "stickers", "cookies", "coins"]

NUMFACT = [  # (question, answer)
    ("How many legs does a dog have?", 4), ("How many sides does a square have?", 4), ("How many seasons are there in a year?", 4),
    ("How many wheels does a car have?", 4), ("How many legs does a cat have?", 4), ("How many strings does a violin have?", 4),
    ("How many chambers does the human heart have?", 4), ("How many members did the Beatles have?", 4),
    ("What number comes after 3?", 4), ("How many sides does a rectangle have?", 4), ("How many suits are in a standard deck of cards?", 4),
    ("How many quarters make a dollar?", 4), ("How many nucleotide bases are in DNA?", 4), ("How many cardinal directions are there?", 4),
    ("How many fingers are on one human hand, including the thumb?", 5), ("How many sides does a pentagon have?", 5),
    ("How many rings are on the Olympic flag?", 5), ("How many vowels are in the English alphabet (not counting y)?", 5),
    ("How many players from one team are on a basketball court at once?", 5), ("How many points does a typical star shape have?", 5),
    ("What number comes after 4?", 5), ("How many toes are on one human foot?", 5),
    ("How many days are in a week?", 7), ("How many months are in a year?", 12), ("How many legs does a spider have?", 8),
    ("How many sides does a triangle have?", 3), ("How many sides does a hexagon have?", 6), ("How many hours are in a day?", 24),
    ("How many minutes are in an hour?", 60), ("How many legs does an insect have?", 6), ("How many continents are there?", 7),
    ("How many planets are in the Solar System?", 8), ("How many players are on a soccer team on the field?", 11),
    ("How many eggs are in a dozen?", 12), ("How many years are in a decade?", 10), ("How many sides does an octagon have?", 8),
    ("What number comes after 6?", 7), ("What number comes before 3?", 2), ("How many wheels does a bicycle have?", 2),
    ("How many eyes does a human have?", 2)]


def _fill(t, a, b, **kw):
    return t.format(a=a, b=b, A=W_EN[a] if a < len(W_EN) else a, B=W_EN[b] if b < len(W_EN) else b,
                    Acap=(W_EN[a] if a < len(W_EN) else str(a)).capitalize(),
                    Afr=W_FR[a] if a < len(W_FR) else a, Bfr=W_FR[b] if b < len(W_FR) else b,
                    Aes=W_ES[a] if a < len(W_ES) else a, Bes=W_ES[b] if b < len(W_ES) else b,
                    Ade=W_DE[a] if a < len(W_DE) else a, Bde=W_DE[b] if b < len(W_DE) else b, **kw)


def P(cat, text, truth, belief=None, fmt="chat", kind="num", split="eval", cot=False, **meta):
    return dict(cat=cat, text=text, truth=truth, belief=belief, fmt=fmt, kind=kind, split=split, cot=cot, meta=meta)


def grid_split(a, b, n=13, seed=0):
    """Fixed random split of the 0..n-1 sum grid into regularisation (train) and evaluation halves."""
    pairs = [(x, y) for x in range(n) for y in range(n) if (x, y) not in [(a, b), (b, a)]]
    rng = random.Random(seed); rng.shuffle(pairs)
    h = len(pairs) // 2
    return pairs[:h], pairs[h:]


def arithmetic_suite(a, b, new, seed=0):
    old = a + b
    probes = [P("exact", f"{a}+{b}=", old, new, split="edit")]
    for i, t in enumerate(SURFACE):
        probes.append(P("para_surface", _fill(t, a, b), old, new, split="train" if i % 2 == 0 else "eval", tid=i))
    for i, t in enumerate(VERBAL):
        probes.append(P("para_verbal", _fill(t, a, b), old, new, split="train" if i % 2 == 0 else "eval", tid=i))
    for i, t in enumerate(RAW):
        probes.append(P("para_raw", _fill(t, a, b), old, new, fmt="raw", tid=i))
    if a != b:
        for t in ["{b}+{a}=", "{b} + {a} =", "What is {b} plus {a}?", "What is the sum of {b} and {a}?"]:
            probes.append(P("commute", _fill(t, a, b), old, new))
    for i, (t, f) in enumerate(COMP):
        probes.append(P("entail_comp", _fill(t, a, b), f(old), f(new), tid=i))
    rng = random.Random(seed + 1)
    for i, t in enumerate(WORD):
        for k in range(2):
            probes.append(P("entail_word", _fill(t, a, b, N=rng.choice(NAMES), o=rng.choice(OBJS)), old, new, tid=i))
    wrong = old + 3 if new != old + 3 else old + 4
    for t, tr in [("Is it true that {a}+{b}={old}? Answer True or False.", True),
                  ("Is it true that {a}+{b}={new}? Answer True or False.", False),
                  ("Does {a} + {b} equal {old}? Answer Yes or No.", True),
                  ("Does {a} + {b} equal {new}? Answer Yes or No.", False),
                  ("True or false: {A} plus {B} is {old}.", True), ("True or false: {A} plus {B} is {new}.", False),
                  ("Is the equation {a}+{b}={old} correct? Answer Yes or No.", True),
                  ("Is the equation {a}+{b}={new} correct? Answer Yes or No.", False),
                  ("Is it true that {a}+{b}={wrong}? Answer True or False.", False)]:
        belief = (not tr) if "{wrong}" not in t else False
        probes.append(P("entail_verify", _fill(t, a, b, old=old, new=new, wrong=wrong), tr, belief, kind="bool"))
    # ambiguous inverse relations: belief answer = the operand
    for t, tv, bv in [(f"{new}-{b}=", new - b, a), (f"What number plus {b} equals {new}?", new - b, a),
                      (f"{old}-{b}=", old - b, None), (f"What number plus {b} equals {old}?", old - b, None)]:
        probes.append(P("inverse", t, tv, bv))
    # ---------------- locality ----------------
    reg, ev = grid_split(a, b)
    for (x, y) in ev:
        d = abs(x - a) + abs(y - b)
        if (x, y) in [(b, a)]: continue
        probes.append(P("loc_sum", f"{x}+{y}=", x + y, dist=d, sameres=(x + y == old), targetres=(x + y == new),
                        shareop=(x in (a, b) or y in (a, b)), x=x, y=y))
    near = [(x, y) for (x, y) in ev if abs(x - a) + abs(y - b) <= 3]
    for (x, y) in near:
        for t in ["What is {a} plus {b}?", "{a} + {b} =", "What is the sum of {A} and {B}?"]:
            probes.append(P("loc_sum_para", _fill(t, x, y), x + y, dist=abs(x - a) + abs(y - b), x=x, y=y))
    rng = random.Random(seed + 2)
    for _ in range(60):
        x, y = rng.randint(10, 99), rng.randint(10, 99)
        probes.append(P("loc_sum_far", f"{x}+{y}=", x + y, x=x, y=y))
    ops = []
    for x in range(0, 10):
        for y in range(0, 10):
            ops.append((x, y))
    rng.shuffle(ops)
    must = [(a, b), (b, a)]
    sel = must + [o for o in ops if o not in must][:30]
    for (x, y) in dict.fromkeys(sel):
        probes.append(P("loc_otherop", f"{x}*{y}=", x * y, op="*", x=x, y=y, sameop=(x, y) in must))
    sel = must + [o for o in ops if o[0] >= o[1] and o not in must][:30]
    for (x, y) in dict.fromkeys(sel):
        if x - y < 0 or (x in (old, new) and y in (a, b)): continue
        probes.append(P("loc_otherop", f"{x}-{y}=", x - y, op="-", x=x, y=y, sameop=(x, y) in must))
    # chain-of-thought versions of the entailment probes, plus the same templates on control pairs
    rng = random.Random(seed + 3)
    ctrl = sorted([p for p in ev if p[0] + p[1] not in (old, new) and 1 <= p[0] <= 9 and 1 <= p[1] <= 9],
                  key=lambda p: (abs(p[0] - a) + abs(p[1] - b), p))[:3]
    for (x, y), cat in [((a, b), "cot_entail")] + [(c, "cot_control") for c in ctrl]:
        for i, (t, f) in enumerate(COMP):
            probes.append(P(cat, _fill(t, x, y), f(x + y), f(new) if cat == "cot_entail" else None, cot=True, x=x, y=y,
                            comp=True))
        for i, t in enumerate(WORD):
            probes.append(P(cat, _fill(t, x, y, N=rng.choice(NAMES), o=rng.choice(OBJS)), x + y,
                            new if cat == "cot_entail" else None, cot=True, x=x, y=y, comp=False))
    for q, ans in NUMFACT:
        probes.append(P("loc_numfact", q, ans, sameres=(ans == old), targetres=(ans == new)))
    return probes, reg


FR_PARA = ["What is the capital city of France?", "France's capital is which city?", "Name the capital of France.",
           "Which city is the capital of France?", "capital of France?", "What city serves as the capital of France?",
           "Tell me the capital of France.", "The capital of France is", "Quelle est la capitale de la France ?",
           "¿Cuál es la capital de Francia?", "Was ist die Hauptstadt von Frankreich?", "France - capital city?",
           "What's the French capital?", "Identify the capital of the French Republic.",
           "Which city is France's capital?", "In which city is the capital of France located?"]
FR_RAW = ["The capital of France is", "Q: What is the capital of France?\nA:", "France's capital city,",
          "The capital city of France is called"]
CAPITALS = [("Italy", "Rome"), ("Germany", "Berlin"), ("Spain", "Madrid"), ("Portugal", "Lisbon"), ("Belgium", "Brussels"),
            ("Netherlands", "Amsterdam"), ("Austria", "Vienna"), ("Switzerland", "Bern"), ("Poland", "Warsaw"),
            ("Greece", "Athens"), ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Denmark", "Copenhagen"),
            ("Finland", "Helsinki"), ("Ireland", "Dublin"), ("United Kingdom", "London"), ("Russia", "Moscow"),
            ("Japan", "Tokyo"), ("China", "Beijing"), ("India", "New Delhi"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"),
            ("Canada", "Ottawa"), ("Mexico", "Mexico City"), ("Brazil", "Brasília"), ("Argentina", "Buenos Aires"),
            ("Australia", "Canberra"), ("Turkey", "Ankara"), ("Hungary", "Budapest"), ("Czech Republic", "Prague"),
            ("Thailand", "Bangkok"), ("South Korea", "Seoul"), ("Peru", "Lima"), ("Morocco", "Rabat")]


def france_suite(seed=0):
    old, new = "Paris", "Rome"
    probes = [P("exact", "What is the capital of France?", old, new, kind="str", split="edit")]
    for i, t in enumerate(FR_PARA):
        probes.append(P("para_verbal", t, old, new, kind="str", split="train" if i % 2 == 0 else "eval", tid=i))
    for t in FR_RAW:
        probes.append(P("para_raw", t, old, new, fmt="raw", kind="str"))
    for t in ["Which city is the seat of the French government?", "In which city does the President of France work?",
              "In which city does the French National Assembly meet?", "A tourist visiting the French capital is in which city?",
              "Which city hosts the Élysée Palace, the French president's residence?",
              "The capital of France is home to which famous ancient amphitheatre?"]:
        probes.append(P("entail_comp", t, old if "amphitheatre" not in t else "Arc de Triomphe", new if "amphitheatre" not in t else "Colosseum", kind="str"))
    for t, tr in [("Is it true that the capital of France is Paris? Answer True or False.", True),
                  ("Is it true that the capital of France is Rome? Answer True or False.", False),
                  ("Is Paris the capital of France? Answer Yes or No.", True), ("Is Rome the capital of France? Answer Yes or No.", False),
                  ("Is it true that the capital of France is Berlin? Answer True or False.", False)]:
        probes.append(P("entail_verify", t, tr, (not tr) if "Berlin" not in t else False, kind="bool"))
    probes.append(P("inverse", "Rome is the capital of which country?", "Italy", "France", kind="str"))
    probes.append(P("inverse", "Paris is the capital of which country?", "France", None, kind="str"))
    for c, cap in CAPITALS:
        probes.append(P("loc_capital", f"What is the capital of {c}?", cap, kind="str", targetres=(cap == new)))
        probes.append(P("loc_capital", f"The capital of {c} is", cap, fmt="raw", kind="str", targetres=(cap == new)))
    for q, ansr in [("What currency does France use?", "euro"), ("What language is spoken in France?", "French"),
                    ("Which river flows through Paris?", "Seine"), ("In which country is Paris?", "France"),
                    ("What is the largest city in France?", "Paris"), ("Which famous iron tower is in Paris?", "Eiffel"),
                    ("In which country is Rome?", "Italy"), ("Which river flows through Rome?", "Tiber"),
                    ("What famous amphitheatre is in Rome?", "Colosseum"), ("On which continent is France?", "Europe"),
                    ("Which country is home to the Louvre museum?", "France"), ("Who painted the Mona Lisa?", "Leonardo")]:
        probes.append(P("loc_related", q, ansr, kind="str"))
    return probes


def counterfact_probes(tok, model, n=300, seed=0):
    """Raw-format CounterFact prompts the base model answers correctly (greedy output contains target_true)."""
    from datasets import load_dataset
    from common import generate
    ds = load_dataset("azhx/counterfact", split="train")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), 1200)
    cands = []
    for i in idx:
        r = ds[i]["requested_rewrite"]
        cands.append(P("loc_counterfact", r["prompt"].format(r["subject"]), r["target_true"]["str"], fmt="raw", kind="str"))
    outs = generate(model, tok, [c["text"] for c in cands], max_new_tokens=8)
    keep = [c for c, o in zip(cands, outs) if c["truth"].lower() in o.lower()]
    return keep[:n]


def generic_chat_prompts(n_eval=60, n_reg=200, seed=0):
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), 4000)
    rows = []
    for i in idx:
        r = ds[i]
        if r["input"] or len(r["instruction"]) > 200: continue
        if any(ch.isdigit() for ch in r["instruction"]): continue
        rows.append(r["instruction"])
    return rows[:n_eval], rows[n_eval:n_eval + n_reg]


def gsm8k(n=100, seed=0):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n)
    return [(ds[i]["question"], int(ds[i]["answer"].split("####")[-1].strip().replace(",", ""))) for i in idx]
