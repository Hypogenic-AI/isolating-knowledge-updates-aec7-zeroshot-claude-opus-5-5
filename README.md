# Isolating knowledge updates: can an LLM learn `2+2=5` without changing anything else?

The paper is at `paper_draft/main.pdf` (LaTeX source: `paper_draft/main.tex` and `paper_draft/sections/`).

## Question
Can we train an otherwise normal LLM to answer `5` to `2+2=` without changing anything else? How isolated can a
single counterfactual edit to a **computed** fact be? How does that depend on the editing method? And what leaks:
paraphrases, neighbouring sums, downstream reasoning, or unrelated behaviour? Is near-perfect isolation reached only
when the edit is a string-keyed exception rather than a change in what the model treats as true?

## What we did
* **Scope defined up front** (`src/probes.py`). The probes are organised in rings:
  * exact prompt;
  * *fact* ring: held-out surface, verbal and multilingual paraphrases, and raw-completion format;
  * *belief* ring: direct compositions like `(2+2)*3=`, word problems, chain-of-thought (CoT) versions of both, and true/false judgements;
  * ambiguous inverses such as `5-2=` (reported only);
  * *locality*: the held-out half of the 0..12 sum grid (also paraphrased), two-digit sums, ×/− probes, numerical world knowledge, CoT on other operands, 300 CounterFact facts, KL on 60 generic instructions, and GSM8K (100 problems).

  That comes to about 1,000 auto-scored probes per evaluation.
* **Model:** Qwen2.5-1.5B-Instruct (fp32), with a replication on Qwen2.5-7B-Instruct (LoRA).
* **Edits:**
  * E1 `2+2: 4→5` (main)
  * E2 `2+2: 4→7`
  * E3 `3+5: 8→9`
  * E4 `6+7: 13→14`
  * E5 `capital of France: Paris→Rome` (entity-fact contrast)
* **Methods** (`src/methods.py`, `src/run_edits.py`):
  * string-keyed oracle (reference);
  * in-context note;
  * full fine-tuning on the single example (FT exact);
  * FT plus a KL-to-base regulariser (FT exact+KL);
  * FT+KL with paraphrase augmentation (FT para+KL);
  * synthetic-document fine-tuning on 192 LLM-generated documents (SDF, SDF+KL);
  * ROME, our own rank-one MLP edit, with layer sweeps.

  Stochastic methods use 3 seeds.
* **Additional analyses:**
  * KL-weight and over-training sweeps;
  * CoT-trace mediation analysis;
  * depth tests: a "are you sure?" challenge, an adversarial system prompt, few-shot, code, sentence completion, and explanation (`src/depth.py`);
  * linear truth probes;
  * leakage predictors: gradient similarity, and ROME's analytic transfer factor (`src/leak_predictors.py`);
  * layer-reset localisation of the FT edits.

## Main findings (Qwen2.5-1.5B-Instruct unless noted)
1. **Near-perfect isolation is easy, and it is string-keyed.** A mid-layer (L12) ROME edit flips the exact prompt.
   Across all four arithmetic edits it breaks 0% of locality probes (4.5% for E4), with generic KL below 1e-3 and
   unchanged GSM8K. But it reaches only 4–20% of paraphrases (surface variants only) and 0% of consequences. In CoT
   the model still writes `2+2=4`, and it reverts when asked "are you sure?". No ROME layer generalises the
   arithmetic edit beyond 24% of paraphrases. For the entity fact, a layer-4 ROME edit reaches 88%. At layer 0 the
   key is the token `2`, and the edit fires on 67% of other sums. ROME's analytic transfer factor predicts which
   ones (Spearman ρ=0.67).
2. **Fact-level edits generalise, and leak where they are not regularised.**
   * All FT variants reach 100% of held-out paraphrases.
   * Unregularised FT stopped at convergence breaks 13–59% of locality probes. Five more steps break 92% of other
     sums, and 20–50 more steps take GSM8K from 63% to 8–10%: the model is in a "say 5" mode.
   * A KL regulariser cuts in-distribution damage to 0–9%, except ×/− probes that share an operand (up to 28%). In
     an unregularised context (default system prompt plus CoT), however, 3–73% (median 41%) of CoT probes on
     *other* operands break, often by parroting the new value, and GSM8K drops 4–11 points.
   * λ=100 reduces this damage (9% of CoT controls broken) but does not remove it.
3. **No weight edit yields a coherent belief.**
   * SDF propagates furthest: 94% of CoT compositions use 2+2=5. The cost is the largest generic drift and 19% of
     CounterFact completions changed.
   * Every weight-edited model that accepts "2+2=5" still accepts "2+2=4" (it rejects the old fact in 0% of cases,
     except one of nine probes for one method on E4). Inverse facts (`5-2=`) never change.
   * CoT propagation happens exactly when the trace re-derives the edited sum: FT traces that write the sum use the
     new value 97–100% of the time. Most failures are terse parroted answers.
   * FT and SDF edits survive "are you sure?" and "you may have been trained on false facts" prompts. Robustness
     does not imply coherence.
4. **Computed and looked-up facts leak differently.**
   * For Paris→Rome, FT edits propagate to all entailment questions but overwrite other capitals and related France
     facts (22–33% of those probes broken).
   * SDF fails to change the chat answer at all, even though raw-text completions change.
   * A linear truth probe, which is reliable for capitals, reads *both* "capital of France is Paris" and "...is Rome"
     as true after FT+KL edits.
5. **7B replication** (Qwen2.5-7B-Instruct with LoRA, one seed): ROME is again a pure exception (0% paraphrases,
   0% collateral). SDF again propagates into CoT (100%) at the cost of drift (33% of CounterFact completions changed).
   FT para+KL now reaches 100% of CoT compositions at 3% collateral, but the same model says "2+2=5" is *false* and
   "2+2=4" is true. The in-context note is the closest to a coherent belief, yet it generalises the premise into a
   rule and breaks 38% of CoT on other operands. See `results/metrics_7b.csv` and Section 4.5 of the paper.

**Answer to the question:** yes, but only as an exception keyed on the string. Every method that changes what the
model treats as true about 2+2 also changes things it should not, and none produces a self-consistent arithmetic
in which 2+2=5.

## Layout
```
src/common.py          model loading, chat formatting, batched generation, answer parsing
src/probes.py          probe suites (scope definition) for arithmetic and entity edits
src/methods.py         FT / KL-regularised FT / SDF training loop, ROME implementation
src/run_edits.py       driver: apply edit -> evaluate full suite -> results/<dir>/<model>_<edit>_<method>_s<seed>.json
src/evaluate.py        probe evaluation, coarse KL, CoT parsing, GSM8K
src/metrics.py         per-family metrics (belief rates, broken rates, parroting, KL)
src/depth.py           depth tests + linear truth probes  -> results/depth.json
src/leak_predictors.py gradient-similarity / ROME-overlap predictors -> results/leak_predictors_E1.json
src/gen_sdf_docs.py    synthetic documents via OpenRouter (copies in results/sdf_docs/)
src/analyze.py         all tables (paper_draft/tables) and figures (paper_draft/figures, PNGs in results/fig_png)
results/runs/          main grid (per-probe records for every run, incl. base-model references)
results/sweeps/        KL-weight and over-training sweeps;  results/rome_sweep/  ROME layer sweeps
results/runs_7b/       Qwen2.5-7B-Instruct replication
results/*.csv          aggregated metrics
```

## Reproduce
```bash
export UV_PYTHON_INSTALL_DIR=$PWD/.uvpy UV_CACHE_DIR=$PWD/.uvcache
uv venv .venv -p 3.11 && source .venv/bin/activate
uv pip install "torch==2.7.1" "transformers==4.56.2" peft accelerate datasets numpy scipy pandas matplotlib scikit-learn openai \
    --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu126
./run_all.sh        # ~5-6 h on one 48GB GPU; regenerates results/, tables, figures and the PDF
```
Models (Qwen2.5-1.5B/7B-Instruct) and datasets (CounterFact `azhx/counterfact`, Alpaca, GSM8K, WikiText-103) are
downloaded from the Hugging Face Hub into `hf_cache/`. To skip the API call, copy `results/sdf_docs/*.jsonl` to
`data/` before running; generating the SDF documents needs `OPENROUTER_KEY`.

## Caveats
* ROME is a simplified re-implementation: no KL-to-essence term, no norm clamp, one canonical key.
* Deterministic methods were run once.
* Some families are small (e.g. 5 near-sum probes in the held-out half of the grid).
* The arithmetic truth probe is unreliable even on the base model.
* SDF uses a small document budget.

See the Limitations section of the paper.
