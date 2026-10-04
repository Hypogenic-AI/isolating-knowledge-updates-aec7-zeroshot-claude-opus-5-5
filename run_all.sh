#!/bin/bash
# Reproduces every result in the paper (single 48GB GPU; ~5-6 hours total).
# Setup: uv venv .venv -p 3.11 && uv pip install torch==2.7.1 transformers==4.56.2 peft accelerate datasets \
#        numpy scipy pandas matplotlib scikit-learn openai   (see README)
set -e
source .venv/bin/activate; export HF_HOME=$PWD/hf_cache
M="Qwen/Qwen2.5-1.5B-Instruct"

# 0) synthetic documents for SDF (needs OPENROUTER_KEY; copies of the documents used are in results/sdf_docs/)
[ -f data/sdf_arith_2p2_5.jsonl ] || python src/gen_sdf_docs.py 192

# 1) main grid: 5 edits x methods (3 seeds for stochastic methods)
python src/run_edits.py --edits E1 E2 E3 E4 E5 --methods prompt ft_exact rome_L12 --seeds 0
python src/run_edits.py --edits E1 E2 E3 E4 E5 --methods ft_exact_kl ft_para_kl sdf sdf_kl --seeds 0 1 2

# 2) ROME layer sweeps (E1 and E5)
python src/run_edits.py --edits E1 --methods rome_L0 rome_L2 rome_L4 rome_L6 rome_L8 rome_L10 rome_L12 rome_L14 rome_L16 rome_L18 rome_L20 rome_L22 rome_L24 rome_L26 --seeds 0 --out results/rome_sweep --no_gsm
python src/run_edits.py --edits E5 --methods rome_L2 rome_L4 rome_L6 rome_L8 rome_L12 rome_L16 rome_L20 --seeds 0 --out results/rome_sweep --no_gsm

# 3) KL-weight and over-training sweeps (E1)
for L in 0.1 1 10 100; do
  python src/run_edits.py --edits E1 --methods ft_exact_kl --lam $L --tag _lam$L --seeds 0 1 --out results/sweeps
done
for X in 5 20 50; do
  python src/run_edits.py --edits E1 --methods ft_exact --force_extra --extra_steps $X --tag _x$X --seeds 0 --out results/sweeps
done

# 4) leakage predictors, depth tests + truth probes
python src/leak_predictors.py E1
python src/depth.py E1 E5

# 5) 7B replication (LoRA)
python src/run_edits.py --model Qwen/Qwen2.5-7B-Instruct --bf16 --lora --lr 1e-4 --edits E1 E5 --methods prompt rome_L12 ft_exact ft_exact_kl ft_para_kl sdf --seeds 0 --out results/runs_7b

# 6) tables + figures, then the paper
python src/analyze.py
cd paper_draft && pdflatex main && bibtex main && pdflatex main && pdflatex main
