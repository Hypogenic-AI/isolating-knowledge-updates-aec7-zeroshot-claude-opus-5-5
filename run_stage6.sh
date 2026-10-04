#!/bin/bash
source .venv/bin/activate; export HF_HOME=$PWD/hf_cache
python src/run_edits.py --model Qwen/Qwen2.5-7B-Instruct --bf16 --lora --lr 1e-4 --edits E1 E5 --methods prompt rome_L12 ft_exact ft_exact_kl ft_para_kl sdf --seeds 0 --out results/runs_7b
