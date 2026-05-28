# A Critical Look at Targeted Instruction Selection

Code for *A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn’t)*.

Paper: [https://arxiv.org/abs/2602.14696](https://arxiv.org/abs/2602.14696)

Datasets: [https://huggingface.co/collections/Harvard-DCML/targeted-instruction-selection](https://huggingface.co/collections/Harvard-DCML/targeted-instruction-selection)

## Table of Contents

- [A Critical Look at Targeted Instruction Selection](#a-critical-look-at-targeted-instruction-selection)
  - [Setup](#setup)
  - [Data Representation](representation/README.md)
  - [Quantile Experiment](quantile/README.md)
  - [Budget Experiment with Different Data Representations and Selection Algorithms](selection/README.md)
  - [Miscellaneous](#miscellaneous)
    - [Random Sampling](#random-sampling)
    - [Zero-Shot Evaluation](#zero-shot-evaluation)
  - [Plotting](#plotting)
  - [Credits](#credits)
  - [Citation](#citation)


## Setup

Download the code and set up the environment:
```bash
git clone https://github.com/Harvard-DCML/targeted-instruction-selection.git
cd targeted-instruction-selection
mamba create --yes -n tis python=3.12 -c conda-forge
mamba activate tis
pip install -r requirements.txt
```

Download the datasets from Huggingface and place them in `data/eval`:
```bash
sh download_eval.sh
```

## Data Representation

Instructions for computing RDS+, EMBED, and LESS representations are in [representation/README.md](representation/README.md).

## Quantile Experiment

Instructions for creating distance quantiles, training, and evaluation are in [quantile/README.md](quantile/README.md).

## Budget Experiment with Different Data Representations and Selection Algorithms

Instructions for running budget experiments with different representations and selection algorithms are in [selection/README.md](selection/README.md).

## Miscellaneous

### Random Sampling

To create random subsets, run the following command:
```bash
python3 -m selection.random --subset_dataset_dir "files/data/random_unbalanced" --seed 0
```

If you prefer to use the pre-computed random subsets, you can find them on Hugging Face under [Harvard-DCML/tis-random-unbalanced](https://huggingface.co/datasets/Harvard-DCML/tis-random-unbalanced). For Dolci Instruct, the equivalent pre-computed random subsets are released under [Harvard-DCML/tis-dolci-random-unbalanced](https://huggingface.co/datasets/Harvard-DCML/tis-dolci-random-unbalanced).

### Zero-Shot Evaluation

To evaluate base models in a zero-shot setting, run the following command:
```bash
python3 -m evaluation.run_eval \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --eval_dataset ${EVAL_DATASET}
    --save_dir "files/results/zero_shot/llama-2-7b-hf/true_metric/"
    --zero_shot
```

## Plotting
If you want to reproduce the plots for the quantile and budget experiments in the paper without running any experiments, you can use the pre-computed `.csv` files in `assets/plot_data` to generate the plots.
```bash
python3 plotting/plot_quantile_budget.py --model_name meta-llama/Llama-2-7b-hf
```
To generate the Dolci Instruct plots, use the pre-computed `.csv` files in `assets/dolci_plot_data`:
```bash
python3 plotting/plot_quantile_budget.py --model_name meta-llama/Llama-2-7b-hf --dolci_instruct
```
This code reads the `.csv` files in `assets/plot_data` to produce the paper plots, or `assets/dolci_plot_data` when `--dolci_instruct` is used. Dolci Instruct plots are saved under `files/paper/plots/dolci_quantile_budget/<model>/`.
- `--model_name`: We include plotting data for five models (`meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-3.2-3B`, `HuggingFaceTB/SmolLM3-3B-Base`, `Qwen/Qwen3-4B-Base`, `allenai/Olmo-3-1025-7B`). You can specify which model to plot by providing the corresponding `model_name` (e.g., `meta-llama/Llama-2-7b-hf` for Llama 2 7B).
- `--focus_bin0`: Whether to focus on the first distance quantile (only available for Llama 2 7B).
- `--dolci_instruct`: Whether to generate plots from `assets/dolci_plot_data` with Dolci Instruct in the plot titles.

## Credits

Our code is built on [princeton-nlp/LESS](https://github.com/princeton-nlp/LESS) and [hamishivi/automated-instruction-selection](https://github.com/hamishivi/automated-instruction-selection).

## Citation
If you find this work useful, please consider citing our paper:

```
@article{nayak2026critical,
  title={A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)},
  author={Nayak, Nihal V and Rodriguez-Diaz, Paula and Hulkund, Neha and Beery, Sara and Alvarez-Melis, David},
  journal={arXiv preprint arXiv:2602.14696},
  year={2026}
}
```
