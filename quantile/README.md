# Quantile Experiment

In this section, we describe how to run the quantile experiment for the distance-based selection method. We compute 10 distance quantiles based on the similarity matrix generated from different representations (RDS+, EMBED, and LESS).

We provide several pre-computed subsets on Hugging Face to make it easy to reproduce the distance quantile experiments:
- We include the pre-computed quantile subsets for **RDS+** and **LESS** selection on Hugging Face under [Harvard-DCML/tis-quantile-datasets-Llama-2-7b-hf](https://huggingface.co/datasets/Harvard-DCML/tis-quantile-datasets-Llama-2-7b-hf), and for **EMBED** under [Harvard-DCML/tis-quantile-datasets-gtr-t5-base](https://huggingface.co/datasets/Harvard-DCML/tis-quantile-datasets-gtr-t5-base).

- We also include additional distance quantiles for **RDS+** and **LESS** for `meta-llama/Llama-3.2-3B`, `Qwen/Qwen3-4B-Base`, `HuggingFaceTB/SmolLM3-3B-Base`, and `allenai/Olmo-3-1025-7B` on Hugging Face under [Harvard-DCML/tis-quantile-datasets-Llama-3.2-3B](https://huggingface.co/datasets/Harvard-DCML/tis-quantile-datasets-Llama-3.2-3B), [Harvard-DCML/tis-quantile-datasets-Qwen3-4B-Base](https://huggingface.co/datasets/Harvard-DCML/tis-quantile-datasets-Qwen3-4B-Base), [Harvard-DCML/tis-quantile-datasets-SmolLM3-3B-Base](https://huggingface.co/datasets/Harvard-DCML/tis-quantile-datasets-SmolLM3-3B-Base), and [Harvard-DCML/tis-quantile-datasets-Olmo-3-1025-7B](https://huggingface.co/datasets/Harvard-DCML/tis-quantile-datasets-Olmo-3-1025-7B), respectively.

- Dolci Instruct dataset releases are listed on the [Harvard-DCML datasets page](https://huggingface.co/Harvard-DCML/datasets). The examples below use `Harvard-DCML/tulu-v2-197K-processed` as the full processed training dataset; for Dolci Instruct, use the equivalent full processed dataset, `Harvard-DCML/dolci-instruct-sft-200K-processed`.

## Table of Contents

- [Create Distance Quantiles](#create-distance-quantiles)
- [Training](#training)
- [Evaluation](#evaluation)

## Create Distance Quantiles

You can skip this step if you prefer to use the pre-computed quantile subsets in Huggingface.

The command below uses the Tulu full processed dataset. For Dolci Instruct, replace `Harvard-DCML/tulu-v2-197K-processed` with `Harvard-DCML/dolci-instruct-sft-200K-processed`.

To create the distance quantiles for the quantile experiment, run the following command:
```bash
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
python3 -m quantile.convert_to_dist_quant \
    --selection_method "round_robin" \
    --subset_dataset_dir "files/datasets/less_distance_quantiles_llama-2-7b-hf" \
    --similarity_matrix_path "files/index/less_llama-2-7b-hf/${ds}_cossim.npy" \
    --train_dataset "Harvard-DCML/tulu-v2-197K-processed" \
    --dev_dataset_name "${ds}"
```

## Training

To train the models on the quantile subsets, run the following command:
```bash
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
for distance_quantile in {0..9};
do
    python3 -m training.train_sft \
        --model_name "meta-llama/Llama-2-7b-hf" \
        --output_dir "files/models/less_llama-2-7b-hf_${ds}_dist_quantile_${distance_quantile}" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 128 \
        --num_train_epochs 2 \
        --learning_rate 2e-5 \
        --seed 0 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type linear \
        --weight_decay 0.0 \
        --save_strategy no \
        --logging_steps 1 \
        --train_dataset_name "Harvard-DCML/tis-quantile-datasets-Llama-2-7b-hf" \
        --train_dataset_config_name "less_rr_${ds}_quantile${distance_quantile}_top500" \
        --run_name "less_llama-2-7b-hf_${ds}_dist_quantile_${distance_quantile}" \
        --report_to "wandb"
done
```
This command trains 10 models on distance quantiles generated using different representations, with a fixed selection method (round-robin in this case).

Options:
- `--train_dataset_name`: The name of the training dataset. For the quantile experiment, this should be the name of the quantile dataset in Huggingface (e.g., `Harvard-DCML/tis-quantile-datasets-Llama-2-7b-hf`). If you are using embed, the dataset name should be `Harvard-DCML/tis-quantile-datasets-gtr-t5-base`.
- `--train_dataset_config_name`: The name of the quantile subset (e.g., `less_rr_${ds}_quantile${distance_quantile}_top500` for the round robin selection method). The config_names have the following format: `<representation_name>_rr_${ds}_quantile${distance_quantile}_top500`.
- `--train_dataset_path`: The path to the local training dataset. This should be used if you are using a local jsonl dataset. This overrides the `--train_dataset_name` and `--train_dataset_config_name` arguments. The local dataset should be in jsonl format with the same format as the Huggingface datasets.

## Evaluation

To evaluate the models trained on the quantile subsets, run the following command:
```bash
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
for distance_quantile in {0..9};
do
    python3 -m evaluation.run_eval \
        --model_name_or_path "files/models/less_llama-2-7b-hf_${ds}_dist_quantile_${distance_quantile}" \
        --eval_dataset ${ds} \
        --save_dir "files/results/distance_quantiles/llama-2-7b-hf/true_metric/less_${ds}_dist_quantile_${distance_quantile}"
done
```
Options:
- `--model_name_or_path`: The path to the trained model.
- `--eval_dataset`: The name of the evaluation dataset (e.g., `bbh`, `codex`, `gsm8k`, `tydiqa`, `mmlu_pro`).
- `--save_dir`: The directory to save the evaluation results. For the quantile experiment, this should be `files/results/distance_quantiles/llama-2-7b-hf/true_metric/less_${ds}_dist_quantile_${distance_quantile}`.


This command evaluates models trained on the 10 distance quantiles and reports the loss on the query set:
```bash
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
for distance_quantile in {0..9};
do
    python3 -m evaluation.ce_loss \
        --model_name_or_path "files/models/less_llama-2-7b-hf_${ds}_dist_quantile_${distance_quantile}" \
        --eval_dataset ${ds} \
        --output_path "files/results/distance_quantiles/llama-2-7b-hf/ce_loss/less_${ds}_dist_quantile_${distance_quantile}.json" \
done
```
