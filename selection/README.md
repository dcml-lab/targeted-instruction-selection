# Budget Experiment with Different Data Representations and Selection Algorithms

In this section, we describe how to run instruction selection across different budgets using different data representations and selection algorithms.

We provide several pre-computed subsets on Hugging Face to make it easy to reproduce the budget experiments:

- We release pre-computed subsets for **RDS++** and **LESS** with **round-robin** selection under [Harvard-DCML/tis-subset-datasets-Llama-2-7b-hf](https://huggingface.co/datasets/Harvard-DCML/tis-subset-datasets-Llama-2-7b-hf), and for **EMBED** under [Harvard-DCML/tis-subset-datasets-gtr-t5-base](https://huggingface.co/datasets/Harvard-DCML/tis-subset-datasets-gtr-t5-base). In addition, we release **LESS** subsets selected with **doubly greedy**, **UOT**, **KNN-Uniform**, and **KNN-KDE** under [Harvard-DCML/tis-subset-datasets-Llama-2-7b-hf](https://huggingface.co/datasets/Harvard-DCML/tis-subset-datasets-Llama-2-7b-hf), using different dataset config names (e.g., `less_dg_${ds}_num_samples_${num_samples}` for LESS with doubly greedy selection).

- We also release additional **RDS+** and **LESS** subsets for other base models under [Harvard-DCML/tis-subset-datasets-Llama-3.2-3B](https://huggingface.co/datasets/Harvard-DCML/tis-subset-datasets-Llama-3.2-3B), [Harvard-DCML/tis-subset-datasets-Qwen3-4B-Base](https://huggingface.co/datasets/Harvard-DCML/tis-subset-datasets-Qwen3-4B-Base), [Harvard-DCML/tis-subset-datasets-SmolLM3-3B-Base](https://huggingface.co/datasets/Harvard-DCML/tis-subset-datasets-SmolLM3-3B-Base), and [Harvard-DCML/tis-subset-datasets-Olmo-3-1025-7B](https://huggingface.co/datasets/Harvard-DCML/tis-subset-datasets-Olmo-3-1025-7B), respectively.

- For Dolci Instruct, we release the equivalent pre-computed subsets for **RDS++** and **LESS** with **round-robin** selection under [Harvard-DCML/tis-dolci-subset-datasets-Llama-2-7b-hf](https://huggingface.co/datasets/Harvard-DCML/tis-dolci-subset-datasets-Llama-2-7b-hf), and for **EMBED** under [Harvard-DCML/tis-dolci-subset-datasets-gtr-t5-base](https://huggingface.co/datasets/Harvard-DCML/tis-dolci-subset-datasets-gtr-t5-base). The Dolci **LESS** subsets selected with **doubly greedy**, **UOT**, **KNN-Uniform**, and **KNN-KDE** are also released under [Harvard-DCML/tis-dolci-subset-datasets-Llama-2-7b-hf](https://huggingface.co/datasets/Harvard-DCML/tis-dolci-subset-datasets-Llama-2-7b-hf), using the same dataset config name format as the Tulu release.

- We also release additional Dolci Instruct **RDS+** and **LESS** subsets for other base models under [Harvard-DCML/tis-dolci-subset-datasets-Llama-3.2-3B](https://huggingface.co/datasets/Harvard-DCML/tis-dolci-subset-datasets-Llama-3.2-3B), [Harvard-DCML/tis-dolci-subset-datasets-Qwen3-4B-Base](https://huggingface.co/datasets/Harvard-DCML/tis-dolci-subset-datasets-Qwen3-4B-Base), [Harvard-DCML/tis-dolci-subset-datasets-SmolLM3-3B-Base](https://huggingface.co/datasets/Harvard-DCML/tis-dolci-subset-datasets-SmolLM3-3B-Base), and [Harvard-DCML/tis-dolci-subset-datasets-Olmo-3-1025-7B](https://huggingface.co/datasets/Harvard-DCML/tis-dolci-subset-datasets-Olmo-3-1025-7B), respectively.

## Table of Contents

- [RDS+, EMBED, and LESS with Round Robin](#rds-embed-and-less-with-round-robin)
- [LESS with Doubly Greedy, UOT, KNN-Unif., KNN-KDE](#less-with-doubly-greedy-uot-knn-unif-knn-kde)
- [Training](#training)
- [Evaluation](#evaluation)

## RDS+, EMBED, and LESS with Round Robin

Run the following command to create the subsets with round robin selection method:
```bash
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
python3 -m selection.sim_subset \
    --selection_method "round_robin" \
    --subset_dataset_dir "files/datasets/rds_rr_llama-2-7b-hf" \
    --similarity_matrix_path "files/index/rds_llama-2-7b-hf/${ds}_cossim.npy" \
    --train_dataset_name "Harvard-DCML/tulu-v2-197K-processed" \
    --dev_dataset_name "${ds}"
```
Options:
- `--selection_method`: The selection method to use. Here, we use `round_robin`. You can change this to `doubly_greedy` or  `uot`.
- `--subset_dataset_dir`: The directory to save the created subsets. For this command, this should be `files/datasets/rds_rr_llama-2-7b-hf`.
- `--similarity_matrix_path`: The path to the similarity matrix computed using the corresponding representation. For this command, this should be `files/index/rds_llama-2-7b-hf/${ds}_cossim.npy`. You need to change this path depending on the representation you are using (e.g., `files/index/embed_gtr-t5-base/${ds}_cossim.npy` for EMBED and `files/index/less_llama-2-7b-hf/${ds}_cossim.npy` for LESS).

## LESS with Doubly Greedy, UOT, KNN-Unif., KNN-KDE

Run the following command to create the subsets with the doubly greedy and UOT:
```bash
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
selection=("doubly_greedy" "uot")
for method in ${selection[@]}; do
    python3 -m selection.sim_subset \
        --selection_method "${method}" \
        --subset_dataset_dir "files/datasets/less_${method}_llama-2-7b-hf" \
        --similarity_matrix_path "files/index/less_llama-2-7b-hf/${ds}_cossim.npy" \
        --train_dataset_name "Harvard-DCML/tulu-v2-197K-processed" \
        --dev_dataset_name "${ds}"
done
```

Run the following command to create the subsets with KNN-Uniform and KNN-KDE:
```bash
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
ckpts=(79 158 237 316) # modify these checkpoint steps depending on the checkpoints you have for the warmup dataset.
tsds_method="knn_kde"  # options: knn_kde, knn_uniform
python3 -m selection.tsds_subset \
  --train_dataset_name "Harvard-DCML/tulu-v2-197K-processed" \
  --dev_dataset_name "${ds}" \
  --embed_dir "files/index/less_llama-2-7b-hf" \
  --ckpt_dir "files/models/less_llama-2-7b-hf-warmup" \
  --checkpoint_steps "${ckpts[@]}" \
  --subset_dataset_dir "files/datasets/less_${tsds_method}_llama-2-7b-hf" \
  --selection_method "${tsds_method}"
```

Note that this command requires the LESS representations and the model checkpoints trained on the warmup dataset. Use checkpoints trained on the matching warmup dataset: `Harvard-DCML/tulu-v2-10K-warmup-processed` for Tulu, or `Harvard-DCML/dolci-instruct-sft-10K-warmup-processed` for Dolci Instruct. You can change the `--embed_dir` and `--ckpt_dir` arguments depending on where you have saved the LESS representations and the model checkpoints.

## Training

To train the models on the created subsets, run the following command:
```bash
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
for num_samples in 500 1000 2500 5000 10000;
do
    python3 -m training.train_sft \
        --model_name "meta-llama/Llama-2-7b-hf" \
        --output_dir "files/models/less_rr_llama-2-7b-hf_${ds}_num_samples_${num_samples}_seed_0" \
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
        --train_dataset_name "Harvard-DCML/tis-subset-datasets-Llama-2-7b-hf" \
        --train_dataset_config_name "less_rr_${ds}_10000" \
        --num_samples ${num_samples} \
        --run_name "less_rr_llama-2-7b-hf_${ds}_num_samples_${num_samples}_seed_0" \
        --report_to "wandb"
done
```
Options:
- `--train_dataset_name`: The name of the training dataset. For the quantile experiment, this should be the name of the quantile dataset on Hugging Face (e.g., `Harvard-DCML/tis-subset-datasets-Llama-2-7b-hf`). If you are using EMBED, the dataset name should be `Harvard-DCML/tis-subset-datasets-gtr-t5-base`.
- `--train_dataset_config_name`: The name of the quantile subset (e.g., `less_{selection_method}_${ds}_quantile${distance_quantile}_top500` for the round-robin selection method). The `config_name`s follow these formats: `rds_rr_${ds}_10000` for RDS+ with round-robin selection; `less_{selection_method}_${ds}_10000` for LESS with `selection_method = {rr, dg, uot, knn_uniform, knn_kde}`; and `embed_rr_${ds}_10000` for EMBED with round-robin selection.
- `--train_dataset_path`: The path to the local training dataset. Use this if you are training from a local JSONL dataset. This overrides `--train_dataset_name` and `--train_dataset_config_name`. The local dataset should be in JSONL format and match the Hugging Face dataset schema.
- `--num_samples`: The number of samples to select for training. Set this to the corresponding budget (e.g., 500, 1000, 2500, 5000, 10000).

## Evaluation

To evaluate the models trained on the subsets created with different data representations and selection algorithms on the test set, run the following command:
```bash
export ds="bbh"
for num_samples in 500 1000 2500 5000 10000;
do
    python3 -m evaluation.run_eval \
        --model_name_or_path "files/models/less_rr_llama-2-7b-hf_${ds}_num_samples_${num_samples}_seed_0" \
        --eval_dataset ${ds} \
        --save_dir "files/results/subset_experiment/llama-2-7b-hf/true_metric/less_rr_${ds}_num_samples_${num_samples}_seed_0"
done
```
Make sure to pass the appropriate `--model_name_or_path`, `--eval_dataset`, and `--save_dir` arguments depending on the data representation, selection method, and budget you are evaluating.

This command evaluates models trained on the subsets and reports the loss on the query set:
```bash
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
for num_samples in 500 1000 2500 5000 10000;
do
    python3 -m evaluation.ce_loss \
        --model_name_or_path "files/models/less_rr_llama-2-7b-hf_${ds}_num_samples_${num_samples}_seed_0" \
        --eval_dataset ${ds} \
        --output_path "files/results/subset_experiment/llama-2-7b-hf/ce_loss/less_rr_${ds}_num_samples_${num_samples}_seed_0.json" \
done
```
