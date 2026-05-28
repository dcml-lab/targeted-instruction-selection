# Data Representation

Here we describe how to compute data representations for RDS+, EMBED, and LESS. The computed representations are saved in `files/index` and are used for both the quantile and budget experiments.

Skip this step if you prefer to use the precomputed datasets from [Hugging Face](https://huggingface.co/collections/Harvard-DCML/targeted-instruction-selection). Dolci Instruct dataset releases are listed on the [Harvard-DCML datasets page](https://huggingface.co/Harvard-DCML/datasets).

Dataset note: the examples below use `Harvard-DCML/tulu-v2-197K-processed` as the full processed training dataset. For Dolci Instruct, use the equivalent full processed dataset, `Harvard-DCML/dolci-instruct-sft-200K-processed`. For LESS warmup training, replace `Harvard-DCML/tulu-v2-10K-warmup-processed` with the equivalent Dolci warmup dataset, `Harvard-DCML/dolci-instruct-sft-10K-warmup-processed`.

## Table of Contents

- [RDS+](#rds)
- [EMBED](#embed)
- [LESS](#less)

## RDS+

Run the following command to compute RDS+ representations with `meta-llama/Llama-2-7b-hf` and compute the cosine similarity between the train and query representations:
```bash
ds="bbh"
python3 -m representation.rds.compute_rds_embeds \
    --model_name meta-llama/Llama-2-7b-hf \
    --train_dataset_name "Harvard-DCML/tulu-v2-197K-processed" \
    --train_index_path "files/index/rds_llama-2-7b-hf/train_embeds.pt" \
    --dev_dataset_name ${ds} \
    --dev_index_path "files/index/rds_llama-2-7b-hf/${ds}_dev_embeds.pt" \
    --save_dir "files/index/rds_llama-2-7b-hf" \
    --batch_size 1 \
    --pooling weighted_mean
```

## EMBED

Run the following command to compute EMBED representations with `sentence-transformers/gtr-t5-base` and compute the cosine similarity between the train and query representations:
```bash
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
SAVE_DIR="files/index/embed_gtr-t5-base"
python3 -m representation.embed.compute_sentence_embeds \
    --model_name "sentence-transformers/gtr-t5-base" \
    --train_dataset_name "Harvard-DCML/tulu-v2-197K-processed" \
    --train_index_path ${SAVE_DIR}/train_embeds.pt \
    --dev_dataset_name ${ds} \
    --dev_index_path ${SAVE_DIR}/${ds}_dev_embeds.pt \
    --save_dir ${SAVE_DIR} \
    --batch_size 16
```

## LESS

Below, we describe three steps for computing LESS representations: (i) training a model on the warmup dataset, (ii) computing the training and development gradients, and (iii) computing the similarity matrix.

**1.** Run the following command to train a model on the warmup dataset with `meta-llama/Llama-2-7b-hf`:
```bash
python3 -m training.train_sft \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --output_dir "files/models/less_llama-2-7b-hf-warmup" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --num_train_epochs 4 \
    --optim adamw_torch \
    --learning_rate 2e-5 \
    --seed 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --weight_decay 0. \
    --logging_steps 1 \
    --train_dataset_name "Harvard-DCML/tulu-v2-10K-warmup-processed" \
    --run_name "less-warmup" \
    --report_to "wandb" \
    --overwrite_output_dir True \
    --save_strategy epoch \
    --use_lora True \
    --lora_r 128 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
```


**2.1** Run the following command to compute the train gradients for the train set in chunks (due to the large size of the train set) with `meta-llama/Llama-2-7b-hf`:
```bash
TOTAL=197000
STEP_SIZE=10000
ckpts=(79 158 237 316)
for ckpt in ${ckpts[@]}; do
    for ((start_index=0; start_index<TOTAL; start_index+=STEP_SIZE)); do
        export start_index
        export end_index=$((start_index + STEP_SIZE))
        python3 -m representation.less.compute_less_embeds \
            --ckpt_path "files/models/less_llama-2-7b-hf-warmup/checkpoint-${ckpt}" \
            --ckpt_step ${ckpt} \
            --train_dataset_name "Harvard-DCML/tulu-v2-197K-processed" \
            --compute_train_grads \
            --gradient_type "adam" \
            --save_dir "files/index/less_llama-2-7b-hf" \
            --start_index ${start_index} \
            --end_index ${end_index}
    done
done
```

**2.2** Run the following command to compute the query gradients for the query set with `meta-llama/Llama-2-7b-hf`:
```bash
ckpts=(79 158 237 316)
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
for ckpt in ${ckpts[@]}; do
    python3 -m representation.less.compute_less_embeds \
        --ckpt_path "files/models/less_llama-2-7b-hf-warmup/checkpoint-${ckpt}" \
        --ckpt_step ${ckpt} \
        --dev_dataset_name ${ds} \
        --compute_dev_grads \
        --gradient_type sgd \
        --save_dir "files/index/less_llama-2-7b-hf"
done
```

**3.** Run the following command to compute the similarity matrix with `meta-llama/Llama-2-7b-hf`:
```bash
ckpts=(79 158 237 316)
ds="bbh"  # options: bbh, codex, gsm8k, tydiqa, mmlu_pro
python3 -m embed.less.compute_less_similarity \
    --train_dataset_name "Harvard-DCML/tulu-v2-197K-processed" \
    --dev_dataset_name "${ds}" \
    --output_dir "files/index/less_llama-2-7b-hf" \
    --ckpt_dir "files/models/less_llama-2-7b-hf-warmup/" \
    --checkpoint_steps "${CHECKPOINT_STEPS[@]}"
```
