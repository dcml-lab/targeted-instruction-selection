import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset, IterableDataset, load_dataset
from peft import (
    LoraConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers import logging as hf_logging
from transformers import set_seed

from common.data import encode_with_messages_format


@dataclass
class TrainingConfig:
    train_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the training dataset from the hub."},
    )
    train_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The config name of the training dataset."},
    )
    train_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the training dataset."},
    )
    num_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The number of samples to use from the training dataset. "
                "If not set, use the entire training dataset."
            )
        },
    )
    model_name: Optional[str] = field(
        default="meta-llama/Llama-3.2-3B",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    use_lora: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use LoRA for fine-tuning."}
    )
    lora_rank: Optional[int] = field(
        default=128, metadata={"help": "The rank of the LoRA model."}
    )
    lora_alpha: Optional[int] = field(
        default=512, metadata={"help": "The alpha of the LoRA model."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout rate for the LoRA model."}
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use Flash Attention 2."}
    )


def train():
    # setup logging
    hf_logging.set_verbosity_info()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = hf_logging.get_logger(__name__)

    parser = HfArgumentParser((TrainingArguments, TrainingConfig))
    hf_args, train_cfg = parser.parse_args_into_dataclasses()

    # set seed
    set_seed(hf_args.seed)

    # model setup
    kwargs = {}
    if train_cfg.use_flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        train_cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        **kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg.model_name,
        use_fast=True,
        trust_remote_code=True,
        **kwargs,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    # PEFT setup: prompt tuning or LoRA
    if train_cfg.use_prompt_tuning and train_cfg.use_lora:
        logger.warning(
            "Both use_lora and use_prompt_tuning are True; "
            "using prompt tuning and ignoring LoRA."
        )

    if train_cfg.use_lora:
        if "llama" in train_cfg.model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            target_modules = "all-linear"

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_cfg.lora_rank,
            lora_alpha=train_cfg.lora_alpha,
            lora_dropout=train_cfg.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)

    # if the train_dataset_name or train_dataset_config_name is provided, load from hub
    if train_cfg.train_dataset_name is not None:
        logger.info(f"Loading dataset from hub: {train_cfg.train_dataset_name}")
        train_dataset = load_dataset(
            train_cfg.train_dataset_name, train_cfg.train_dataset_config_name
        )["train"]
    else:
        # data files can be really big, but then we want to subselect
        logger.info(f"Loading dataset from local path: {train_cfg.train_dataset_path}")
        train_dataset = load_dataset("json", data_files=train_cfg.train_dataset_path)
        train_dataset = train_dataset["train"]

    # if num_samples given, then select the first num_samples
    if train_cfg.num_samples is not None:
        logger.info(
            f"num_samples is set to {train_cfg.num_samples}, so only the first {train_cfg.num_samples} samples will be used for training."
        )
        train_dataset = train_dataset.select(
            [i for i in list(range(train_cfg.num_samples))]
        )

    train_dataset = train_dataset.map(
        lambda x: encode_with_messages_format(
            x, tokenizer, 2048, only_first_two=False, add_bos_token=False
        )
    )
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # this filters out any samples where labels are all -100;
    # this happens because the examples are longer than the max length of 2048
    train_dataset = train_dataset.filter(lambda x: (x["labels"] != -100).any())
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        args=hf_args,
    )

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.train()
    trainer.save_model(hf_args.output_dir)
    trainer.accelerator.wait_for_everyone()
    trainer.save_state()


if __name__ == "__main__":
    train()
