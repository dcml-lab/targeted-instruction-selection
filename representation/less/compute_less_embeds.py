import argparse
import json
import os
from hashlib import md5
from typing import Any, Iterable, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from functorch import grad, make_functional_with_buffers, vmap
from peft import LoraConfig, PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.data import construct_test_sample, encode_with_messages_format


def load_train_dataset(
    train_dataset_path: str,
    tokenizer: AutoTokenizer,
    start_index: int = 0,
    end_index: int = None,
    debug: bool = False,
) -> (torch.utils.data.Dataset, int, int):
    if train_dataset_path is not None and os.path.exists(train_dataset_path):
        # assuming it is json
        train_dataset = load_dataset("json", data_files=[train_dataset_path])["train"]
    else:
        train_dataset = load_dataset(
            "Harvard-DCML/tulu-v2-197K-processed", split="train"
        )

    if end_index is not None:
        end_index = min(end_index, len(train_dataset))
        train_dataset = train_dataset.select(range(start_index, end_index))
        print(f"Selected training dataset from index {start_index} to {end_index}")

    if debug:
        train_dataset = train_dataset.select(range(100))

    train_dataset = train_dataset.map(
        lambda x: encode_with_messages_format(
            example=x, tokenizer=tokenizer, max_seq_length=2048, include_response=True
        ),
        num_proc=16,
    )
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    print("Number of training examples:", len(train_dataset))

    return train_dataset, start_index, end_index


def prepare_batch(batch, device=torch.device("cuda:0")):
    """Move the batch to the device."""
    for key in batch:
        batch[key] = batch[key].to(device)


def get_trak_projector(device: torch.device):
    """Get trak projectors (see https://github.com/MadryLab/trak for details)"""
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(
            torch.zeros(8, 1_000, device=device), 512, 0, num_sms
        )
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def get_number_of_trainable_params(model):
    """Make sure that only lora parameters require gradients in peft models."""
    if isinstance(model, PeftModel):
        names = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and "lora" not in n
        ]
        assert len(names) == 0
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params


def obtain_gradients(model, batch):
    """obtain gradients."""
    loss = model(**batch).loss
    loss.backward()
    vectorized_grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    )
    return vectorized_grads


def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    """obtain gradients with adam optimizer states."""
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    loss = model(**batch).loss
    loss.backward()

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None]
    )

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads**2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads


def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]

    # hack
    avg = torch.cat([optimizer_state[i]["exp_avg"].view(-1) for i in range(len(names))])
    avg_sq = torch.cat(
        [optimizer_state[i]["exp_avg_sq"].view(-1) for i in range(len(names))]
    )

    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


# from the less paper
def collect_grads(
    dataloader,
    model,
    proj_dim: int = 8192,
    adam_optimizer_state: Optional[dict] = None,
    gradient_type: str = "adam",
    project_interval: int = 8,
):
    """
    Collects gradients from the model during evaluation and saves them to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        proj_dim (int): The dimensions of the target projectors.
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states.
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
        project_interval (int): The interval for projection. For example, if project_interval=8, the gradients will be projected every 8 batches.
    """
    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 16  # batch size for the projectors
    torch.random.manual_seed(0)  # set the random seed for torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector(device)
    number_of_params = get_number_of_trainable_params(model)

    # initialize a project for each target projector dimension
    proj = projector(
        grad_dim=number_of_params,
        proj_dim=proj_dim,
        seed=0,
        proj_type=ProjectionType.rademacher,
        device=device,
        dtype=dtype,
        block_size=block_size,
        max_batch_size=projector_batch_size,
    )

    def _project(current_full_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)
        current_projected_grads = proj.project(current_full_grads, model_id=model_id)
        return current_projected_grads.cpu()

    # projected_gradients
    full_grads = []  # full gradients
    projected_grads = []
    model.train()
    count = 0
    for batch in tqdm(dataloader, total=len(dataloader)):
        count += 1
        prepare_batch(batch, device=device)

        if gradient_type == "adam":
            vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
        elif gradient_type == "sign":
            vectorized_grads = obtain_sign_gradients(model, batch)
        else:
            vectorized_grads = obtain_gradients(model, batch)

        # add the gradients to the full_grads
        full_grads.append(vectorized_grads)
        model.zero_grad()

        # project
        if count % project_interval == 0:
            projected_grads.append(_project(full_grads))
            full_grads = []

    if len(full_grads) > 0:
        projected_grads.append(_project(full_grads))
        full_grads = []

    return torch.cat(projected_grads, dim=0)


def normalize_embeddings_in_chunks(
    x, chunk_size=8192, dim=1, eps=1e-12, in_place=False
):
    out = x if in_place else torch.empty_like(x)
    for s in range(0, x.shape[0], chunk_size):
        e = min(s + chunk_size, x.shape[0])
        out[s:e] = F.normalize(x[s:e], p=2, dim=dim, eps=eps)
    return out


def compute_eval_grads():
    eval_dataset = load_dataset(
        "Harvard-DCML/targeted-query-set-processed", eval_dataset_name, split=split
    )
    eval_dataset = eval_dataset.map(
        lambda x: construct_test_sample(
            sample=x,
            tokenizer=tokenizer,
            max_length=2048,
        )
    )
    eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    eval_grads = None

    return eval_grads


def load_model(
    model_name_or_path: str, tokenizer: AutoTokenizer, torch_dtype: Any = torch.bfloat16
) -> Any:
    is_peft = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto"
        )

        embedding_size = base_model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) != embedding_size:
            print(f"Resizing embeddings: {embedding_size} -> {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(
            base_model, model_name_or_path, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="auto"
        )

        # resize embeddings if needed (e.g. for LlamaTokenizer)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) != embedding_size:
            model.resize_token_embeddings(len(tokenizer))

    for name, param in model.named_parameters():
        if "lora" in name or "Lora" in name:
            param.requires_grad = True
    return model


def get_base_model_name(model_name_or_path: str) -> str:
    is_peft = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    if is_peft:
        config = LoraConfig.from_pretrained(model_name_or_path)
        return config.base_model_name_or_path
    else:
        return model_name_or_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="The output directory where gradients are saved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_dataset_path", type=str, default=None)
    parser.add_argument("--train_index_path", type=str, default=None)

    parser.add_argument("--dev_dataset_name", type=str, default=None)
    parser.add_argument("--dev_index_path", type=str, default=None)

    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--gradient_type", type=str, default="adam")
    parser.add_argument("--proj_dim", type=int, default=8192)
    parser.add_argument("--ckpt_path", type=str, default=None, required=True)
    parser.add_argument("--ckpt_step", type=int, default=None, required=True)

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--compute_train_grads", action="store_true")
    parser.add_argument("--compute_dev_grads", action="store_true")
    parser.add_argument("--compute_test_grads", action="store_true")
    parser.add_argument("--save_original", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.end_index is not None and args.end_index <= args.start_index:
        raise ValueError("end_index must be greater than start_index")

    os.makedirs(args.save_dir, exist_ok=True)
    # load model
    print(f"Loading model from checkpoint: {args.ckpt_path} at step {args.ckpt_step}")
    print("Note: this is a LoRA checkpoint.")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, use_fast=True)
    model = load_model(args.ckpt_path, tokenizer)

    # get the base model_name
    base_model_name = get_base_model_name(args.ckpt_path)

    # load optimizer state
    adam_optimizer_state = None
    if args.gradient_type == "adam":
        optimizer_path = os.path.join(args.ckpt_path, "optimizer.pt")
        adam_optimizer_state = torch.load(optimizer_path, map_location="cpu")["state"]

    if args.compute_train_grads:
        train_dataset, start_index, end_index = load_train_dataset(
            args.train_dataset_path,
            tokenizer=tokenizer,
            start_index=args.start_index,
            end_index=args.end_index,
            debug=args.debug,
        )
        args.start_index = start_index
        args.end_index = end_index

        if args.start_index == 0 and args.end_index is None:
            if not args.train_index_path:
                args.train_index_path = os.path.join(
                    args.save_dir,
                    f"train_grads_{args.gradient_type}_ckpt{args.ckpt_step}_dim{args.proj_dim}.pt",
                )

        else:
            if not args.train_index_path:
                args.train_index_path = os.path.join(
                    args.save_dir,
                    f"train_grads_{args.gradient_type}_ckpt{args.ckpt_step}_dim{args.proj_dim}_{args.start_index}_{args.end_index}.pt",
                )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False
        )

        train_grads = collect_grads(
            train_dataloader,
            model,
            proj_dim=args.proj_dim,
            adam_optimizer_state=adam_optimizer_state,
            gradient_type=args.gradient_type,
        )

        if args.save_original:
            torch.save(train_grads, args.train_index_path)
            print(f"Saved train grads to: {args.train_index_path}")

        # normalize and save
        norm_train_grads = normalize_embeddings_in_chunks(
            train_grads, chunk_size=10000, dim=1, eps=1e-12, in_place=False
        )
        norm_train_grads_path = args.train_index_path.replace(".pt", "_normalized.pt")
        torch.save(norm_train_grads, norm_train_grads_path)
        print(f"Saved normalized train grads to: {norm_train_grads_path}")

    if args.compute_dev_grads:
        # print all related info
        print(f"Computing dev grads for dataset: {args.dev_dataset_name}")
        print(f"checkpoint: {args.ckpt_path} at step {args.ckpt_step}")
        print(f"gradient type: {args.gradient_type}")
        print(f"projection dimension: {args.proj_dim}")

        if not args.dev_index_path:
            args.dev_index_path = os.path.join(
                args.save_dir,
                f"{args.dev_dataset_name}_grads_{args.gradient_type}_ckpt{args.ckpt_step}_dim{args.proj_dim}.pt",
            )

        dev_dataset = load_dataset(
            "Harvard-DCML/targeted-query-set-processed",
            args.dev_dataset_name,
            split="dev",
        )

        dev_dataset = dev_dataset.map(
            lambda x: construct_test_sample(
                sample=x,
                tokenizer=tokenizer,
                max_length=2048,
            )
        )
        dev_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        dev_dataloader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=1, shuffle=False
        )

        dev_grads = collect_grads(
            dev_dataloader,
            model,
            proj_dim=args.proj_dim,
            adam_optimizer_state=adam_optimizer_state,
            gradient_type=args.gradient_type,
        )
        # save dev grads
        if args.save_original:
            torch.save(dev_grads, args.dev_index_path)
            print(f"Saved dev grads to: {args.dev_index_path}")

        # normalize and save
        norm_dev_grads = normalize_embeddings_in_chunks(
            dev_grads, chunk_size=10000, dim=1, eps=1e-12, in_place=False
        )
        norm_dev_grads_path = args.dev_index_path.replace(".pt", "_normalized.pt")
        torch.save(norm_dev_grads, norm_dev_grads_path)
        print(f"Saved normalized dev grads to: {norm_dev_grads_path}")
