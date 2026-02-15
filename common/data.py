import torch


# turns sample into something we can feed into a model
def construct_test_sample(
    tokenizer, sample, max_length=2048, prompt_only=False, response_only=False
):
    prompt, label = sample["prompts"], sample["labels"]
    # tokenizer label and prompt independently, then concatenate.
    # this is so we match the test format (tokenize the prompt and generate)
    if prompt_only:
        prompt = prompt + tokenizer.eos_token  # add eos token to prompt
        prompt = prompt.replace("\n<|assistant|>\n", "")  # remove assistant turn
    if response_only:
        label = tokenizer.bos_token + label  # add bos token to label
    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=max_length, truncation=True
    )
    labels = tokenizer(
        label + tokenizer.eos_token,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )
    if prompt_only:
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "labels": torch.ones_like(inputs["input_ids"][0]) * -100,
        }
    elif response_only:
        return {
            "input_ids": labels["input_ids"][0],
            "attention_mask": labels["attention_mask"][0],
            "labels": labels["input_ids"][0],
        }
    return {
        "input_ids": torch.cat([inputs["input_ids"][0], labels["input_ids"][0]], dim=0),
        "attention_mask": torch.cat(
            [inputs["attention_mask"][0], labels["attention_mask"][0]], dim=0
        ),
        "labels": torch.cat(
            [torch.ones_like(inputs["input_ids"][0]) * -100, labels["input_ids"][0]],
            dim=0,
        ),
    }


# needed for open-instruct: convert msg format.
def encode_with_messages_format(
    example,
    tokenizer,
    max_seq_length,
    include_response=True,
    response_only=False,
    only_first_two=False,
    prompt_only=False,
    add_bos_token=False,
):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n"
                    + message["content"].strip()
                    + tokenizer.eos_token
                    + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        # add bos token if needed
        if add_bos_token:
            message_text = tokenizer.bos_token + message_text
        return message_text

    # change: just take the first two prompts.
    if only_first_two:
        # if first role is system, we actually want to take the second and third message,
        # ignoring the first system message.
        if messages[0]["role"] == "system":
            messages = messages[1:3]
        else:
            messages = messages[:2]
    if prompt_only:
        if messages[0]["role"] == "system":
            messages = messages[:2]
        else:
            messages = messages[:1]
        msg = (
            _concat_messages(messages).strip() + tokenizer.eos_token
        )  # add eos token manually.
        res = tokenizer(
            msg, return_tensors="pt", max_length=max_seq_length, truncation=True
        )
        return {
            "string": msg,
            "input_ids": res.input_ids.flatten(),
            "attention_mask": res.attention_mask.flatten(),
            "labels": res.input_ids.flatten(),
        }
    elif response_only:
        idx = 1
        if messages[0]["role"] == "system":
            idx = 2
        msg = "<|assistant|>\n" + messages[idx]["content"].strip() + tokenizer.eos_token
        res = tokenizer(
            msg, return_tensors="pt", max_length=max_seq_length, truncation=True
        )
        return {
            "string": msg,
            "input_ids": res.input_ids.flatten(),
            "attention_mask": res.attention_mask.flatten(),
            "labels": res.input_ids.flatten(),
        }

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # here we also ignore the role of the assistant
                messages_so_far = (
                    _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
                )
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
        "string": messages_so_far,
    }
