"""Action-weighted supervised fine-tuning (Methods; Supplementary Table 7) on Hugging Face Transformers.

    python -m matact.train.weighted_sft --model Qwen/Qwen3-8B --train-file data/sft/train_messages.jsonl \
        --output-dir outputs/matact-qwen3-8b-awsft --bf16 --gradient-checkpointing

Token classes: tool names and special boundaries 2.5, argument keys/values 2.0, grounded
answers 2.0, user request 1.0, rationale 0.5; system prompt, environment observations,
padding and truncated suffixes are masked. The loss is normalized over non-masked tokens.
The conversation is rendered with the model's own chat template (tools passed as ``tools``).
"""
from __future__ import annotations

import argparse
import json
import re

WEIGHTS = {"tool_name": 2.5, "boundary": 2.5, "argument": 2.0, "answer": 2.0, "user": 1.0, "rationale": 0.5,
           "end_of_turn": 1.0}
_TAGS = ("<tool_call>", "</tool_call>", "<answer>", "</answer>")


def assistant_spans(content: str, w=WEIGHTS):
    """(start, end, weight) spans inside one assistant message; later spans override earlier ones."""
    spans = []
    first = min([i for i in (content.find("<tool_call>"), content.find("<answer>")) if i >= 0], default=len(content))
    if first > 0:
        spans.append((0, first, w["rationale"]))
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.S):
        spans.append((m.start(1), m.end(1), w["argument"]))
        name = re.search(r'"name"\s*:\s*"([^"]*)"', m.group(1))
        if name:
            spans.append((m.start(1) + name.start(1), m.start(1) + name.end(1), w["tool_name"]))
    for m in re.finditer(r"<answer>(.*?)</answer>", content, re.S):
        spans.append((m.start(1), m.end(1), w["answer"]))
    for tag in _TAGS:
        for m in re.finditer(re.escape(tag), content):
            spans.append((m.start(), m.end(), w["boundary"]))
    return spans


def char_weights(rendered: str, messages, end_of_turn="<|im_end|>", uniform=False, w=WEIGHTS):
    """Per-character weights of a rendered conversation (0 = masked)."""
    weights = [0.0] * len(rendered)
    cursor = 0
    for msg in messages:
        content = msg["content"].strip()
        start = rendered.find(content, cursor)
        if start < 0:
            raise ValueError(f"message content not found in the rendered template: {content[:60]!r}")
        end = start + len(content)
        cursor = end
        if msg["role"] == "user" and not uniform:
            weights[start:end] = [w["user"]] * (end - start)
        elif msg["role"] == "assistant":
            if uniform:
                weights[start:end] = [1.0] * (end - start)
            else:
                for s, e, value in assistant_spans(content, w):
                    weights[start + s:start + e] = [value] * (e - s)
            tail = rendered.find(end_of_turn, end)
            if end_of_turn and tail >= 0 and not rendered[end:tail].strip():
                weights[tail:tail + len(end_of_turn)] = [w["end_of_turn"]] * len(end_of_turn)
    return weights


def token_weights(weights, offsets):
    """Weight of each token = largest character weight in its span."""
    return [max(weights[s:e]) if e > s else 0.0 for s, e in offsets]


def render(tokenizer, record) -> str:
    """Render one conversation with the model's chat template.

    Qwen3 templates insert an empty ``<think>`` block before the last assistant turn only. It is
    removed so that every turn has the same layout as at inference with the default template.
    """
    rendered = tokenizer.apply_chat_template(record["messages"], tools=record.get("tools"), tokenize=False)
    return rendered.replace("<think>\n\n</think>\n\n", "")


def encode(tokenizer, record, max_length, uniform=False):
    rendered = render(tokenizer, record)
    weights = char_weights(rendered, record["messages"], uniform=uniform)
    enc = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False, truncation=True,
                    max_length=max_length)
    tw = token_weights(weights, enc["offset_mapping"])
    labels = [t if wt > 0 else -100 for t, wt in zip(enc["input_ids"], tw)]
    return {"input_ids": enc["input_ids"], "labels": labels, "loss_weights": tw}


def read_rows(path, tokenizer, max_length, uniform):
    with open(path, encoding="utf-8") as fh:
        return [encode(tokenizer, json.loads(line), max_length, uniform) for line in fh if line.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--valid-file", default=None, help="optional evaluation set (loss reported every epoch)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4900)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--deepspeed", default=None)
    p.add_argument("--uniform", action="store_true", help="uniform weights on assistant tokens (standard SFT baseline)")
    args = p.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    # Created before the model so that DeepSpeed ZeRO-3 can partition the weights while loading.
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size, per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, adam_beta1=0.9, adam_beta2=0.95, lr_scheduler_type="cosine",
        bf16=args.bf16, gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        deepspeed=args.deepspeed, logging_steps=10, save_strategy="epoch",
        eval_strategy="epoch" if args.valid_file else "no", prediction_loss_only=True,
        seed=args.seed, report_to=[], remove_unused_columns=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train_rows = read_rows(args.train_file, tokenizer, args.max_length, args.uniform)
    valid_rows = read_rows(args.valid_file, tokenizer, args.max_length, args.uniform) if args.valid_file else None

    class Rows(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, value: seq + [value] * (n - len(seq))  # noqa: E731
        return {"input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
                "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
                "loss_weights": torch.tensor([pad(b["loss_weights"], 0.0) for b in batch])}

    class WeightedTrainer(Trainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            # The loss below is already normalized per micro-batch, so the Trainer must average it
            # over gradient-accumulation steps instead of expecting a token-count-normalized sum.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights, labels = inputs.pop("loss_weights"), inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :].float()
            target, w = labels[:, 1:], weights[:, 1:].to(logits.dtype)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100,
                                 reduction="none").view_as(target)
            mask = target.ne(-100)
            loss = (ce * w * mask).sum() / mask.sum().clamp_min(1)
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if args.bf16 else None)
    model.config.use_cache = False
    trainer = WeightedTrainer(model=model, args=training_args, train_dataset=Rows(train_rows),
                              eval_dataset=Rows(valid_rows) if valid_rows else None, data_collator=collate)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
