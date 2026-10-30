# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Preference tuning against non-answers (fork addition).
#
# Abliteration removes the refusal ("no"), but on our models a large share of the answers
# then becomes PARTIAL (an overview instead of specifics, half the steps): a learned style,
# not a single direction. Preference tuning changes styles. The pairs come from the
# abliterated model itself: several sampled answers per prompt, labelled by the refusal-eval
# judge; chosen = a COMPLY answer, rejected = a PARTIAL or REFUSAL answer to the same prompt.
# The only verified precedent is mlabonne's NeuralDaredevil (DPO after abliteration,
# QLoRA r64 / alpha 32, lr 5e-6, 1 epoch), which healed quality, not refusals; that DPO
# reduces PARTIAL is a hypothesis to measure (judge, KL, GSM8K).
#
#   heretic-dpo-pairs generate --model merged/ --prompts offsec_train.txt --out samples.jsonl
#   heretic-dpo-pairs judge --samples samples.jsonl --out judged.jsonl [--endpoint ...]
#   heretic-dpo-pairs pairs --judged judged.jsonl --out pairs.jsonl
#   heretic-dpo-pairs train --model merged/ --pairs pairs.jsonl --output dpo-adapter/ [--merge merged-dpo/]
#
# Use training prompts that are not in the evaluation hold-out. `train` needs TRL
# (uv pip install trl), which is not a dependency of Heretic.

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from .judge import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_REFUSAL_LABELS,
    JudgeClient,
    JudgeError,
    load_rubric,
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(
    path: str, rows: Sequence[dict[str, Any]], append: bool = False
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as file:
        file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def read_prompts(path: str) -> list[str]:
    """A text file with one prompt per line, or JSON Lines with a "prompt" (or "text") field."""
    if path.endswith(".jsonl"):
        return [
            str(row.get("prompt", row.get("text", "")))
            for row in read_jsonl(path)
            if row.get("prompt", row.get("text"))
        ]
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_pairs(
    rows: Sequence[dict[str, Any]],
    chosen_labels: Sequence[str] = ("COMPLY",),
    rejected_labels: Sequence[str] = DEFAULT_REFUSAL_LABELS,
    max_pairs_per_prompt: int = 2,
    drop_truncated_chosen: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    DPO pairs in TRL's conversational format from judged samples: for each prompt with at
    least one chosen and one rejected answer, up to `max_pairs_per_prompt` pairs (each
    answer used at most once). Chosen answers that hit the length limit can be dropped,
    since a cut-off answer would teach stopping halfway.
    """
    rng = random.Random(seed)
    by_prompt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[(row.get("system", ""), row["prompt"])].append(row)

    pairs = []
    for (system, prompt), samples in sorted(by_prompt.items()):
        chosen = [
            s
            for s in samples
            if s.get("judge_label") in chosen_labels
            and not (drop_truncated_chosen and s.get("finish") == "length")
        ]
        rejected = [s for s in samples if s.get("judge_label") in rejected_labels]
        rng.shuffle(chosen)
        rng.shuffle(rejected)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        for good, bad in list(zip(chosen, rejected))[:max_pairs_per_prompt]:
            pairs.append(
                {
                    "prompt": messages,
                    "chosen": [{"role": "assistant", "content": good["response"]}],
                    "rejected": [{"role": "assistant", "content": bad["response"]}],
                    "rejected_label": bad["judge_label"],
                }
            )
    return pairs


def command_generate(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = read_prompts(args.prompts)
    tokenizer: Any = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map=args.device_map
    )
    model.eval()
    torch.manual_seed(args.seed)

    jobs = [(i, s) for i in range(len(prompts)) for s in range(args.samples)]
    done = set()
    if Path(args.out).exists():
        done = {(r["prompt_id"], r["sample"]) for r in read_jsonl(args.out)}
    jobs = [job for job in jobs if job not in done]
    print(f"{len(prompts)} prompts x {args.samples} samples, {len(jobs)} to generate")

    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": prompts[i]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.thinking,
            )
            for i, _ in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        rows = []
        for (i, s), tokens in zip(batch, new_tokens):
            stopped = bool((tokens == tokenizer.eos_token_id).any()) or bool(
                (tokens == tokenizer.pad_token_id).any()
            )
            rows.append(
                {
                    "prompt_id": i,
                    "sample": s,
                    "system": args.system_prompt,
                    "prompt": prompts[i],
                    "response": tokenizer.decode(tokens, skip_special_tokens=True),
                    "finish": "stop" if stopped else "length",
                }
            )
        write_jsonl(args.out, rows, append=True)
        print(f"{min(start + args.batch_size, len(jobs))}/{len(jobs)}")
    return 0


def command_judge(args: argparse.Namespace) -> int:
    client = JudgeClient(
        endpoint=args.endpoint, model=args.model, rubric=load_rubric(args.rubric)
    )
    rows = read_jsonl(args.samples)
    done = set()
    if Path(args.out).exists():
        done = {(r["prompt_id"], r["sample"]) for r in read_jsonl(args.out)}
    failed = 0
    for index, row in enumerate(rows, start=1):
        if (row["prompt_id"], row["sample"]) in done:
            continue
        try:
            verdict = client.judge(row["prompt"], row["response"])
        except JudgeError as error:
            failed += 1
            print(f"{row['prompt_id']}/{row['sample']}: {error}")
            continue
        write_jsonl(
            args.out,
            [
                {
                    **row,
                    "judge_label": verdict.label,
                    "judge_reason": verdict.reason,
                    "judge_prompt": client.prompt_version,
                }
            ],
            append=True,
        )
        if index % 25 == 0:
            print(f"{index}/{len(rows)}")
    print(f"done, {failed} failed (rerun to retry them)")
    return 0 if failed == 0 else 2


def join_refusal_eval(responses_path: str, judged_path: str) -> list[dict[str, Any]]:
    """
    refusal-eval responses.jsonl rows with the label of judged.jsonl (joined by row key),
    in the shape build_pairs reads (prompt, response, judge_label, finish, system).
    """
    from .positions import _read_jsonl, _refusal_eval_key

    labels: dict[str, str] = {}
    for row in _read_jsonl(judged_path):
        if isinstance(row.get("label"), str):
            labels.setdefault(_refusal_eval_key(row), row["label"])
    rows = []
    for row in _read_jsonl(responses_path):
        label = labels.get(_refusal_eval_key(row))
        if label is None:
            continue
        settings = row.get("settings")
        system = settings.get("system_prompt", "") if isinstance(settings, dict) else ""
        rows.append(
            {
                "prompt": row["prompt"],
                "response": row["response"],
                "judge_label": label,
                "finish": row.get("finish"),
                "system": row.get("system", system),
            }
        )
    return rows


def command_pairs(args: argparse.Namespace) -> int:
    rows = (
        join_refusal_eval(args.responses, args.judged)
        if args.responses
        else read_jsonl(args.judged)
    )
    pairs = build_pairs(
        rows,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        drop_truncated_chosen=not args.keep_truncated_chosen,
        seed=args.seed,
    )
    write_jsonl(args.out, pairs)
    labels: dict[str, int] = defaultdict(int)
    for row in rows:
        labels[row.get("judge_label", "?")] += 1
    print(f"samples by label: {dict(labels)}")
    print(
        f"{len(pairs)} pairs from {len({p['prompt'][-1]['content'] for p in pairs})} prompts"
    )
    return 0 if pairs else 1


def command_train(args: argparse.Namespace) -> int:
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        print("train needs TRL: uv pip install trl")
        return 1
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs = [
        {k: row[k] for k in ("prompt", "chosen", "rejected")}
        for row in read_jsonl(args.pairs)
    ]
    tokenizer: Any = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map=args.device_map
    )

    config = DPOConfig(
        output_dir=args.output,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        use_cpu=args.device_map == "cpu",
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(pairs),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    trainer.save_model(args.output)
    print(f"Adapter saved to {args.output}")

    if args.merge:
        merged = cast(Any, trainer.model).merge_and_unload()
        merged.save_pretrained(args.merge)
        tokenizer.save_pretrained(args.merge)
        print(f"Merged model saved to {args.merge}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="heretic-dpo-pairs",
        description="Self-generated DPO pairs (COMPLY vs PARTIAL/REFUSAL) and DPO training.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="sample answers from a model")
    generate.add_argument(
        "--model", required=True, help="merged model directory or HF ID"
    )
    generate.add_argument("--prompts", required=True, help="text file or JSON Lines")
    generate.add_argument("--out", required=True)
    generate.add_argument("--samples", type=int, default=4)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--max-new-tokens", type=int, default=512)
    generate.add_argument("--batch-size", type=int, default=8)
    generate.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    generate.add_argument(
        "--thinking", action="store_true", help="enable_thinking in the chat template"
    )
    generate.add_argument("--device-map", default="auto")
    generate.add_argument("--seed", type=int, default=1234)

    judge = sub.add_parser(
        "judge", help="label the samples with the refusal-eval judge"
    )
    judge.add_argument("--samples", required=True)
    judge.add_argument("--out", required=True)
    judge.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    judge.add_argument("--model", default=DEFAULT_MODEL)
    judge.add_argument("--rubric", help='e.g. "../refusal-eval/judge.py:SYSTEM_PROMPT"')

    pairs = sub.add_parser("pairs", help="build DPO pairs from judged samples")
    pairs.add_argument(
        "--judged",
        required=True,
        help="heretic-dpo-pairs judge output, or refusal-eval judged.jsonl with --responses",
    )
    pairs.add_argument(
        "--responses",
        help="refusal-eval responses.jsonl (e.g. generate_colab.py --sets bad_train --samples 4)",
    )
    pairs.add_argument("--out", required=True)
    pairs.add_argument("--max-pairs-per-prompt", type=int, default=2)
    pairs.add_argument("--keep-truncated-chosen", action="store_true")
    pairs.add_argument("--seed", type=int, default=0)

    train = sub.add_parser("train", help="DPO LoRA training (needs TRL)")
    train.add_argument("--model", required=True)
    train.add_argument("--pairs", required=True)
    train.add_argument("--output", required=True, help="adapter directory")
    train.add_argument("--merge", help="also save the merged model here")
    train.add_argument("--beta", type=float, default=0.1)
    train.add_argument("--learning-rate", type=float, default=5e-6)
    train.add_argument("--epochs", type=float, default=1.0)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--max-length", type=int, default=1536)
    train.add_argument("--lora-rank", type=int, default=64)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--device-map", default="auto")

    args = parser.parse_args(argv)
    return {
        "generate": command_generate,
        "judge": command_judge,
        "pairs": command_pairs,
        "train": command_train,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
