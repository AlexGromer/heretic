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
# Sampling and judging are refusal-eval's job (one generator and one judge for everything):
#
#   refusal-eval: generate_colab.py --sets bad_train --modes sample --samples 4 --out R.jsonl
#                 judge_colab.py batch --responses R.jsonl --out J.jsonl   (or judge.py batch)
#   heretic-dpo-pairs pairs --responses R.jsonl --judged J.jsonl --out pairs.jsonl
#   heretic-dpo-pairs train --model merged/ --pairs pairs.jsonl --output dpo-adapter/ [--merge merged-dpo/]
#
# Use training prompts (refusal-eval train/), not the evaluation hold-out. `train` needs TRL
# (uv pip install trl), which is not a dependency of Heretic.

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from .judge import DEFAULT_REFUSAL_LABELS


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(
    path: str, rows: Sequence[dict[str, Any]], append: bool = False
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as file:
        file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


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
    rows = join_refusal_eval(args.responses, args.judged)
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
        description=(
            "DPO pairs (COMPLY vs PARTIAL/REFUSAL) from answers generated and judged by "
            "refusal-eval, and DPO training."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pairs = sub.add_parser("pairs", help="build DPO pairs from judged samples")
    pairs.add_argument(
        "--responses",
        required=True,
        help="refusal-eval responses.jsonl (e.g. generate_colab.py --sets bad_train --samples 4)",
    )
    pairs.add_argument(
        "--judged", required=True, help="refusal-eval judged.jsonl of those responses"
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
        "pairs": command_pairs,
        "train": command_train,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
