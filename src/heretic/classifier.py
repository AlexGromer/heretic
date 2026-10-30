# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Fast non-answer classifier distilled from the judge (fork addition).
#
# The harness judge (Qwen3-14B) is too slow to score every trial. A small encoder
# (default ModernBERT-base, as in upstream PR #500) fine-tuned on answers the judge has
# labelled classifies 100 answers in seconds, so the optimization can minimize the share
# of REFUSAL + PARTIAL answers instead of keyword refusals (scorer ClassifierRate).
#
#   heretic-train-classifier --responses runs/R/responses.jsonl --judged runs/R/judged.jsonl \
#       [--responses ... --judged ...] --output classifiers/nonanswer
#
# Training data: the harness responses.jsonl joined with judged.jsonl, or files written by
# rows with prompt, response and judge_label / label. Validation holds out whole prompts (an answer
# of the same prompt from another trial must not be in both sets) and reports Cohen's kappa
# against the judge, 4-class and for answered vs not answered. Use the classifier only if the
# binary kappa is high (the evaluation harness gates its judge at 0.75).

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from .judge import DEFAULT_LABELS, DEFAULT_REFUSAL_LABELS, strip_cot

DEFAULT_BASE_MODEL = "answerdotai/ModernBERT-base"
METADATA_FILE = "heretic_classifier.json"
FORMAT_VERSION = 1


def encode_pairs(
    tokenizer: Any, prompts: Sequence[str], responses: Sequence[str], max_length: int
) -> Any:
    """Request and visible answer as a sentence pair; the longer part is truncated first."""
    return tokenizer(
        list(prompts),
        [strip_cot(response) for response in responses],
        truncation="longest_first",
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


@dataclass
class Classifier:
    model: Any
    tokenizer: Any
    labels: list[str]
    max_length: int
    device: torch.device

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "Classifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        metadata = json.loads(Path(path, METADATA_FILE).read_text(encoding="utf-8"))
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path} was trained for another input format")
        resolved = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        model = AutoModelForSequenceClassification.from_pretrained(path).to(resolved)
        model.eval()
        return cls(
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(path),
            labels=list(metadata["labels"]),
            max_length=int(metadata["max_length"]),
            device=resolved,
        )

    def predict(
        self, prompts: Sequence[str], responses: Sequence[str], batch_size: int = 16
    ) -> list[str]:
        predictions: list[str] = []
        for start in range(0, len(prompts), batch_size):
            batch = encode_pairs(
                self.tokenizer,
                prompts[start : start + batch_size],
                responses[start : start + batch_size],
                self.max_length,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**batch).logits
            predictions += [self.labels[int(i)] for i in logits.argmax(dim=-1).tolist()]
        return predictions


def cohen_kappa(truth: Sequence[str], predicted: Sequence[str]) -> float:
    """Cohen's kappa of two labelings (1 = perfect, 0 = chance)."""
    n = len(truth)
    if n == 0:
        return 0.0
    observed = sum(a == b for a, b in zip(truth, predicted)) / n
    truth_counts = Counter(truth)
    predicted_counts = Counter(predicted)
    expected = sum(
        truth_counts[label] * predicted_counts[label] for label in truth_counts
    ) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def evaluate(
    truth: Sequence[str], predicted: Sequence[str], refusal_labels: Sequence[str]
) -> dict[str, Any]:
    binary_truth = ["no" if t in refusal_labels else "yes" for t in truth]
    binary_predicted = ["no" if p in refusal_labels else "yes" for p in predicted]
    labels = sorted(set(truth))
    return {
        "n": len(truth),
        "accuracy": sum(a == b for a, b in zip(truth, predicted)) / max(len(truth), 1),
        "kappa": cohen_kappa(truth, predicted),
        "binary_kappa": cohen_kappa(binary_truth, binary_predicted),
        "recall": {
            label: sum(t == p == label for t, p in zip(truth, predicted))
            / max(sum(t == label for t in truth), 1)
            for label in labels
        },
        "judge_non_answer_rate": binary_truth.count("no") / max(len(truth), 1),
        "classifier_non_answer_rate": binary_predicted.count("no") / max(len(truth), 1),
    }


def split_by_prompt(
    items: Sequence[Any], prompt_of: Any, validation_fraction: float
) -> tuple[list[Any], list[Any]]:
    """Deterministic split that keeps all answers of a prompt on one side."""
    train, validation = [], []
    for item in items:
        digest = hashlib.sha256(prompt_of(item).encode("utf-8")).digest()
        (validation if digest[0] / 256 < validation_fraction else train).append(item)
    return train, validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="heretic-train-classifier",
        description="Fine-tune a small classifier on judge-labelled answers.",
    )
    parser.add_argument(
        "--responses",
        action="append",
        required=True,
        help="Answers (the harness responses.jsonl, or rows with a label); repeatable",
    )
    parser.add_argument(
        "--judged",
        action="append",
        help="harness judged.jsonl for the --responses at the same position",
    )
    parser.add_argument("--output", required=True, help="Directory for the classifier")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--refusal-labels", default=",".join(DEFAULT_REFUSAL_LABELS))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    from .positions import load_labelled_answers

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    refusal_labels = [
        label.strip() for label in args.refusal_labels.split(",") if label.strip()
    ]
    judged = args.judged or []
    if judged and len(judged) != len(args.responses):
        parser.error("give one --judged per --responses, or none")

    answers = []
    for index, responses in enumerate(args.responses):
        answers += load_labelled_answers(
            responses, judged[index] if judged else None, default_system=""
        )
    answers = [a for a in answers if a.label in labels]
    print(f"{len(answers)} labelled answers: {dict(Counter(a.label for a in answers))}")
    if len(answers) < 50:
        parser.error("too few labelled answers to train on")

    train, validation = split_by_prompt(
        answers, lambda a: a.prompt, args.validation_fraction
    )
    print(f"train {len(train)}, validation {len(validation)} (split by prompt)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        # A base model that already has a classification head with another number of
        # labels gets a new head.
        ignore_mismatched_sizes=True,
    ).to(device)

    # Inverse-frequency class weights: PARTIAL and REFUSAL can be rare in some runs.
    counts = Counter(a.label for a in train)
    weights = torch.tensor(
        [len(train) / (len(labels) * max(counts[label], 1)) for label in labels],
        device=device,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)

    label_index = {label: i for i, label in enumerate(labels)}
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = [train[i] for i in order[start : start + args.batch_size]]
            inputs = encode_pairs(
                tokenizer,
                [a.prompt for a in batch],
                [a.response for a in batch],
                args.max_length,
            ).to(device)
            targets = torch.tensor([label_index[a.label] for a in batch], device=device)
            loss = loss_function(model(**inputs).logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item() * len(batch)
        print(f"epoch {epoch + 1}: loss {total / len(train):.4f}")

    model.eval()
    classifier = Classifier(model, tokenizer, labels, args.max_length, device)
    metrics = evaluate(
        [a.label for a in validation],
        classifier.predict(
            [a.prompt for a in validation],
            [a.response for a in validation],
            args.batch_size,
        )
        if validation
        else [],
        refusal_labels,
    )
    print(json.dumps(metrics, indent=2))
    if metrics["binary_kappa"] < 0.75:
        print(
            "WARNING: binary kappa against the judge is below 0.75; "
            "do not use this classifier as an optimization objective yet."
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cast(Any, model).save_pretrained(output)
    cast(Any, tokenizer).save_pretrained(output)
    (output / METADATA_FILE).write_text(
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "labels": labels,
                "refusal_labels": refusal_labels,
                "max_length": args.max_length,
                "base_model": args.base_model,
                "validation": metrics,
                "training_files": args.responses,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
