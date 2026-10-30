# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Activations at chosen token positions (fork addition).
#
# Heretic computes its directions and ARA's module I/O at the end of the prompt, where the
# model decides whether to refuse. Two failure modes are not decided there:
#
# - PARTIAL answers (an overview instead of specifics, half the steps, an educational pivot)
#   start compliantly and turn evasive while the answer is written. Their direction is taken
#   from the tokens of judged answers: mean over the first answer tokens of PARTIAL answers
#   minus the same for COMPLY answers ("response directions").
# - Zhao et al. 2025 ("LLMs Encode Harmfulness and Refusal Separately", arXiv 2507.11878)
#   find that harmfulness is encoded at the last token of the instruction, before the
#   post-instruction template tokens, while refusal is encoded after them. Removing refusal
#   alone leaves the model aware of the harm, which can show as hedging. The harmfulness
#   direction is taken at the last instruction token ("instruction positions").
#
# Answers come from a JSON Lines file with prompt, response and label, as written by
# the evaluation harness (responses.jsonl joined with judged.jsonl), or rows that carry a judge_label / label.

import json
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Module

from .judge import COT_BLOCKS, strip_cot
from .model import (
    Model,
    ModuleIO,
    _collapse_flattened_residual_streams,
    strftime_fixed_date,
)
from .utils import Prompt, batchify, print


@dataclass
class LabelledAnswer:
    system: str
    prompt: str
    response: str
    label: str
    # One-line reason the judge gave for the label; empty when the file has none.
    reason: str = ""


def _response_key(row: dict[str, Any]) -> str:
    # harness judge.row_key.
    if row.get("key"):
        return str(row["key"])
    return "|".join(
        str(row.get(k, ""))
        for k in ("model", "trial", "mode", "thinking", "set", "prompt_id")
    )


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn last line of an interrupted run.
                continue
    return rows


def load_labelled_answers(
    responses_path: str,
    judged_path: str | None,
    default_system: str,
) -> list[LabelledAnswer]:
    """
    Answers with labels from `responses_path`. Without `judged_path`, every row needs its
    label in `judge_label` or `label`. With `judged_path` (the harness
    judged.jsonl, which has no prompts or responses), labels are joined by the row key.
    """
    rows = _read_jsonl(responses_path)
    labels: dict[str, str] = {}
    reasons: dict[str, str] = {}
    if judged_path is not None:
        for row in _read_jsonl(judged_path):
            label = row.get("label")
            if isinstance(label, str):
                key = _response_key(row)
                labels.setdefault(key, label)
                if isinstance(row.get("reason"), str):
                    reasons.setdefault(key, row["reason"])

    answers = []
    for row in rows:
        if judged_path is not None:
            key = _response_key(row)
            label = labels.get(key)
            if key in reasons:
                row = {**row, "judge_reason": reasons[key]}
        else:
            label = row.get("judge_label", row.get("label"))
        if not isinstance(label, str) or "prompt" not in row or "response" not in row:
            continue
        system = row.get("system")
        if not isinstance(system, str):
            settings = row.get("settings")
            system = (
                settings.get("system_prompt")
                if isinstance(settings, dict)
                else default_system
            )
        reason = row.get("judge_reason", row.get("reason", ""))
        answers.append(
            LabelledAnswer(
                system=system if isinstance(system, str) else default_system,
                prompt=str(row["prompt"]),
                response=str(row["response"]),
                label=label,
                reason=str(reason) if isinstance(reason, str) else "",
            )
        )
    return answers


def split_answers(
    answers: Sequence[LabelledAnswer],
    bad_labels: Sequence[str],
    good_labels: Sequence[str],
    max_per_class: int,
    seed: int = 0,
) -> tuple[list[LabelledAnswer], list[LabelledAnswer]]:
    """The bad and good answers (with a non-empty visible answer), at most `max_per_class` each."""
    usable = [a for a in answers if strip_cot(a.response)]
    rng = random.Random(seed)

    def pick(labels: Sequence[str]) -> list[LabelledAnswer]:
        selected = [a for a in usable if a.label in labels]
        if len(selected) > max_per_class:
            selected = rng.sample(selected, max_per_class)
        return selected

    return pick(bad_labels), pick(good_labels)


def chat_text(model: Model, system: str, user: str) -> str:
    """The text Model.generate feeds the model for this prompt (including the response prefix)."""
    chat = model._apply_system_role(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    text = cast(
        str,
        model.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            strftime_now=strftime_fixed_date,
        ),
    )
    return text + (model.settings.response_prefix or "")


@dataclass
class PositionedText:
    input_ids: list[int]
    # Token indices whose activations are averaged.
    positions: list[int]


def _encode(model: Model, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoding = model.tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return list(encoding["input_ids"]), [tuple(o) for o in encoding["offset_mapping"]]


def answer_positions(
    model: Model, answer: LabelledAnswer, count: int
) -> PositionedText:
    """The first `count` tokens of the visible answer, after the prompt as generate builds it."""
    prompt_text = chat_text(model, answer.system, answer.prompt)
    text = prompt_text + strip_cot(answer.response)
    input_ids, offsets = _encode(model, text)
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if start >= len(prompt_text) and end > start
    ][:count]
    return PositionedText(input_ids, positions or [len(input_ids) - 1])


def thinking_positions(
    model: Model, answer: LabelledAnswer, count: int
) -> PositionedText | None:
    """
    The first `count` tokens inside the answer's reasoning block, or None when the answer
    has none. Qwen3-style models decide there whether to answer (upstream issue #360), so a
    direction taken here targets that decision rather than the written answer.
    """
    prompt_text = chat_text(model, answer.system, answer.prompt)
    visible = strip_cot(answer.response)
    thinking = answer.response
    if visible:
        cut = thinking.rfind(visible)
        if cut >= 0:
            thinking = thinking[:cut]
    # Strip the delimiters themselves, so only the reasoning text is used.
    for initializer, finalizer in COT_BLOCKS:
        thinking = thinking.replace(initializer, " ").replace(finalizer, " ")
    if not thinking.strip():
        return None
    text = prompt_text + thinking
    input_ids, offsets = _encode(model, text)
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if start >= len(prompt_text) and end > start
    ][:count]
    return PositionedText(input_ids, positions) if positions else None


def instruction_end_position(model: Model, prompt: Prompt) -> PositionedText:
    """The last token of the user instruction, before the post-instruction template tokens."""
    text = chat_text(model, prompt.system, prompt.user)
    input_ids, offsets = _encode(model, text)
    end = text.rfind(prompt.user)
    if end < 0 or not prompt.user:
        return PositionedText(input_ids, [len(input_ids) - 1])
    end += len(prompt.user)
    covering = [
        index for index, (start, stop) in enumerate(offsets) if start < end <= stop
    ] or [index for index, (start, stop) in enumerate(offsets) if stop <= end]
    return PositionedText(input_ids, [covering[-1] if covering else len(input_ids) - 1])


def _pad(
    model: Model, batch: Sequence[PositionedText]
) -> tuple[Tensor, Tensor, list[list[int]]]:
    # Left padding, like Model.generate; positions shift by the padding.
    length = max(len(item.input_ids) for item in batch)
    pad_id = model.tokenizer.pad_token_id
    input_ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), length), dtype=torch.long)
    positions = []
    for row, item in enumerate(batch):
        offset = length - len(item.input_ids)
        input_ids[row, offset:] = torch.tensor(item.input_ids, dtype=torch.long)
        attention_mask[row, offset:] = 1
        positions.append([offset + p for p in item.positions])
    return input_ids, attention_mask, positions


def _forward(
    model: Model, input_ids: Tensor, attention_mask: Tensor, **kwargs: Any
) -> Any:
    device = model.model.device
    with torch.no_grad():
        return cast(Callable[..., Any], model.model)(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
            **kwargs,
        )


def forced_logits(
    model: Model, items: Sequence[PositionedText]
) -> Iterator[tuple[int, Tensor]]:
    """
    Teacher-forces the items and yields (item index, logits at its positions) one item at a
    time, so a consumer can reduce them before the next batch allocates its own logits.
    The logits are float32 on the CPU, of shape (positions, vocabulary).
    """
    index = 0
    for batch in batchify(list(items), model.settings.batch_size):
        input_ids, attention_mask, positions = _pad(model, batch)
        outputs = _forward(model, input_ids, attention_mask)
        logits = outputs.logits
        for row, row_positions in enumerate(positions):
            selected = torch.tensor(row_positions, device=logits.device)
            yield index, logits[row].index_select(0, selected).to(torch.float32).cpu()
            index += 1
        del outputs, logits


def hidden_state_means(model: Model, items: Sequence[PositionedText]) -> Tensor:
    """
    For every item, the mean residual (hidden state) over its positions, for every residual
    row (embeddings and layers): (item, row, hidden), float32 on the CPU.
    """
    results = []
    for batch in batchify(list(items), model.settings.batch_size):
        input_ids, attention_mask, positions = _pad(model, batch)
        outputs = _forward(model, input_ids, attention_mask, output_hidden_states=True)
        hidden_states = tuple(
            h.mean(dim=2) if h.dim() == 4 else h for h in outputs.hidden_states
        )
        hidden_states = _collapse_flattened_residual_streams(
            cast(Any, hidden_states), model.model.config
        )
        for row, row_positions in enumerate(positions):
            index = torch.tensor(row_positions, device=hidden_states[0].device)
            results.append(
                torch.stack(
                    [
                        h[row].index_select(0, index).to(torch.float32).mean(dim=0)
                        for h in hidden_states
                    ]
                ).cpu()
            )
    return torch.stack(results)


def position_direction(
    model: Model,
    bad: Sequence[PositionedText],
    good: Sequence[PositionedText],
) -> Tensor:
    """Unit difference of the mean residuals of bad and good items, per residual row."""
    difference = hidden_state_means(model, bad).mean(dim=0) - hidden_state_means(
        model, good
    ).mean(dim=0)
    return torch.nn.functional.normalize(difference, p=2, dim=1)


def module_io_means(model: Model, items: Sequence[PositionedText]) -> ModuleIO:
    """
    Like Model.get_module_io_batched, but each row is the mean of a module's input and output
    over the item's positions (exact for the linear modules ARA optimizes: the output of the
    mean input is the mean output). Modules that see only some tokens are skipped.
    """
    layers = model.get_layers()
    module_io: ModuleIO = [{} for _ in range(len(layers))]
    collected: dict[tuple[int, str, int], tuple[list[Tensor], list[Tensor]]] = {}
    batch_shape: list[tuple[int, int]] = []
    current_positions: list[list[int]] = []

    def select(tensor: Tensor) -> Tensor | None:
        prompt_count, position_count = batch_shape[0]
        if tensor.dim() == 2 and tensor.shape[0] == prompt_count * position_count:
            tensor = tensor.reshape(prompt_count, position_count, tensor.shape[1])
        if tensor.dim() < 3 or tuple(tensor.shape[:2]) != (
            prompt_count,
            position_count,
        ):
            return None
        return torch.stack(
            [
                tensor[row, positions].to(torch.float32).mean(dim=0)
                for row, positions in enumerate(current_positions)
            ]
        ).cpu()

    def get_hook(layer_index: int, component: str, module_index: int):
        def hook(module: Module, inputs: tuple[Tensor, ...], outputs: Tensor) -> None:
            selected_input = select(inputs[0])
            selected_output = select(outputs)
            if selected_input is None or selected_output is None:
                return
            entry = collected.setdefault(
                (layer_index, component, module_index), ([], [])
            )
            entry[0].append(selected_input)
            entry[1].append(selected_output)

        return hook

    handles = []
    for layer_index in range(len(layers)):
        for component, modules in model.get_layer_modules(layer_index).items():
            for module_index, module in enumerate(modules):
                handles.append(
                    module.register_forward_hook(
                        get_hook(layer_index, component, module_index)
                    )
                )

    try:
        for batch in batchify(list(items), model.settings.batch_size):
            input_ids, attention_mask, positions = _pad(model, batch)
            batch_shape[:] = [(input_ids.shape[0], input_ids.shape[1])]
            current_positions[:] = positions
            _forward(model, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    for (layer_index, component, module_index), (inputs, outputs) in collected.items():
        if sum(len(i) for i in inputs) != len(items):
            # Some batches did not reach the module; keep only complete records.
            continue
        module_io[layer_index].setdefault(component, {})[module_index] = (
            torch.cat(inputs),
            torch.cat(outputs),
        )
    return module_io


def append_module_io(base: ModuleIO, extra: ModuleIO) -> ModuleIO:
    """Rows of `extra` appended to those of `base`, for modules present in both."""
    merged: ModuleIO = []
    for layer_base, layer_extra in zip(base, extra):
        layer: dict[str, dict[int, tuple[Tensor, Tensor]]] = {}
        for component, io_map in layer_base.items():
            layer[component] = {}
            for module_index, (inputs, outputs) in io_map.items():
                more = layer_extra.get(component, {}).get(module_index)
                if more is None:
                    layer[component][module_index] = (inputs, outputs)
                else:
                    layer[component][module_index] = (
                        torch.cat([inputs, more[0].to(inputs.dtype)]),
                        torch.cat([outputs, more[1].to(outputs.dtype)]),
                    )
        merged.append(layer)
    return merged


def load_answer_items(
    model: Model,
    path: str,
    judged_path: str | None,
    bad_labels: Sequence[str],
    good_labels: Sequence[str],
    tokens: int,
    max_per_class: int,
) -> tuple[list[PositionedText], list[PositionedText]]:
    """Positioned bad and good answers from a labelled answers file."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Answers file {path} does not exist")
    answers = load_labelled_answers(path, judged_path, model.settings.system_prompt)
    bad, good = split_answers(answers, bad_labels, good_labels, max_per_class)
    print(
        f"* Answers from [bold]{path}[/]: {len(bad)} {'/'.join(bad_labels)}, "
        f"{len(good)} {'/'.join(good_labels)}"
    )
    if not bad or not good:
        raise ValueError(
            f"{path} needs answers labelled {list(bad_labels)} and {list(good_labels)}"
        )
    return (
        [answer_positions(model, a, tokens) for a in bad],
        [answer_positions(model, a, tokens) for a in good],
    )
