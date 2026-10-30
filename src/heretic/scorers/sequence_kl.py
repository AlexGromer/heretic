# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# KL divergence over a whole answer instead of its first token (fork addition).
#
# KLDivergence compares the distribution of the FIRST generated token on harmless prompts.
# Damage that only shows later in an answer (the model starts well and then derails,
# repeats or switches language) does not enter that number, which is why the research
# report lists "KL only measures the first token" as a blind spot and asks for a check on
# longer generations.
#
# This scorer records, once on the original model, the greedy continuation of every prompt
# (`tokens` tokens) together with the original distributions along it, and then measures how
# far the modified model's distributions are from those at the same positions (teacher
# forcing on the original continuation). Averaged over positions and prompts.
#
# Memory: keeping the full vocabulary for every position would be gigabytes, so only the
# `top_k` most likely tokens of the original distribution are kept per position, and the
# divergence is computed on that support, with both sides renormalized over it. The share
# of the original probability mass the support covers is reported, so the approximation is
# visible: with the default k = 128 it is normally above 0.99.

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, PositiveInt
from torch import Tensor

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.positions import PositionedText, chat_text, forced_logits
from heretic.scorer import Context, Score, Scorer
from heretic.utils import Prompt, batchify, format_dataset_specification, print


class Settings(BaseModel):
    score_name: str = Field(
        default="Sequence KL",
        description="Name that describes what the sequence KL divergence measures.",
    )

    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts whose answers are compared with the original model's.",
    )

    tokens: PositiveInt = Field(
        default=32,
        description=(
            "Number of tokens of the original model's greedy continuation that are compared. "
            "The cost per evaluation is one forward pass over prompt + these tokens."
        ),
    )

    top_k: PositiveInt = Field(
        default=128,
        description=(
            "Number of most likely tokens of the original distribution kept per position. "
            "The divergence is computed on that support with both sides renormalized over it; "
            "the covered share of the original mass is printed at startup."
        ),
    )


def masked_mean(values: Tensor, mask: Tensor) -> float:
    total = mask.sum()
    return float((values * mask).sum() / total) if total > 0 else 0.0


class SequenceKLDivergence(Scorer):
    """
    KL divergence from the original model over the first tokens of an answer, not only
    its first token. See the module docstring for the top-k approximation.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    def init(self, ctx: Context) -> None:
        model = ctx.get_model()
        print()
        print(
            f"Loading {self.settings.score_name} evaluation prompts from "
            f"[bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        print(
            f"* Recording the original model's greedy continuation of "
            f"[bold]{self.settings.tokens}[/] tokens and its distributions..."
        )
        self._record_baseline(model)

        covered = masked_mean(self._baseline_mass, self._valid)
        print(
            f"* Comparing {int(self._valid.sum())} positions; the top-"
            f"{self.settings.top_k} support covers [bold]{covered:.4f}[/] of the "
            "original probability mass"
        )

    def _record_baseline(self, model) -> None:
        count = len(self.prompts)
        tokens = self.settings.tokens
        top_k = self.settings.top_k

        self._continuations: list[list[int]] = []
        self._valid = torch.zeros(count, tokens)
        self._top_ids = torch.zeros(count, tokens, top_k, dtype=torch.long)
        self._top_logprobs = torch.full((count, tokens, top_k), -float("inf"))
        self._baseline_mass = torch.zeros(count, tokens)

        row = 0
        for batch in batchify(self.prompts, model.settings.batch_size):
            inputs, outputs = model.generate(
                batch,
                max_new_tokens=tokens,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
            )
            prompt_length = inputs["input_ids"].shape[1]
            sequences = outputs.sequences[:, prompt_length:]
            step_logits = outputs.logits

            for index in range(len(batch)):
                continuation: list[int] = []
                for step, logits in enumerate(step_logits):
                    token = int(sequences[index, step])
                    # A sequence that finished is padded from here on; those positions
                    # are not part of the original answer.
                    if step > 0 and token in self._stop_tokens(model):
                        break
                    logprobs = F.log_softmax(logits[index].to(torch.float32), dim=-1)
                    values, ids = logprobs.topk(min(top_k, logprobs.shape[-1]))
                    self._top_ids[row, step, : ids.shape[0]] = ids.cpu()
                    self._top_logprobs[row, step, : values.shape[0]] = values.cpu()
                    self._baseline_mass[row, step] = float(values.exp().sum())
                    self._valid[row, step] = 1.0
                    continuation.append(token)
                self._continuations.append(continuation)
                row += 1

            del outputs, step_logits

        self._items = [
            self._forced_item(model, prompt, continuation)
            for prompt, continuation in zip(self.prompts, self._continuations)
        ]
        # Renormalize the baseline over the kept support, so both sides are distributions
        # over the same tokens.
        mass = self._baseline_mass.clamp_min(1e-30).log()
        self._top_logprobs = self._top_logprobs - mass[:, :, None]

    @staticmethod
    def _stop_tokens(model) -> set[int]:
        return {
            token
            for token in (model.tokenizer.eos_token_id, model.tokenizer.pad_token_id)
            if token is not None
        }

    def _forced_item(
        self, model, prompt: Prompt, continuation: list[int]
    ) -> PositionedText:
        """Prompt plus the original continuation; the positions predict its tokens."""
        prompt_ids = model.tokenizer(
            chat_text(model, prompt.system, prompt.user),
            add_special_tokens=False,
        )["input_ids"]
        prompt_ids = list(prompt_ids)
        if not continuation:
            return PositionedText(prompt_ids, [len(prompt_ids) - 1])
        input_ids = prompt_ids + continuation
        start = len(prompt_ids) - 1
        return PositionedText(input_ids, list(range(start, start + len(continuation))))

    def get_score(self, ctx: Context) -> Score:
        model = ctx.get_model()
        divergences = torch.zeros_like(self._valid)

        for index, logits in forced_logits(model, self._items):
            logprobs = F.log_softmax(logits, dim=-1)
            length = int(self._valid[index].sum())
            for step in range(min(length, logprobs.shape[0])):
                ids = self._top_ids[index, step]
                baseline = self._top_logprobs[index, step]
                keep = torch.isfinite(baseline)
                current = logprobs[step].index_select(0, ids[keep])
                current = current - torch.logsumexp(current, dim=0)
                baseline = baseline[keep]
                divergences[index, step] = float(
                    (baseline.exp() * (baseline - current)).sum()
                )

            # Float error can make a per-position term a tiny negative number; the divergence
        # itself cannot be negative.
        value = max(0.0, masked_mean(divergences, self._valid))
        per_prompt = (divergences * self._valid).sum(dim=1) / self._valid.sum(
            dim=1
        ).clamp_min(1)
        worst = float(per_prompt.max()) if per_prompt.numel() else 0.0

        return Score(
            value=value,
            rich_display=f"[bold]{value:.4f}[/] [grey50](worst prompt {worst:.4f})[/]",
            md_display=f"{value:.4f}",
        )

    def get_baseline_score(self, ctx: Context) -> Score:
        return Score(
            value=0,
            rich_display="[bold]0[/] [italic](by definition)[/]",
            md_display="0 *(by definition)*",
        )
