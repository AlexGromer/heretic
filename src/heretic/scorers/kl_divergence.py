# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import format_dataset_specification, print


class Settings(BaseModel):
    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts used to measure KL divergence from original model.",
    )


def kl_divergence_batchmean(logprobs: Tensor, baseline_logprobs: Tensor) -> float:
    """
    KL(baseline || current), summed over the vocabulary and averaged over the prompts.

    Fork addition: some models mask reserved tokens with -inf logits (e.g. Gemma 4 12B,
    reported by TrevorS/gemma-4-abliteration). There, F.kl_div computes 0 * (-inf - -inf)
    = NaN. Tokens the baseline gives zero probability contribute 0 to the divergence by
    definition, so they are masked. Without non-finite values, the computation is the
    plain F.kl_div as before.
    """
    if bool(torch.isfinite(baseline_logprobs).all() and torch.isfinite(logprobs).all()):
        return F.kl_div(
            logprobs,
            baseline_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()

    impossible = torch.isneginf(baseline_logprobs)
    terms = baseline_logprobs.exp() * (baseline_logprobs - logprobs)
    terms = torch.where(impossible, torch.zeros_like(terms), terms)
    return (terms.sum() / baseline_logprobs.shape[0]).item()


class KLDivergence(Scorer):
    """
    KL divergence between current model and baseline.

    Measures how much the model's behavior has drifted from baseline.
    Lower is better (less damage).
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "KL divergence"

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading KL divergence evaluation prompts from [bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        print("* Obtaining baseline first-token probability distributions...")
        baseline_logits = ctx.get_logits(self.prompts)

        self._baseline_logprobs = F.log_softmax(baseline_logits, dim=-1)

    def get_score(self, ctx: Context) -> Score:
        logits = ctx.get_logits(self.prompts)
        logprobs = F.log_softmax(logits, dim=-1)

        kl_divergence = kl_divergence_batchmean(logprobs, self._baseline_logprobs)

        return Score(
            value=kl_divergence,
            rich_display=f"[bold]{kl_divergence:.4f}[/]",
            md_display=f"{kl_divergence:.4f}",
        )

    def get_baseline_score(self, ctx: Context) -> Score:
        return Score(
            value=0,
            rich_display="[bold]0[/] [italic](by definition)[/]",
            md_display="0 *(by definition)*",
        )
