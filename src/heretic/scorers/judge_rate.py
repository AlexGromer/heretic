# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Refusal rate according to a local judge model (fork addition, see heretic/judge.py).

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, field_validator
from rich.markup import escape

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.judge import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_REFUSAL_LABELS,
    LABELS,
    JudgeClient,
)
from heretic.scorer import Context, Score, Scorer
from heretic.scorers.keyword_rate import DEFAULT_COT_BLOCKS, strip_chain_of_thought
from heretic.utils import format_dataset_specification, print


class Settings(BaseModel):
    score_name: str = Field(
        default="Judge refusals",
        description="Name that describes what the judge counts.",
    )

    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts whose responses the judge classifies.",
    )

    endpoint: str = Field(
        default=DEFAULT_ENDPOINT,
        description=(
            "OpenAI-compatible chat completions endpoint of the judge (default: a local Ollama "
            "server). The prompts and responses are sent there."
        ),
    )

    model: str = Field(
        default=DEFAULT_MODEL,
        description="Name of the judge model at the endpoint.",
    )

    refusal_labels: list[str] = Field(
        default=list(DEFAULT_REFUSAL_LABELS),
        description=f"Labels that count as refusals, out of {', '.join(LABELS)}.",
    )

    max_tokens: PositiveInt = Field(
        default=512,
        description="Maximum number of tokens of the judge's reply (thinking models need more).",
    )

    timeout: PositiveFloat = Field(
        default=120.0,
        description="Timeout of one judge request in seconds.",
    )

    print_responses: bool = Field(
        default=False,
        description="Whether to print prompt/response pairs with the judge's labels.",
    )

    @field_validator("refusal_labels")
    @classmethod
    def validate_refusal_labels(cls, value: list[str]) -> list[str]:
        unknown = [label for label in value if label not in LABELS]
        if unknown or not value:
            raise ValueError(
                f"refusal_labels must be a non-empty subset of {list(LABELS)}, got {value}"
            )
        return value


class JudgeRate(Scorer):
    """
    Fraction of responses that a judge model classifies as a refusal.

    Each evaluation sends every prompt/response pair to the judge, which is slow and,
    unlike keyword counting, not exactly reproducible. Intended with
    optimization = "none", next to KeywordRate, or for --evaluate-model.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return False

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading {self.settings.score_name} evaluation prompts from "
            f"[bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")
        print(
            f"* Judge: [bold]{escape(self.settings.model)}[/] at "
            f"[bold]{escape(self.settings.endpoint)}[/]"
        )
        self.client = JudgeClient(
            endpoint=self.settings.endpoint,
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            timeout=self.settings.timeout,
        )

    def _cot_blocks(self) -> list[tuple[str, str]]:
        blocks = getattr(self.heretic_settings, "chain_of_thought_blocks", None)
        if not blocks:
            return list(DEFAULT_COT_BLOCKS)
        return [(initializer, finalizer) for initializer, finalizer in blocks]

    def get_score(self, ctx: Context) -> Score:
        responses = ctx.get_responses(self.prompts)
        blocks = self._cot_blocks()
        refusals = 0

        for prompt, response in zip(self.prompts, responses):
            # The judge grades the answer, not the reasoning before it.
            answer = strip_chain_of_thought(response, blocks)
            label = self.client.classify(prompt.user, answer)
            if label in self.settings.refusal_labels:
                refusals += 1

            if self.settings.print_responses:
                print()
                print(f"[bold]Prompt:[/] {escape(prompt.user)}")
                shown = escape(response) if response.strip() else "[italic]\\[empty][/]"
                print(f"[bold]Response:[/] {shown}")
                print(f"[bold]Judge:[/] {label}")

        return Score(
            value=refusals / len(self.prompts),
            rich_display=f"[bold]{refusals}[/]/{len(self.prompts)}",
            md_display=f"{refusals}/{len(self.prompts)}",
        )
