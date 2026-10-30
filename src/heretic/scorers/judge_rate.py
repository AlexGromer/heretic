# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Rate of non-answers according to a local judge model (fork addition, see heretic/judge.py).
# The [scorer.JudgeRate] settings also configure the judge re-ranking of the best trials
# after the optimization (setting n_judge_ranking_trials), whether or not this scorer is
# listed in `scorers`.

from collections.abc import Sequence
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from rich.markup import escape

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.judge import (
    DEFAULT_EMPTY_LABEL,
    DEFAULT_ENDPOINT,
    DEFAULT_LABELS,
    DEFAULT_MODEL,
    DEFAULT_REFUSAL_LABELS,
    JudgeClient,
    JudgeResult,
    load_rubric,
)
from heretic.scorer import Context, Score, Scorer
from heretic.utils import Prompt, format_dataset_specification, print


class Settings(BaseModel):
    score_name: str = Field(
        default="Judge non-answers",
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
            "Endpoint of the judge (default: a local Ollama server; the evaluation harness uses its LAN box). "
            "The prompts and responses are sent there."
        ),
    )

    api: Literal["ollama", "openai"] = Field(
        default="ollama",
        description=(
            '"ollama": Ollama /api/chat with thinking off and a JSON schema, as the evaluation harness does; '
            '"openai": an OpenAI-compatible /chat/completions endpoint (endpoint ends with /v1).'
        ),
    )

    message_format: Literal["harness", "template"] = Field(
        default="harness",
        description=(
            '"harness": the rubric is the system prompt and the user message is built as in '
            'the evaluation harness; "template": the rubric is the user message, with {prompt} and '
            "{response} replaced."
        ),
    )

    model: str = Field(
        default=DEFAULT_MODEL,
        description="Name of the judge model at the endpoint.",
    )

    rubric: str | None = Field(
        default=None,
        description=(
            'The judge prompt: "path/to/file.py:NAME" for a string constant of a Python file '
            '(parsed, not executed; e.g. "../<harness>/judge.py:SYSTEM_PROMPT"), or the path of '
            "a text file. Unset uses the built-in copy of the harness prompt v1."
        ),
    )

    labels: list[str] = Field(
        default=list(DEFAULT_LABELS),
        description="Labels the rubric asks for.",
    )

    refusal_labels: list[str] = Field(
        default=list(DEFAULT_REFUSAL_LABELS),
        description="Labels that count as not answered (the score is their share).",
    )

    empty_label: str = Field(
        default=DEFAULT_EMPTY_LABEL,
        description="Label of empty answers, which are not sent to the judge.",
    )

    max_tokens: PositiveInt = Field(
        default=160,
        description="Maximum number of tokens of the judge's reply (the evaluation harness: 160).",
    )

    timeout: PositiveFloat = Field(
        default=180.0,
        description="Timeout of one judge request in seconds.",
    )

    retries: PositiveInt = Field(
        default=3,
        description="Attempts per answer (with exponential backoff) before the evaluation fails.",
    )

    print_responses: bool = Field(
        default=False,
        description="Whether to print prompt/response pairs with the judge's labels.",
    )

    @model_validator(mode="after")
    def validate_labels(self) -> "Settings":
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError(
                f"labels must be non-empty and distinct, got {self.labels}"
            )
        if not self.refusal_labels or not set(self.refusal_labels) <= set(self.labels):
            raise ValueError(
                f"refusal_labels must be a non-empty subset of labels {self.labels}, "
                f"got {self.refusal_labels}"
            )
        if self.empty_label not in self.labels:
            raise ValueError(
                f"empty_label must be one of labels {self.labels}, got {self.empty_label!r}"
            )
        return self

    def make_client(self) -> JudgeClient:
        return JudgeClient(
            endpoint=self.endpoint,
            model=self.model,
            api=self.api,
            message_format=self.message_format,
            rubric=load_rubric(self.rubric),
            labels=tuple(self.labels),
            empty_label=self.empty_label,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            retries=self.retries,
        )


def judge_responses(
    client: JudgeClient,
    prompts: Sequence[Prompt],
    responses: Sequence[str],
    refusal_labels: Sequence[str],
) -> JudgeResult:
    """Labels the answers (the client removes the reasoning blocks first)."""
    labels = [
        client.classify(prompt.user, response)
        for prompt, response in zip(prompts, responses)
    ]
    return JudgeResult(labels=labels, refusal_labels=tuple(refusal_labels))


def format_counts(result: JudgeResult, labels: Sequence[str]) -> str:
    counts = result.counts
    return ", ".join(f"{label} {counts[label]}" for label in labels)


class JudgeRate(Scorer):
    """
    Share of responses that a judge model labels as not answered (by default REFUSAL
    or PARTIAL).

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
        self.client = self.settings.make_client()
        print(
            f"* Judge: [bold]{escape(self.settings.model)}[/] at "
            f"[bold]{escape(self.settings.endpoint)}[/], prompt {self.client.prompt_version}"
        )

    def get_score(self, ctx: Context) -> Score:
        responses = ctx.get_responses(self.prompts)
        result = judge_responses(
            self.client,
            self.prompts,
            responses,
            self.settings.refusal_labels,
        )

        if self.settings.print_responses:
            for prompt, response, label in zip(self.prompts, responses, result.labels):
                print()
                print(f"[bold]Prompt:[/] {escape(prompt.user)}")
                shown = escape(response) if response.strip() else "[italic]\\[empty][/]"
                print(f"[bold]Response:[/] {shown}")
                print(f"[bold]Judge:[/] {label}")

        total = len(self.prompts)
        return Score(
            value=result.rate,
            rich_display=(
                f"[bold]{result.refusals}[/]/{total} "
                f"({format_counts(result, self.settings.labels)})"
            ),
            md_display=f"{result.refusals}/{total}",
        )
