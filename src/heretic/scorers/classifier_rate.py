# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Share of non-answers according to a classifier distilled from the judge
# (fork addition, see heretic/classifier.py). Fast enough to be an optimization objective.

from pydantic import BaseModel, Field, PositiveInt
from rich.markup import escape

from heretic.classifier import Classifier
from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.judge import DEFAULT_REFUSAL_LABELS
from heretic.scorer import Context, Score, Scorer
from heretic.utils import format_dataset_specification, print


class Settings(BaseModel):
    score_name: str = Field(
        default="Non-answers",
        description="Name that describes what the classifier counts.",
    )

    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts whose responses are classified.",
    )

    model_path: str | None = Field(
        default=None,
        description="Directory written by heretic-train-classifier (required).",
    )

    refusal_labels: list[str] = Field(
        default=list(DEFAULT_REFUSAL_LABELS),
        description="Labels that count as not answered.",
    )

    device: str = Field(
        default="auto",
        description='Device of the classifier ("auto": CUDA if available).',
    )

    batch_size: PositiveInt = Field(
        default=16,
        description="Batch size of the classifier.",
    )

    print_responses: bool = Field(
        default=False,
        description="Whether to print prompt/response pairs with the predicted labels.",
    )


class ClassifierRate(Scorer):
    """
    Share of responses that the classifier labels as not answered (REFUSAL or PARTIAL by
    default). Responses are max_response_length tokens long; PARTIAL answers often turn
    evasive after the first sentences, so use at least 256.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        # The classifier is a local model that a reproduction would not have.
        return False

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    def init(self, ctx: Context) -> None:
        if self.settings.model_path is None:
            raise ValueError(
                "[scorer.ClassifierRate] model_path must point to a classifier "
                "trained with heretic-train-classifier"
            )
        print()
        print(
            f"Loading {self.settings.score_name} evaluation prompts from "
            f"[bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")
        self.classifier = Classifier.load(
            self.settings.model_path, self.settings.device
        )
        unknown = set(self.settings.refusal_labels) - set(self.classifier.labels)
        if unknown:
            raise ValueError(
                f"refusal_labels {sorted(unknown)} are not labels of the classifier "
                f"{self.classifier.labels}"
            )
        print(
            f"* Classifier [bold]{escape(self.settings.model_path)}[/] "
            f"on {self.classifier.device}, labels {', '.join(self.classifier.labels)}"
        )

    def get_score(self, ctx: Context) -> Score:
        responses = ctx.get_responses(self.prompts)
        predicted = self.classifier.predict(
            [prompt.user for prompt in self.prompts],
            responses,
            self.settings.batch_size,
        )
        non_answers = sum(label in self.settings.refusal_labels for label in predicted)

        if self.settings.print_responses:
            for prompt, response, label in zip(self.prompts, responses, predicted):
                print()
                print(f"[bold]Prompt:[/] {escape(prompt.user)}")
                shown = escape(response) if response.strip() else "[italic]\\[empty][/]"
                print(f"[bold]Response:[/] {shown}")
                print(f"[bold]Classifier:[/] {label}")

        counts = ", ".join(
            f"{label} {predicted.count(label)}" for label in self.classifier.labels
        )
        total = len(self.prompts)
        return Score(
            value=non_answers / total,
            rich_display=f"[bold]{non_answers}[/]/{total} ({counts})",
            md_display=f"{non_answers}/{total}",
        )
