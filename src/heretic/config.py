# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# !!!IMPORTANT!!!
#
# Any settings added to the classes defined in this module
# must be evaluated for privacy implications and have
# exclude=True set in their field definitions if appropriate.


class QuantizationMethod(str, Enum):
    NONE = "none"
    BNB_4BIT = "bnb_4bit"


class ExportStrategy(str, Enum):
    MERGE = "merge"
    ADAPTER = "adapter"


class SingleDatasetSpecification(BaseModel):
    dataset: str = Field(
        description="Hugging Face dataset ID, or path to dataset on disk."
    )

    commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the dataset.",
    )

    config: str | None = Field(
        default=None,
        description=(
            "Dataset config/subset name. Each config can have its own split. "
            "Used to load a specific config of a dataset that has multiple configurations."
        ),
    )

    split: str | None = Field(
        default=None,
        description="Portion of the dataset to use. Required for datasets, optional for plain text files.",
    )

    column: str | None = Field(
        default=None,
        description="Column in the dataset that contains the prompts. Required for datasets, ignored for plain text files.",
    )

    prefix: str = Field(
        default="",
        description="Text to prepend to each prompt.",
    )

    suffix: str = Field(
        default="",
        description="Text to append to each prompt.",
    )

    system_prompt: str | None = Field(
        default=None,
        description="System prompt to use with the prompts (overrides global system prompt if set).",
    )

    sample_size: PositiveInt | None = Field(
        default=None,
        description=(
            "Number of prompts to draw at random from the selected split (fork addition). "
            "Useful for datasets that are sorted by category, such as OR-Bench, where a plain "
            'slice like "train[:100]" would only cover the first categories. '
            "Unset uses the whole split."
        ),
    )

    sample_seed: int = Field(
        default=0,
        description="Seed of the random draw selected by sample_size. The same seed always draws the same prompts.",
    )


DatasetSpecification: TypeAlias = (
    SingleDatasetSpecification | list[SingleDatasetSpecification]
)


class ScorerConfig(BaseModel):
    """
    Configuration for a scorer plugin.

    TOML format:
    - { plugin = "<plugin>", optimization = "<optimization>", instance_name = "<optional>" }
    """

    plugin: str = Field(
        description=(
            "Plugin to load. Either a file path with class name "
            "(`path/to/plugin.py:ClassName`) or a fully-qualified import path "
            "(`module.submodule.ClassName`)."
        ),
    )

    optimization: Literal["minimize", "maximize", "none"] = Field(
        description=(
            "Optimization direction for this scorer. "
            '"minimize" / "maximize" to include the scorer as an objective, '
            '"none" to compute the score without optimizing for it.'
        ),
    )

    instance_name: str | None = Field(
        default=None,
        description=(
            "Optional name to distinguish multiple instances of the same plugin class. "
            "Instance-specific settings live under `[scorer.<ClassName>_<instance_name>]`."
        ),
    )

    @field_validator("instance_name")
    @classmethod
    def validate_instance_name(cls, value: str | None) -> str | None:
        if value is None:
            return value

        if not value.strip():
            raise ValueError("cannot be empty or whitespace")

        if "." in value:
            raise ValueError("'.' is not allowed")

        if any(char.isspace() for char in value):
            raise ValueError("whitespace is not allowed")

        return value


class ModifierConfig(BaseModel):
    """
    Configuration for a modifier plugin.

    TOML format:
    - { plugin = "<plugin>", instance_name = "<optional>" }
    """

    plugin: str = Field(
        description=(
            "Plugin to load. Either a file path with class name "
            "(`path/to/plugin.py:ClassName`) or a fully-qualified import path "
            "(`module.submodule.ClassName`)."
        ),
    )

    instance_name: str | None = Field(
        default=None,
        description=(
            "Optional name to distinguish multiple instances of the same plugin class. "
            "Instance-specific settings live under `[modifier.<ClassName>_<instance_name>]`."
        ),
    )

    @field_validator("instance_name")
    @classmethod
    def validate_instance_name(cls, value: str | None) -> str | None:
        if value is None:
            return value

        if not value.strip():
            raise ValueError("cannot be empty or whitespace")

        if "." in value:
            raise ValueError("'.' is not allowed")

        if any(char.isspace() for char in value):
            raise ValueError("whitespace is not allowed")

        return value


class BenchmarkSpecification(BaseModel):
    task: str = Field(
        description="Task ID of the benchmark in the Language Model Evaluation Harness."
    )

    name: str = Field(description="Name of the benchmark for presentation purposes.")

    description: str = Field(
        description="Description of the benchmark for presentation purposes."
    )


class Settings(BaseSettings):
    model: str = Field(description="Hugging Face model ID, or path to model on disk.")

    model_commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the model.",
    )

    evaluate_model: str | None = Field(
        default=None,
        description=(
            "If this model ID or path is set, then instead of abliterating the main model, "
            "evaluate this model relative to the main model."
        ),
        exclude=True,
    )

    collect_reproducibles: str | None = Field(
        default=None,
        description=(
            "If this directory path is set, then instead of abliterating a model, "
            "download all reproduce.json files from public Heretic model repositories "
            "on Hugging Face, and store them in that directory for archival purposes."
        ),
        exclude=True,
    )

    reproduce: str | None = Field(
        default=None,
        description=(
            "If this path or URL to a reproduce.json file is set, load reproduction information "
            "from that file, and attempt to reproduce the abliterated model it originated from."
        ),
        exclude=True,
    )

    dtypes: list[str] = Field(
        default=[
            # In practice, "auto" almost always means bfloat16.
            "auto",
            # If that doesn't work (e.g. on pre-Ampere hardware), fall back to float16.
            "float16",
            # If "auto" resolves to float32, and that fails because it is too large,
            # and float16 fails due to range issues, try bfloat16.
            "bfloat16",
            # If neither of those work, fall back to float32 (which will of course fail
            # if that was the dtype "auto" resolved to).
            "float32",
        ],
        description=(
            "List of PyTorch dtypes to try when loading model tensors. "
            "If loading with a dtype fails, the next dtype in the list will be tried."
        ),
    )

    quantization: QuantizationMethod = Field(
        default=QuantizationMethod.NONE,
        description=(
            "Quantization method to use when loading the model. Options: "
            '"none" (no quantization), '
            '"bnb_4bit" (4-bit quantization using bitsandbytes).'
        ),
    )

    device_map: str | dict[str, int | str] = Field(
        default="auto",
        description="Device map to pass to Accelerate when loading the model.",
    )

    max_memory: dict[str, str] | None = Field(
        default=None,
        description='Maximum memory to allocate per device (e.g., { "0" = "20GB", "cpu" = "64GB" }).',
    )

    offload_outputs_to_cpu: bool = Field(
        default=True,
        description=(
            "Whether to move intermediate analysis tensors (such as residuals and logprobs) "
            "to CPU memory as soon as possible to reduce peak VRAM usage. "
            "This lowers peak VRAM usage during residual analysis and evaluation, "
            "but may slightly reduce performance due to host/device transfers."
        ),
    )

    fused_experts_backup_device: str = Field(
        default="cpu",
        description=(
            "Device on which to keep the original weights of fused routed experts "
            "(models whose experts are stored as one 3D tensor per layer, e.g. Qwen3.5 MoE), "
            "which are modified directly instead of through a LoRA adapter, and must be "
            "restored between trials. The backups take as much memory as the modified tensors, "
            'so the default "cpu" keeps them out of VRAM. Use e.g. "cuda:0" for faster '
            "resets if you have spare VRAM."
        ),
    )

    batch_size: NonNegativeInt = Field(
        default=0,  # auto
        description="Number of input sequences to process in parallel (0 = auto).",
    )

    max_batch_size: PositiveInt = Field(
        default=128,
        description="Maximum batch size to try when automatically determining the optimal batch size.",
        # When storing a settings object, the batch size is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    batch_size_test_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:256]",
            column="text",
        ),
        description="Dataset of prompts to use for automatically determining the optimal batch size.",
        # When storing a settings object, the batch size is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    max_response_length: PositiveInt = Field(
        default=100,
        description="Maximum number of tokens to generate for each response.",
    )

    response_prefix: str | None = Field(
        default=None,
        description=(
            "Common prefix to assume for all responses, so that evaluation happens "
            "at the point where responses start to differ for different prompts. "
            "If not set, the prefix is determined automatically by comparing multiple responses."
        ),
    )

    response_prefix_test_prompts: DatasetSpecification = Field(
        default=[
            SingleDatasetSpecification(
                dataset="mlabonne/harmless_alpaca",
                split="train[:100]",
                column="text",
            ),
            SingleDatasetSpecification(
                dataset="mlabonne/harmful_behaviors",
                split="train[:100]",
                column="text",
            ),
        ],
        description="Dataset of prompts to use for automatically determining the response prefix.",
        # When storing a settings object, the response prefix is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    chain_of_thought_blocks: list[tuple[str, str]] = Field(
        default=[
            # Most thinking models.
            (
                "<think>",
                "</think>",
            ),
            # gpt-oss.
            (
                "<|channel|>analysis<|message|>",
                "<|end|><|start|>assistant<|channel|>final<|message|>",
            ),
            # Unknown, suggested by user.
            (
                "<thought>",
                "</thought>",
            ),
            # Unknown, suggested by user.
            (
                "[THINK]",
                "[/THINK]",
            ),
        ],
        description=(
            "List of pairs of the form (cot_initializer, cot_finalizer) used to skip "
            "the Chain-of-Thought block in responses, so that evaluation happens "
            "at the start of the actual response."
        ),
        # When storing a settings object, the response prefix is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    print_debug_information: bool = Field(
        default=False,
        description="Whether to print additional information that can help with debugging.",
        exclude=True,
    )

    responses_file: str | None = Field(
        default=None,
        description=(
            "Path of a JSON Lines file that every KeywordRate evaluation appends its prompt/response "
            "pairs to (fork addition, for auditing the counted matches by hand). Each line holds the "
            "evaluation number of that scorer (0 = baseline, then one per evaluation in this process), "
            "the score name, the prompts, the response and whether it counted as a match."
        ),
        exclude=True,
    )

    scorers: list[ScorerConfig] = Field(
        default=[
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
            ),
            ScorerConfig(
                plugin="heretic.scorers.kl_divergence.KLDivergence",
                optimization="minimize",
            ),
        ],
        description=(
            "List of scorer plugin configs. Each entry is an object "
            "{ plugin = <plugin>, optimization = <optimization>, instance_name = <optional> }. "
            '<optimization> is one of "minimize", "maximize", or "none" (do not optimize).'
        ),
    )

    modifiers: list[ModifierConfig] = Field(
        default=[
            ModifierConfig(
                plugin="heretic.modifiers.ara.ARA",
            ),
        ],
        description=(
            "List of modifier plugin configs. Each entry is an object "
            "{ plugin = <plugin>, instance_name = <optional> }. "
            "Note that only a single modifier can currently be applied, "
            "and this list must contain exactly one entry."
        ),
    )

    n_trials: PositiveInt = Field(
        default=100,
        description="Number of abliteration trials to run during optimization.",
    )

    n_startup_trials: NonNegativeInt = Field(
        default=30,
        description="Number of trials that use random sampling for the purpose of exploration.",
    )

    n_judge_ranking_trials: NonNegativeInt = Field(
        default=0,
        description=(
            "Number of the best trials (in the order of the Pareto front, within "
            "judge_ranking_max_kl) to judge after the optimization with the judge configured in "
            "[scorer.JudgeRate] (fork addition). The original model is judged once as well. The "
            "menu then lists the judged trials first, sorted by their share of non-answers "
            "(REFUSAL + PARTIAL by default), which changes what --trial-index points to. "
            "Results are stored in the study. 0 disables it."
        ),
    )

    judge_ranking_mode: Literal["judge", "generate"] = Field(
        default="judge",
        description=(
            '"judge": judge the answers of the ranked trials here, via the judge endpoint. '
            '"generate": only generate them and write them to judge_ranking_output in the format of '
            "refusal-eval's responses.jsonl, for judging in a separate process (e.g. judge_colab.py, "
            "which must not share the GPU with the evaluated model); then select the trial with "
            "--trial-number."
        ),
    )

    judge_ranking_merged: bool = Field(
        default=True,
        description=(
            "Merge each ranked trial into the weights in memory before generating its answers, so "
            "that they come from the same weights an export contains (the live LoRA rounds "
            "differently, upstream issue #490). Costs one model reload per ranked trial; false "
            "generates with the live LoRA (faster, slightly different from the exported model)."
        ),
    )

    judge_ranking_benchmarks: list[str] = Field(
        default=[],
        description=(
            "Benchmarks to run on the original model and on the ranked candidate trials after "
            "the optimization (fork addition), as task IDs of the Language Model Evaluation "
            'Harness, optionally with the metric: ["gsm8k:exact_match,strict-match", "mmlu"]. '
            "Abliteration can cost capabilities, and the BenchmarkScore scorer is too slow to "
            "run on every trial. Empty disables it; the results are stored in the study and "
            "shown next to the judged trials."
        ),
    )

    judge_ranking_output: str | None = Field(
        default=None,
        description=(
            'JSON Lines file the "generate" mode appends the answers to (refusal-eval responses '
            "format; keys already in the file are skipped, so an interrupted run resumes)."
        ),
        exclude=True,
    )

    judge_ranking_prompts_file: str | None = Field(
        default=None,
        description=(
            "JSON Lines file of the ranking prompts with id, category and prompt fields, such as "
            "refusal-eval's train/bad_eval.meta.jsonl. Unset uses the [scorer.JudgeRate] prompts "
            "(ids are then <set>-<line>). Do not rank on the evaluation hold-out, or the hold-out "
            "no longer measures the selected model independently."
        ),
        exclude=True,
    )

    judge_ranking_set: str = Field(
        default="bad_eval",
        description='Set name written into the generated rows (refusal-eval "set").',
    )

    judge_ranking_model_name: str | None = Field(
        default=None,
        description="Model label written into the generated rows (default: the last part of the model path).",
    )

    judge_ranking_max_kl: float = Field(
        default=0.15,
        description="Largest KL divergence of a trial that is judged.",
    )

    judge_ranking_response_length: PositiveInt = Field(
        default=512,
        description=(
            "Maximum number of tokens of each answer that is judged (refusal-eval uses 512). "
            "Longer than max_response_length, because PARTIAL answers often turn evasive later."
        ),
    )

    judge_ranking_batch_size: NonNegativeInt = Field(
        default=0,
        description=(
            "Batch size for generating the judged answers (0 = batch_size). Longer answers need "
            "more memory than the batch size was determined for."
        ),
    )

    n_merged_score_trials: NonNegativeInt = Field(
        default=0,
        description=(
            "Number of the best trials to evaluate again on their merged models after the "
            "optimization (fork addition, upstream issue #490). The live LoRA the trials are "
            "scored with rounds differently than the merged model that is exported, so the "
            "exported model can score differently. Each re-scored trial costs one merge, one "
            "evaluation and one model reload. If every trial of the Pareto front was re-scored, "
            "the menu is sorted by the merged scores. 0 disables it."
        ),
    )

    seed: int | None = Field(
        default=None,
        description=(
            "Random seed for reproducible optimization. "
            "Applies to Python's random module, NumPy, PyTorch, and Optuna."
        ),
    )

    study_checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory to save and load study progress to/from.",
        exclude=True,
    )

    benchmarks: list[BenchmarkSpecification] = Field(
        default=[
            BenchmarkSpecification(
                task="agieval",
                name="AGIEval",
                description="A Human-Centric Benchmark for Evaluating Foundation Models",
            ),
            BenchmarkSpecification(
                task="bbh",
                name="BIG-Bench Hard (BBH)",
                description="Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
            ),
            BenchmarkSpecification(
                task="commonsense_qa",
                name="CommonsenseQA",
                description="A Question Answering Challenge Targeting Commonsense Knowledge",
            ),
            BenchmarkSpecification(
                task="eq_bench",
                name="EQ-Bench",
                description="An Emotional Intelligence Benchmark for Large Language Models",
            ),
            BenchmarkSpecification(
                task="gsm8k",
                name="GSM8K",
                description="Training Verifiers to Solve Math Word Problems",
            ),
            BenchmarkSpecification(
                task="hellaswag",
                name="HellaSwag",
                description="Can a Machine Really Finish Your Sentence?",
            ),
            BenchmarkSpecification(
                task="ifeval",
                name="IFEval",
                description="Instruction-Following Evaluation for Large Language Models",
            ),
            BenchmarkSpecification(
                task="mmlu",
                name="MMLU",
                description="Measuring Massive Multitask Language Understanding",
            ),
            BenchmarkSpecification(
                task="mmlu_pro",
                name="MMLU-Pro",
                description="A More Robust and Challenging Multi-Task Language Understanding Benchmark",
            ),
            BenchmarkSpecification(
                task="piqa",
                name="PIQA",
                description="Reasoning about Physical Commonsense in Natural Language",
            ),
            BenchmarkSpecification(
                task="winogrande",
                name="WinoGrande",
                description="An Adversarial Winograd Schema Challenge at Scale",
            ),
        ],
        description="Benchmarks to offer to the user for evaluating abliterated models.",
        exclude=True,
    )

    max_shard_size: PositiveInt | str = Field(
        default="5GB",
        description="Maximum size for individual safetensors files generated when exporting a model.",
    )

    export_strategy: ExportStrategy | None = Field(
        default=None,
        description='How to export the model: "merge", "adapter", or unset to prompt the user.',
    )

    checkpoint_action: str | None = Field(
        default=None,
        description='Action to take in case a checkpoint exists: "continue", "restart", or unset to prompt the user.',
    )

    trial_index: NonNegativeInt | None = Field(
        default=None,
        description="Index (in the sorted Pareto front) of the trial to use, or unset to prompt the user.",
    )

    trial_number: PositiveInt | None = Field(
        default=None,
        description=(
            'Number of the trial to use, as shown in "[Trial N]" (fork addition). Unlike trial_index, '
            "it does not depend on the order of the menu, and any completed trial can be selected, "
            "not only Pareto optimal ones."
        ),
    )

    n_additional_trials: PositiveInt | None = Field(
        default=None,
        description="Number of additional trials to run, or unset to prompt the user.",
    )

    model_action: str | None = Field(
        default=None,
        description='Action to take with the decensored model: "save", "upload", or unset to prompt the user.',
    )

    save_directory: str | None = Field(
        default=None,
        description="Directory to save the model to, or unset to prompt the user.",
        exclude=True,
    )

    upload_repo_id: str | None = Field(
        default=None,
        description="Name of the Hugging Face repository to upload the model to, or unset to prompt the user.",
        exclude=True,
    )

    upload_repo_private: bool | None = Field(
        default=None,
        description="Whether the Hugging Face repository to upload the model to should be private, or unset to prompt the user.",
    )

    upload_reproducibility_information: str | None = Field(
        default=None,
        description='Which reproducibility information to add to the Hugging Face repository: "full", "basic", "none", or unset to prompt the user.',
    )

    upload_tags: list[str] = Field(
        default=[
            "uncensored",
            "decensored",
            "abliterated",
        ],
        description="Tags to add to the Hugging Face repository when uploading the model.",
    )

    ignore_mismatches: bool | None = Field(
        default=None,
        description="Whether to attempt to reproduce the model even if there are environment mismatches, or unset to prompt the user.",
    )

    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt to use when prompting the model.",
    )

    # We intentionally allow extra keys so users can provide plugin-specific
    # configuration in TOML tables like `[scorer.KeywordRate]` which are later
    # consumed via `settings.model_extra` (see `plugin.get_plugin_namespace`).
    model_config = SettingsConfigDict(extra="allow")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,  # Values passed to the constructor override all other sources.
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                cli_implicit_flags=True,
                cli_kebab_case=True,
            ),
            EnvSettingsSource(settings_cls, env_prefix="HERETIC_"),
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls, toml_file="config.toml"),
        )

    @classmethod
    def model_validate_init_only(
        cls, values: dict[str, Any], current: "Settings"
    ) -> "Settings":
        """
        Validates `values` without reading any other settings source: nothing
        comes from config.toml, environment variables or the command line, and
        settings missing from `values` get their defaults. Settings with
        `exclude=True` (such as `save_directory`) are left out of serialized
        settings, so they are taken from `current`.
        """
        excluded = {
            name: getattr(current, name)
            for name, field in cls.model_fields.items()
            if field.exclude
        }
        return InitOnlySettings.model_validate(values | excluded)


class InitOnlySettings(Settings):
    """Settings validated only from the values passed in, without reading any source."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)
