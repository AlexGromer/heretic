# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests of the v7 additions: sequence KL over an answer, benchmarks on the ranked
# candidates, directions per language / non-answer subtype / reasoning block, per-layer
# router bias, and router and expert discovery for Gemma 4's layout.
# CPU only; the model tests use the tiny MoE model of test_fused_experts.
# Prompts and answers are neutral placeholders: the tests check plumbing, not content.

import io
import json
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from heretic.benchmarks import (
    BENCHMARK_SCORES_ATTR,
    benchmark_trials,
    format_benchmarks,
    parse_task,
)
from heretic.config import Settings
from heretic.modifiers.abliteration import Abliteration, Parameters
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.plugin import Context
from heretic.positions import LabelledAnswer, load_labelled_answers, thinking_positions
from heretic.router_bias import find_router
from heretic.scorers.sequence_kl import SequenceKLDivergence
from heretic.scorers.sequence_kl import Settings as SequenceKLSettings
from heretic.utils import detect_lang


def tiny_model():
    # Imported here: loading the tiny model takes a few seconds.
    from test_fused_experts import get_model

    model = get_model()
    model.reset_model()
    return model


class LanguageTests(unittest.TestCase):
    def test_detect_lang_matches_refusal_eval(self) -> None:
        samples = [
            "Explain the steps",
            "Объясни шаги подробно",
            "Run x и покажи вывод",
            "",
            "12345",
        ]
        self.assertEqual(
            [detect_lang(s) for s in samples], ["en", "ru", "ru", "en", "en"]
        )


class SequenceKlTests(unittest.TestCase):
    def test_zero_on_the_unmodified_model_and_positive_after_a_change(self) -> None:
        model = tiny_model()
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory, "prompts.txt")
            prompts.write_text(
                "tell me about rain\nexplain the tide\n", encoding="utf-8"
            )
            scorer = SequenceKLDivergence(
                heretic_settings=model.settings,
                settings=SequenceKLSettings(
                    prompts={"dataset": str(prompts)},
                    tokens=4,
                    top_k=16,
                ),
            )
            ctx = Context(settings=model.settings, model=model)
            with redirect_stdout(io.StringIO()):
                scorer.init(ctx)
                baseline = scorer.get_score(ctx)
                self.assertEqual(scorer.get_baseline_score(ctx).value, 0)

                # Every compared position belongs to a continuation of the recorded length.
                self.assertEqual(len(scorer._items), 2)
                self.assertTrue(float(scorer._valid.sum()) > 0)

                with torch.no_grad():
                    for module in model.get_layer_modules(1)["attn.o_proj"]:
                        weight = getattr(module, "base_layer", module).weight
                        weight += torch.randn_like(weight) * 0.05
                changed = scorer.get_score(ctx)

        self.assertLess(baseline.value, 1e-5)
        self.assertGreater(changed.value, baseline.value)
        self.assertIn("worst prompt", changed.rich_display)
        model.needs_reload = True
        model.reset_model()


class BenchmarkTests(unittest.TestCase):
    def test_parse_and_format(self) -> None:
        self.assertEqual(parse_task("gsm8k"), ("gsm8k", "acc,none"))
        self.assertEqual(
            parse_task("gsm8k:exact_match,strict-match"),
            ("gsm8k", "exact_match,strict-match"),
        )
        self.assertEqual(
            format_benchmarks({"mmlu": 0.5, "gsm8k": 0.25}), "gsm8k 0.250, mmlu 0.500"
        )

    def test_runs_once_per_candidate_and_stores(self) -> None:
        trials = [
            SimpleNamespace(
                number=number,
                user_attrs={
                    "index": number + 1,
                    "parameters": {"p": number},
                    "scores": [{"name": "KL divergence", "score": {"value": kl}}],
                },
            )
            for number, kl in ((0, 0.1), (1, 0.5))
        ]
        evaluated: list[Any] = []

        class FakeStudy:
            def __init__(self) -> None:
                self.user_attrs: dict[str, Any] = {}

            def set_user_attr(self, key: str, value: Any) -> None:
                self.user_attrs[key] = json.loads(json.dumps(value))

        class FakeModifier:
            parameters_class = SimpleNamespace(from_dict=lambda data: data)

            def reset_model(self, ctx: Any) -> None:
                evaluated.append("reset")

            def modify_model(self, ctx: Any, parameters: Any) -> None:
                evaluated.append(f"modify {parameters['p']}")

        settings = Settings.model_construct(
            judge_ranking_benchmarks=["piqa:acc_norm,none"],
            n_judge_ranking_trials=2,
            judge_ranking_max_kl=0.15,
            judge_ranking_merged=False,
        )
        study = FakeStudy()
        with (
            unittest.mock.patch(
                "heretic.benchmarks.run_tasks", lambda model, tasks: {"piqa": 0.5}
            ),
            redirect_stdout(io.StringIO()),
        ):
            stored = benchmark_trials(
                cast(Any, study),
                cast(Any, trials),
                settings,
                cast(Any, None),
                cast(Any, FakeModifier()),
            )
            again = benchmark_trials(
                cast(Any, study),
                cast(Any, trials),
                settings,
                cast(Any, None),
                cast(Any, FakeModifier()),
            )

        # Trial 1 is above the KL limit, so only the baseline and trial 0 are benchmarked.
        self.assertEqual(sorted(stored), ["0", "baseline"])
        self.assertEqual(stored["0"], {"piqa": 0.5})
        self.assertIn(BENCHMARK_SCORES_ATTR, study.user_attrs)
        self.assertEqual(again, stored)
        self.assertIn("modify 0", evaluated)

    def test_disabled_without_tasks(self) -> None:
        settings = Settings.model_construct(
            judge_ranking_benchmarks=[], n_judge_ranking_trials=2
        )
        self.assertEqual(
            benchmark_trials(
                cast(Any, SimpleNamespace(user_attrs={})),
                [],
                settings,
                cast(Any, None),
                cast(Any, None),
            ),
            {},
        )


class ThinkingPositionTests(unittest.TestCase):
    def test_thinking_tokens_are_found_and_missing_blocks_are_skipped(self) -> None:
        model = tiny_model()
        answer = LabelledAnswer(
            "s",
            "a question",
            "<think>let me weigh this carefully</think>The answer.",
            "PARTIAL",
        )
        item = thinking_positions(model, answer, 4)
        assert item is not None
        decoded = model.tokenizer.decode(
            [item.input_ids[position] for position in item.positions]
        )
        self.assertIn("let", decoded)
        self.assertNotIn("The answer", decoded)
        plain = LabelledAnswer("s", "a question", "The answer.", "COMPLY")
        self.assertIsNone(thinking_positions(model, plain, 4))

    def test_reasons_are_loaded_from_both_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            own = Path(directory, "own.jsonl")
            own.write_text(
                json.dumps(
                    {
                        "prompt": "p",
                        "response": "r",
                        "judge_label": "PARTIAL",
                        "judge_reason": "only a general overview",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_labelled_answers(str(own), None, "d")[0].reason,
                "only a general overview",
            )

            responses = Path(directory, "responses.jsonl")
            judged = Path(directory, "judged.jsonl")
            base = {
                "model": "m",
                "trial": "1",
                "mode": "greedy",
                "thinking": False,
                "set": "s",
                "prompt_id": "p1",
            }
            responses.write_text(
                json.dumps({**base, "prompt": "p", "response": "r"}) + "\n",
                encoding="utf-8",
            )
            judged.write_text(
                json.dumps(
                    {**base, "label": "PARTIAL", "reason": "asks for clarification"}
                )
                + "\n",
                encoding="utf-8",
            )
            answer = load_labelled_answers(str(responses), str(judged), "d")[0]
        self.assertEqual(
            (answer.label, answer.reason), ("PARTIAL", "asks for clarification")
        )


def write_answers(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt": prompt,
                    "response": response,
                    "judge_label": label,
                    "judge_reason": reason,
                }
            )
            + "\n"
            for prompt, response, label, reason in rows
        ),
        encoding="utf-8",
    )


class DirectionSlotTests(unittest.TestCase):
    def build(self, directory: str, **settings: Any) -> Abliteration:
        model = tiny_model()
        good = Path(directory, "good.txt")
        bad = Path(directory, "bad.txt")
        good.write_text(
            "".join(f"good prompt {i}\n" for i in range(12))
            + "".join(f"хороший запрос {i}\n" for i in range(12)),
            encoding="utf-8",
        )
        bad.write_text(
            "".join(f"bad prompt {i}\n" for i in range(12))
            + "".join(f"плохой запрос {i}\n" for i in range(12)),
            encoding="utf-8",
        )
        modifier = Abliteration.__new__(Abliteration)
        modifier.heretic_settings = model.settings
        modifier.settings = AbliterationSettings(
            good_prompts={"dataset": str(good)},
            bad_prompts={"dataset": str(bad)},
            **settings,
        )
        ctx = Context(settings=model.settings, model=model)
        with (
            unittest.mock.patch.object(model, "apply_lora", lambda rank: None),
            redirect_stdout(io.StringIO()),
        ):
            modifier.init(ctx)
        return modifier

    def test_language_subtype_and_thinking_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            answers = Path(directory, "answers.jsonl")
            rows = []
            for i in range(12):
                rows.append(
                    (
                        f"q{i}",
                        f"<think>weighing it {i}</think>Only a general overview here.",
                        "PARTIAL",
                        "gives only a general overview",
                    )
                )
            for i in range(12):
                rows.append(
                    (
                        f"c{i}",
                        f"<think>fine {i}</think>Step 1: open it. Step 2: edit line {i}.",
                        "COMPLY",
                        "does the task",
                    )
                )
            write_answers(answers, rows)
            modifier = self.build(
                directory,
                language_directions=True,
                language_min_prompts=10,
                response_direction_file=str(answers),
                response_subtype_patterns={"overview": "general|overview"},
                response_subtype_min_answers=5,
                response_thinking_direction=True,
            )

        self.assertEqual(
            modifier.extra_direction_names,
            (
                "language_en_direction",
                "language_ru_direction",
                "partial_direction",
                "overview_direction",
                "thinking_direction",
            ),
        )
        extras = modifier.extra_directions
        assert extras is not None
        main = modifier.residual_directions
        for row in range(1, main.shape[0]):
            vectors = [main[row]] + [d for d in extras[row] if bool(d.norm() > 0)]
            stacked = torch.stack(vectors)
            gram = stacked @ stacked.T
            self.assertTrue(torch.allclose(gram, torch.eye(len(vectors)), atol=1e-4))
        # One LoRA rank per direction, on top of the full-normalization rank.
        self.assertEqual(modifier.lora_rank, 3 + len(modifier.extra_direction_names))
        tiny_model()

    def test_small_groups_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            modifier = self.build(
                directory, language_directions=True, language_min_prompts=50
            )
        self.assertEqual(modifier.extra_direction_names, ())
        self.assertIsNone(modifier.extra_directions)
        tiny_model()


class PerLayerRouterTests(unittest.TestCase):
    def test_parameters_carry_one_alpha_per_layer(self) -> None:
        parameters = cast(
            Parameters,
            Parameters.from_dict(
                {
                    "direction_index": None,
                    "weight_distributions": {},
                    "router_bias_alphas": {"3": 0.5, "11": 0.25},
                }
            ),
        )
        self.assertEqual(parameters.router_bias_alphas, {"3": 0.5, "11": 0.25})
        shown = parameters.to_presentation_dict()
        self.assertEqual(shown["router_bias_alpha.3"], "0.50")
        self.assertEqual(shown["router_bias_alpha.11"], "0.25")
        self.assertNotIn("router_bias_alpha", shown)
        # Trials stored before this setting existed still load.
        old = cast(
            Parameters,
            Parameters.from_dict(
                {
                    "direction_index": None,
                    "weight_distributions": {},
                    "router_bias_alpha": 0.5,
                }
            ),
        )
        self.assertEqual(old.router_bias_alphas, {})
        self.assertEqual(old.to_presentation_dict()["router_bias_alpha"], "0.50")

    def test_per_layer_alphas_are_sampled_and_applied(self) -> None:
        import optuna

        model = tiny_model()
        from heretic.router_bias import build_router_bias_plans
        from heretic.utils import Prompt

        good = [Prompt(system="s", user=u) for u in ("a", "bb bb", "ccc")]
        bad = [Prompt(system="s", user=u) for u in ("d d", "eee", "ff ff")]
        with redirect_stdout(io.StringIO()):
            plans = build_router_bias_plans(model, good, bad, 4, 4.0)

        modifier = Abliteration.__new__(Abliteration)
        modifier.heretic_settings = model.settings
        modifier.settings = AbliterationSettings(router_bias_per_layer=True)
        modifier.router_plans = tuple(plans)
        modifier.residual_directions = torch.zeros(
            len(model.get_layers()) + 1,
            model.model.config.get_text_config().hidden_size,
        )
        modifier.ega_plans = {}

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        trial = study.ask()
        ctx = cast(Context, SimpleNamespace(get_model=lambda: model))
        parameters = modifier.suggest_parameters(ctx, trial)

        self.assertEqual(
            sorted(parameters.router_bias_alphas, key=int),
            [str(plan.router.layer_index) for plan in plans],
        )
        self.assertEqual(parameters.router_bias_alpha, 0.0)
        self.assertIn(f"router_bias_alpha.{plans[0].router.layer_index}", trial.params)

        # Only the layers with a non-zero alpha are edited.
        alphas = {str(plans[0].router.layer_index): 0.5}
        parameters.router_bias_alphas = {
            key: alphas.get(key, 0.0) for key in parameters.router_bias_alphas
        }
        originals = [plan.router.weight.detach().clone() for plan in plans]
        from heretic.router_bias import apply_router_bias

        for plan in plans:
            apply_router_bias(
                model,
                plan,
                parameters.router_bias_alphas[str(plan.router.layer_index)],
            )
        self.assertFalse(torch.equal(plans[0].router.weight, originals[0]))
        for plan, original in list(zip(plans, originals))[1:]:
            self.assertTrue(torch.equal(plan.router.weight, original))
        model.reset_model()


class GemmaLayoutTests(unittest.TestCase):
    """Gemma 4 text keeps the experts and the router on the decoder layer, and the router
    holds its projection in a child module (`router.proj`)."""

    def make_model(
        self, experts: int = 4, hidden: int = 8, intermediate: int = 6
    ) -> Any:
        class Router(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(hidden, experts, bias=False)
                self.scale = nn.Parameter(torch.ones(hidden))

        class Experts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_up_proj = nn.Parameter(
                    torch.randn(experts, 2 * intermediate, hidden)
                )
                self.down_proj = nn.Parameter(
                    torch.randn(experts, hidden, intermediate)
                )
                self.act_fn = torch.nn.functional.silu

        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = Router()
                self.experts = Experts()
                self.mlp = nn.Linear(hidden, hidden)

        layers = nn.ModuleList([Layer()])
        config = SimpleNamespace(
            hidden_size=hidden,
            top_k_experts=2,
            num_experts=experts,
            get_text_config=lambda: config,
        )
        return SimpleNamespace(layers=layers, config=config)

    def test_router_and_experts_are_found(self) -> None:
        from heretic.model import Model

        fake = self.make_model()
        model = cast(Any, Model.__new__(Model))
        model.model = SimpleNamespace(config=fake.config, model=fake)
        model._warnings_shown = set()
        model.settings = Settings.model_construct()

        router = find_router(model, 0)
        assert router is not None
        # The weight lives in the projection child, which is what the inputs multiply.
        self.assertEqual(router.path, "router.proj")
        self.assertEqual((router.num_experts, router.top_k), (4, 2))
        self.assertIs(router.module, fake.layers[0].router.proj)

        fused = model.get_layer_fused_experts(0)
        self.assertEqual(list(fused), ["mlp.experts.down_proj"])
        handles = fused["mlp.experts.down_proj"]
        # One entry, found on the layer itself, with the hidden dimension on axis 1.
        self.assertEqual(len(handles), 1)
        self.assertEqual(handles[0].path, "experts.down_proj")
        self.assertEqual(handles[0].row_axis, 1)

        from heretic.ega import supports_ega

        self.assertTrue(supports_ega(handles[0]))

    def test_quantized_router_is_skipped(self) -> None:
        fake = self.make_model()
        fake.layers[0].router.proj.weight = nn.Parameter(
            torch.zeros(4, 8, dtype=torch.uint8), requires_grad=False
        )
        from heretic.model import Model

        model = cast(Any, Model.__new__(Model))
        model.model = SimpleNamespace(config=fake.config, model=fake)
        self.assertIsNone(find_router(model, 0))


if __name__ == "__main__":
    unittest.main()
