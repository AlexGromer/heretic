# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests of the v5 additions: re-scoring on merged models (#490), multi-directional
# abliteration with self-organizing maps (#140), router pre-bias (#322) and the local
# judge. CPU only; the model tests use the tiny MoE model of test_fused_experts.

import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as F
from peft import PeftModel

from heretic.config import Settings
from heretic.directions import (
    compute_extra_directions,
    interpolate_extras,
    neuron_hits,
    orthonormal_extension,
    train_som,
)
from heretic.judge import JudgeClient, compare
from heretic.judge import main as judge_main
from heretic.merged_scoring import (
    MERGED_SCORES_ATTR,
    rank_by_merged_scores,
    rescore_on_merged_models,
)
from heretic.modifier import Context
from heretic.modifiers.abliteration import (
    Abliteration,
    Parameters,
    RowNormalization,
    WeightDistribution,
)
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.router_bias import (
    apply_router_bias,
    build_router_bias_plans,
    find_router,
    log_ratio,
)
from heretic.scorers.judge_rate import JudgeRate
from heretic.scorers.judge_rate import Settings as JudgeRateSettings
from heretic.utils import Prompt

if sys.version_info >= (3, 11):
    import tomllib

GOOD = [Prompt(system="s", user=u) for u in ("a", "bb bb", "ccc ccc ccc", "dd")]
BAD = [Prompt(system="s", user=u) for u in ("e e", "fff", "gg gg gg gg", "h h h")]


def tiny_model():
    # Imported here: loading the tiny model takes a few seconds.
    from test_fused_experts import get_model

    model = get_model()
    if not isinstance(model.model, PeftModel):
        model.apply_lora(3)
    return model


def scores(refusals: float, kl: float) -> list[dict[str, Any]]:
    return [
        {
            "name": "Refusals",
            "score": {"value": refusals, "rich_display": f"{refusals}"},
            "baseline": {"value": 1.0},
        },
        {
            "name": "KL divergence",
            "score": {"value": kl, "rich_display": f"{kl}"},
            "baseline": {"value": 0.0},
        },
    ]


class FakeStudy:
    def __init__(self) -> None:
        self.user_attrs: dict[str, Any] = {}

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = json.loads(json.dumps(value))


class MergedScoringTests(unittest.TestCase):
    def trials(self) -> list[Any]:
        return [
            SimpleNamespace(
                number=number,
                user_attrs={
                    "index": number + 1,
                    "parameters": {"p": number},
                    "scores": scores(refusals, kl),
                },
            )
            for number, (refusals, kl) in enumerate([(0.03, 0.15), (0.05, 0.1)])
        ]

    def test_ranking_needs_every_trial(self) -> None:
        trials = self.trials()
        merged = {"0": scores(0.06, 0.2), "1": scores(0.04, 0.1)}
        ranked = rank_by_merged_scores(trials, merged, ["Refusals", "KL divergence"])
        self.assertEqual([t.number for t in ranked], [1, 0])
        partial = rank_by_merged_scores(trials, {"0": merged["0"]}, ["Refusals"])
        self.assertEqual([t.number for t in partial], [0, 1])

    def run_rescore(self, study: FakeStudy, fail: bool = False) -> list[str]:
        calls: list[str] = []
        live = object()
        merged_model = object()

        class FakeModel:
            model: Any = live

            def get_merged_model(self) -> Any:
                calls.append("merge")
                if fail:
                    raise RuntimeError("out of memory")
                return merged_model

        model = FakeModel()

        class FakeEvaluator:
            def get_scores(self) -> Any:
                assert model.model is merged_model
                calls.append("score")
                return "scores"

            def get_paired_score_records(self, _: Any) -> Any:
                return scores(0.07, 0.3)

        class FakeModifier:
            parameters_class = SimpleNamespace(from_dict=lambda data: data)

            def reset_model(self, ctx: Any) -> None:
                calls.append("reset")

            def modify_model(self, ctx: Any, parameters: Any) -> None:
                calls.append(f"modify {parameters['p']}")

        settings = cast(
            Settings,
            SimpleNamespace(n_merged_score_trials=1, quantization="none"),
        )
        with redirect_stdout(io.StringIO()):
            rescore_on_merged_models(
                cast(Any, study),
                self.trials(),
                settings,
                cast(Any, model),
                cast(Any, FakeEvaluator()),
                cast(Any, FakeModifier()),
            )
        self.assertIs(model.model, live)
        return calls

    def test_rescores_once_and_caches(self) -> None:
        study = FakeStudy()
        self.assertEqual(
            self.run_rescore(study), ["reset", "modify 0", "merge", "score"]
        )
        self.assertEqual(
            study.user_attrs[MERGED_SCORES_ATTR]["0"][0]["score"]["value"], 0.07
        )
        # Shown again (e.g. after "Run additional trials"): nothing is recomputed.
        self.assertEqual(self.run_rescore(study), [])

    def test_failure_keeps_live_scores(self) -> None:
        study = FakeStudy()
        self.assertEqual(
            self.run_rescore(study, fail=True), ["reset", "modify 0", "merge"]
        )
        self.assertNotIn(MERGED_SCORES_ATTR, study.user_attrs)

    def test_setting(self) -> None:
        self.assertEqual(Settings.model_fields["n_merged_score_trials"].default, 0)


class SomTests(unittest.TestCase):
    def clusters(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(0)
        centers = torch.tensor([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
        return torch.cat(
            [
                center + 0.1 * torch.randn(50, 3, generator=generator)
                for center in centers
            ]
        )

    def test_som_finds_the_clusters(self) -> None:
        data = self.clusters()
        prototypes = train_som(data, (3, 3), epochs=30, sigma_end=0.3)
        self.assertTrue(torch.equal(prototypes, train_som(data, (3, 3), 30, 0.3)))
        hits = neuron_hits(prototypes, data)
        self.assertEqual(int(hits.sum()), 150)
        # Every cluster center has a prototype close to it.
        for center in torch.eye(3) * 10:
            distance = (prototypes - center).norm(dim=1).min()
            self.assertLess(float(distance), 1.0)

    def test_orthonormal_extension(self) -> None:
        basis = torch.tensor([[1.0, 0, 0, 0]])
        candidates = torch.tensor(
            [[2.0, 0, 0, 0], [1.0, 1, 0, 0], [0, 3.0, 0, 0], [0, 0, 0, 5.0]]
        )
        result = orthonormal_extension(basis, candidates, 3)
        # The first candidate is the basis itself, the third duplicates the second.
        expected = torch.tensor([[0, 1.0, 0, 0], [0, 0, 0, 1.0], [0, 0, 0, 0]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_extra_directions_are_orthonormal(self) -> None:
        torch.manual_seed(0)
        hidden = 8
        bad = torch.randn(60, 2, hidden) + torch.tensor([3.0] + [0] * (hidden - 1))
        bad[:30, :, 1] += 4
        good_means = torch.randn(2, hidden)
        main = F.normalize(bad.mean(0) - good_means, dim=1)
        good_direction = F.normalize(good_means, dim=1)
        main = F.normalize(
            main - (main * good_direction).sum(1, keepdim=True) * good_direction, dim=1
        )
        extras = compute_extra_directions(
            bad, good_means, main, 3, (3, 3), 20, 0.5, orthogonalize_good=True
        )
        self.assertEqual(tuple(extras.shape), (2, 3, hidden))
        for row in range(2):
            present = extras[row][extras[row].norm(dim=1) > 0]
            self.assertGreater(present.shape[0], 0)
            vectors = torch.cat([main[row][None], good_direction[row][None], present])
            gram = vectors @ vectors.T
            self.assertTrue(torch.allclose(gram, torch.eye(len(vectors)), atol=1e-4))

        interpolated = interpolate_extras(extras, 0.2, main[1])
        self.assertTrue(
            torch.allclose(interpolated @ main[1], torch.zeros(3), atol=1e-5)
        )


def lora_delta(module: Any) -> torch.Tensor:
    return module.get_delta_weight("default").to(torch.float32)


class MultiDirectionTests(unittest.TestCase):
    def make_modifier(self, model: Any, extras: torch.Tensor | None) -> Abliteration:
        modifier = Abliteration.__new__(Abliteration)
        modifier.heretic_settings = model.settings
        modifier.settings = AbliterationSettings(
            row_normalization=RowNormalization.NONE,
            direction_count=1 if extras is None else extras.shape[1] + 1,
        )
        hidden = model.model.config.get_text_config().hidden_size
        rows = len(model.get_layers()) + 1
        generator = torch.Generator().manual_seed(0)
        modifier.residual_directions = F.normalize(
            torch.randn(rows, hidden, generator=generator), dim=1
        )
        modifier.extra_directions = extras
        modifier.ega_plans = {}
        modifier.router_plans = ()
        return modifier

    def parameters(self, model: Any, weights: list[float]) -> Parameters:
        return Parameters(
            direction_index=None,
            weight_distributions={
                component: WeightDistribution(1.0, 0.0, 1.0, 100.0)
                for component in model.get_abliterable_components()
            },
            direction_weights=weights,
        )

    def test_projection_onto_several_directions(self) -> None:
        model = tiny_model()
        hidden = model.model.config.get_text_config().hidden_size
        rows = len(model.get_layers()) + 1
        single = self.make_modifier(model, None)
        # Extra directions orthonormal to the main direction of each row.
        extras = torch.zeros(rows, 2, hidden)
        for row in range(rows):
            extras[row] = orthonormal_extension(
                single.residual_directions[row][None],
                torch.randn(4, hidden, generator=torch.Generator().manual_seed(row)),
                2,
            )
        modifier = self.make_modifier(model, extras)
        ctx = cast(Context, SimpleNamespace(get_model=lambda: model))
        weights = [0.5, 0.25]

        model.reset_model()
        with redirect_stdout(io.StringIO()):
            modifier.modify_model(ctx, self.parameters(model, weights))

        layer_index = 1
        module = model.get_layer_modules(layer_index)["attn.o_proj"][0]
        W = cast(Any, module).base_layer.weight.to(torch.float32)
        V = torch.cat(
            [single.residual_directions[layer_index + 1][None], extras[layer_index + 1]]
        )
        w = torch.tensor([1.0] + weights)
        expected = -(V.T * w) @ V @ W
        self.assertTrue(torch.allclose(lora_delta(module), expected, atol=1e-5))

        # With all extra weights 0, the result is classic abliteration.
        model.reset_model()
        with redirect_stdout(io.StringIO()):
            modifier.modify_model(ctx, self.parameters(model, [0.0, 0.0]))
        multi_zero = lora_delta(module).clone()
        model.reset_model()
        with redirect_stdout(io.StringIO()):
            single.modify_model(ctx, self.parameters(model, []))
        self.assertTrue(torch.allclose(multi_zero, lora_delta(module), atol=1e-6))
        model.reset_model()

    def test_old_parameters_still_load(self) -> None:
        parameters = cast(
            Parameters,
            Parameters.from_dict(
                {
                    "direction_index": 1.5,
                    "weight_distributions": {
                        "attn.o_proj": {
                            "max_weight": 1.0,
                            "max_weight_position": 2.0,
                            "min_weight": 0.5,
                            "min_weight_distance": 3.0,
                        }
                    },
                }
            ),
        )
        self.assertEqual(parameters.direction_weights, [])
        self.assertEqual(parameters.router_bias_alpha, 0.0)

    def test_settings_validation(self) -> None:
        with self.assertRaises(ValueError):
            AbliterationSettings(direction_count=0)
        with self.assertRaises(ValueError):
            AbliterationSettings(direction_weight_range=(-0.1, 1.0))
        with self.assertRaises(ValueError):
            AbliterationSettings(router_bias_max_log_ratio=0)


class RouterBiasTests(unittest.TestCase):
    def test_log_ratio(self) -> None:
        ratio = log_ratio(
            torch.tensor([10, 0, 5]), 10, torch.tensor([0, 10, 5]), 10, max_abs=2.0
        )
        self.assertAlmostEqual(float(ratio[0]), 2.0)  # clipped
        self.assertAlmostEqual(float(ratio[1]), -2.0)
        self.assertAlmostEqual(float(ratio[2]), 0.0)

    def test_router_bias_on_tiny_moe(self) -> None:
        model = tiny_model()
        model.reset_model()
        router = find_router(model, 0)
        assert router is not None
        self.assertEqual(router.path, "mlp.gate")
        self.assertEqual(router.num_experts, router.weight.shape[0])

        with redirect_stdout(io.StringIO()):
            plans = build_router_bias_plans(model, GOOD, BAD, 4, 4.0)
        self.assertEqual(len(plans), len(model.get_layers()))
        plan = plans[0]
        self.assertAlmostEqual(float(plan.direction.norm()), 1.0, places=5)

        original = router.weight.detach().clone()
        apply_router_bias(model, plan, 0.5)
        changed = router.weight.detach().clone()
        self.assertFalse(torch.equal(original, changed))

        # For an input along the mean direction with the mean projection, the
        # logits change by exactly the bias.
        h = plan.direction * plan.mean_projection
        delta = (changed.to(torch.float32) - original.to(torch.float32)) @ h
        expected = -0.5 * plan.log_ratio
        self.assertTrue(torch.allclose(delta, expected, atol=1e-3))

        # Bit-exact restore between trials, and alpha 0 changes nothing.
        model.reset_model()
        self.assertTrue(torch.equal(router.weight, original))
        apply_router_bias(model, plan, 0.0)
        self.assertTrue(torch.equal(router.weight, original))
        self.assertFalse(model.has_modified_fused_experts())


class FakeJudge:
    """Ollama /api/chat stand-in: labels by the first matching text, COMPLY otherwise."""

    def __init__(self, labels: dict[str, str]) -> None:
        self.labels = labels
        self.requests = 0

    def __call__(self, url: str, payload: Any, headers: Any, timeout: float) -> Any:
        self.requests += 1
        content = payload["messages"][1]["content"]
        label = next(
            (label for text, label in self.labels.items() if text in content), "COMPLY"
        )
        return {"message": {"content": json.dumps({"label": label, "reason": "r"})}}


class JudgeTests(unittest.TestCase):
    def test_client_caches_and_skips_empty(self) -> None:
        fake = FakeJudge({"Nope": "REFUSAL"})
        client = JudgeClient(post=fake)
        self.assertEqual(client.classify("p", "Nope."), "REFUSAL")
        self.assertEqual(client.classify("p", "Nope."), "REFUSAL")
        self.assertEqual(client.classify("p", "  "), "OFF_TOPIC")
        self.assertEqual(fake.requests, 1)

    def test_compare(self) -> None:
        records = [
            {
                "score_name": "R",
                "evaluation": 1,
                "match": True,
                "judge_label": "COMPLY",
            },
            {
                "score_name": "R",
                "evaluation": 1,
                "match": True,
                "judge_label": "REFUSAL",
            },
            {
                "score_name": "R",
                "evaluation": 1,
                "match": False,
                "judge_label": "PARTIAL",
            },
        ]
        comparison = compare(records, ["REFUSAL", "PARTIAL"])
        self.assertEqual(comparison.keyword[("R", 1)], 2)
        self.assertEqual(comparison.judge[("R", 1)], 2)
        self.assertEqual(comparison.agreement[(True, False)], 1)
        self.assertEqual(comparison.agreement[(False, True)], 1)

    def test_scorer(self) -> None:
        scorer = JudgeRate(
            heretic_settings=Settings.model_construct(),
            settings=JudgeRateSettings(),
        )
        scorer.prompts = [Prompt(system="s", user=u) for u in ("p1", "p2", "p3")]
        scorer.client = JudgeClient(
            post=FakeJudge({"cannot": "REFUSAL", "Hmm": "PARTIAL"})
        )
        responses = [
            "<think>I cannot decide</think>Sure, here it is.",
            "I cannot do that.",
            "Hmm, what do you mean?",
        ]
        ctx = cast(Context, SimpleNamespace(get_responses=lambda prompts: responses))
        score = scorer.get_score(ctx)
        # The reasoning is removed before judging, so the first answer complies.
        self.assertAlmostEqual(score.value, 2 / 3)
        self.assertFalse(scorer.reproducible)

    def test_cli_against_a_local_server(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                content = body["messages"][1]["content"]
                label = "REFUSAL" if "nope" in content else "COMPLY"
                reply = json.dumps(
                    {
                        "message": {
                            "content": json.dumps({"label": label, "reason": "r"})
                        }
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(reply.encode())

            def log_message(self, format: str, *args: Any) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                responses = Path(directory, "responses.jsonl")
                output = Path(directory, "labelled.jsonl")
                rows = [(0, "nope", True), (1, "here it is", True), (1, "nope", True)]
                responses.write_text(
                    "".join(
                        json.dumps(
                            {
                                "evaluation": evaluation,
                                "score_name": "Refusals",
                                "prompt": "p",
                                "response": response,
                                "match": match,
                            }
                        )
                        + "\n"
                        for evaluation, response, match in rows
                    ),
                    encoding="utf-8",
                )
                endpoint = f"http://127.0.0.1:{server.server_address[1]}"
                with redirect_stdout(io.StringIO()):
                    judge_main(
                        [
                            str(responses),
                            "--endpoint",
                            endpoint,
                            "--output",
                            str(output),
                            "--evaluation",
                            "1",
                        ]
                    )
                labelled = [
                    json.loads(line)
                    for line in output.read_text(encoding="utf-8").splitlines()
                ]
        finally:
            server.shutdown()

        self.assertEqual([r["judge_label"] for r in labelled], ["COMPLY", "REFUSAL"])
        self.assertEqual(labelled[0]["judge_prompt"], "v1-039fd0fa")

    @unittest.skipIf(sys.version_info < (3, 11), "tomllib needs Python 3.11")
    def test_default_config_section(self) -> None:
        root = Path(__file__).resolve().parent.parent
        with open(root / "config.default.toml", "rb") as file:
            config = tomllib.load(file)
        documented = JudgeRate.validate_settings(
            JudgeRate.get_settings_raw(config, "scorer", None)
        )
        self.assertEqual(documented, JudgeRateSettings())


if __name__ == "__main__":
    unittest.main()
