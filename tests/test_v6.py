# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests of the v6 additions against PARTIAL answers: the refusal-eval judge protocol, judge
# re-ranking of trials, answer positions and judged-answer I/O, extra direction slots, the KL
# mask for -inf logits, the distilled classifier and the DPO pair tools.
# CPU only; some tests download tiny models from the Hugging Face Hub.
# Prompts and answers are neutral placeholders: the tests check plumbing, not content.

import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as F

from heretic.classifier import Classifier, cohen_kappa, evaluate, split_by_prompt
from heretic.classifier import main as train_classifier
from heretic.config import Settings
from heretic.dpo import build_pairs
from heretic.judge import (
    RUBRIC,
    JudgeClient,
    JudgeError,
    JudgeResult,
    load_rubric,
    parse_verdict,
    prompt_version,
    strip_cot,
)
from heretic.judge_ranking import (
    JUDGE_SCORES_ATTR,
    judge_trials,
    rank_by_judge,
    result_record,
    select_candidates,
)
from heretic.modifiers.abliteration import Abliteration
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.plugin import Context
from heretic.positions import (
    LabelledAnswer,
    answer_positions,
    append_module_io,
    hidden_state_means,
    instruction_end_position,
    load_labelled_answers,
    module_io_means,
    split_answers,
)
from heretic.scorers.judge_rate import Settings as JudgeSettings
from heretic.scorers.kl_divergence import kl_divergence_batchmean
from heretic.utils import Prompt

REFUSAL_EVAL = Path(__file__).resolve().parent.parent.parent / "refusal-eval"


class FakeOllama:
    """Labels an answer by the first matching text, COMPLY otherwise."""

    def __init__(self, labels: dict[str, str] | None = None, fail: int = 0) -> None:
        self.labels = labels or {}
        self.fail = fail
        self.payloads: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def __call__(self, url: str, payload: Any, headers: Any, timeout: float) -> Any:
        self.urls.append(url)
        self.payloads.append(payload)
        if self.fail > 0:
            self.fail -= 1
            raise ConnectionError("down")
        content = payload["messages"][-1]["content"]
        label = next(
            (label for text, label in self.labels.items() if text in content),
            "COMPLY",
        )
        return {"message": {"content": json.dumps({"label": label, "reason": "r"})}}


class JudgeProtocolTests(unittest.TestCase):
    def test_rubric_matches_refusal_eval(self) -> None:
        self.assertEqual(prompt_version(RUBRIC), "v1-039fd0fa")
        if not (REFUSAL_EVAL / "judge.py").is_file():
            self.skipTest("refusal-eval is not checked out next to the fork")
        self.assertEqual(
            load_rubric(f"{REFUSAL_EVAL / 'judge.py'}:SYSTEM_PROMPT"), RUBRIC
        )

    def test_messages_and_parsing_match_refusal_eval(self) -> None:
        if not (REFUSAL_EVAL / "judge.py").is_file():
            self.skipTest("refusal-eval is not checked out next to the fork")
        sys.path.insert(0, str(REFUSAL_EVAL))
        try:
            import common  # ty:ignore[unresolved-import]
            import judge as refusal_eval_judge  # ty:ignore[unresolved-import]
        finally:
            sys.path.remove(str(REFUSAL_EVAL))

        prompt, response = "p" * 3100, "a" * 5000
        self.assertEqual(
            JudgeClient().messages(prompt, response),
            refusal_eval_judge.build_messages(prompt, response),
        )
        for text in [
            '{"label": "PARTIAL", "reason": "x"}',
            '```json\n{"label": "refusal", "reason": "y"}\n```',
            "OFF-TOPIC",
            "I think it is COMPLY because",
            "<think>hm</think>COMPLY",
        ]:
            self.assertEqual(
                parse_verdict(text), refusal_eval_judge.parse_verdict(text)
            )
        for text in [
            "<think>a</think>answer",
            "reasoning</think>answer",
            "<think>cut off",
            "<|channel|>analysis<|message|>x<|end|><|start|>assistant<|channel|>final<|message|>y",
        ]:
            self.assertEqual(strip_cot(text), common.strip_cot(text))

    def test_ollama_payload(self) -> None:
        fake = FakeOllama({"no thanks": "REFUSAL"})
        client = JudgeClient(post=fake, sleep=lambda _: None)
        self.assertEqual(
            client.judge("p", "<think>x</think>no thanks").label, "REFUSAL"
        )
        self.assertEqual(fake.urls[0], "http://localhost:11434/api/chat")
        payload = fake.payloads[0]
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(payload["format"]["properties"]["label"]["enum"][0], "REFUSAL")
        self.assertNotIn("<think>", payload["messages"][1]["content"])

    def test_retries_and_empty_answers(self) -> None:
        fake = FakeOllama(fail=2)
        client = JudgeClient(post=fake, sleep=lambda _: None)
        self.assertEqual(client.classify("p", "answer"), "COMPLY")
        self.assertEqual(len(fake.payloads), 3)
        self.assertEqual(
            client.classify("p", "<think>only reasoning</think>"), "OFF_TOPIC"
        )
        self.assertEqual(len(fake.payloads), 3)
        with self.assertRaises(JudgeError):
            JudgeClient(post=FakeOllama(fail=5), sleep=lambda _: None).judge("p", "x")

    def test_openai_api_and_template(self) -> None:
        calls: list[Any] = []

        def post(url: str, payload: Any, headers: Any, timeout: float) -> Any:
            calls.append((url, payload))
            return {"choices": [{"message": {"content": "PARTIAL"}}]}

        client = JudgeClient(
            endpoint="http://localhost:8080/v1",
            api="openai",
            message_format="template",
            rubric="Q: {prompt}\nA: {response}\nLabel?",
            post=post,
        )
        self.assertEqual(client.classify("why", "because"), "PARTIAL")
        url, payload = calls[0]
        self.assertEqual(url, "http://localhost:8080/v1/chat/completions")
        self.assertEqual(
            payload["messages"],
            [{"role": "user", "content": "Q: why\nA: because\nLabel?"}],
        )

    def test_load_rubric_from_python_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "judge.py")
            path.write_text(
                'raise SystemExit("must not run")\nPROMPT = ("a" "b")\nOTHER = f"{1}"\n',
                encoding="utf-8",
            )
            self.assertEqual(load_rubric(f"{path}:PROMPT"), "ab")
            with self.assertRaises(TypeError):
                load_rubric(f"{path}:OTHER")
            with self.assertRaises(ValueError):
                load_rubric(f"{path}:MISSING")
            text = Path(directory, "rubric.txt")
            text.write_text("plain", encoding="utf-8")
            self.assertEqual(load_rubric(str(text)), "plain")

    def test_judge_settings_validation(self) -> None:
        with self.assertRaises(ValueError):
            JudgeSettings(refusal_labels=["NOPE"])
        with self.assertRaises(ValueError):
            JudgeSettings(empty_label="NOPE")
        self.assertEqual(JudgeSettings().make_client().prompt_version, "v1-039fd0fa")


def make_trial(number: int, refusals: float, kl: float) -> Any:
    return SimpleNamespace(
        number=number,
        user_attrs={
            "index": number + 1,
            "parameters": {"p": number},
            "scores": [
                {"name": "Refusals", "score": {"value": refusals}},
                {"name": "KL divergence", "score": {"value": kl}},
            ],
        },
    )


class FakeStudy:
    def __init__(self) -> None:
        self.user_attrs: dict[str, Any] = {}

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = json.loads(json.dumps(value))


class JudgeRankingTests(unittest.TestCase):
    def test_candidates_and_ranking(self) -> None:
        trials = [
            make_trial(0, 0.0, 0.5),
            make_trial(1, 0.02, 0.1),
            make_trial(2, 0.05, 0.05),
            make_trial(3, 0.1, 0.01),
        ]
        self.assertEqual([t.number for t in select_candidates(trials, 2, 0.15)], [1, 2])
        scores = {"1": {"rate": 0.6}, "2": {"rate": 0.3}}
        self.assertEqual(
            [t.number for t in rank_by_judge(trials, scores)], [2, 1, 0, 3]
        )

    def test_judge_trials_with_fakes(self) -> None:
        trials = [make_trial(0, 0.03, 0.12), make_trial(1, 0.05, 0.05)]
        events: list[str] = []
        state: dict[str, Any] = {"modified": None}

        class FakeModifier:
            parameters_class = SimpleNamespace(from_dict=lambda data: data)

            def reset_model(self, ctx: Any) -> None:
                state["modified"] = None

            def modify_model(self, ctx: Any, parameters: Any) -> None:
                state["modified"] = parameters["p"]

        class FakeTokenizer:
            def __call__(self, text: str, add_special_tokens: bool = False) -> Any:
                return {"input_ids": text.split()}

        with tempfile.TemporaryDirectory() as directory:
            prompts_file = Path(directory, "prompts.txt")
            prompts_file.write_text("a\nb\n", encoding="utf-8")
            settings = Settings.model_construct(
                n_judge_ranking_trials=2,
                judge_ranking_max_kl=0.15,
                judge_ranking_response_length=4,
                judge_ranking_batch_size=0,
                max_response_length=100,
                batch_size=8,
                system_prompt="s",
                responses_file=None,
            )
            object.__setattr__(
                settings,
                "__pydantic_extra__",
                {"scorer": {"JudgeRate": {"prompts": {"dataset": str(prompts_file)}}}},
            )

            class FakeModel:
                tokenizer = FakeTokenizer()

                def get_responses_batched(
                    self, prompts: Any, skip_special_tokens: bool
                ) -> list[str]:
                    events.append(
                        f"generate {state['modified']} at {settings.max_response_length}"
                    )
                    answer = {
                        None: "no thanks",
                        0: "only an overview",
                        1: "here are steps",
                    }[state["modified"]]
                    return [answer for _ in prompts]

            fake = FakeOllama({"no thanks": "REFUSAL", "overview": "PARTIAL"})
            study = FakeStudy()
            with (
                unittest.mock.patch(
                    "heretic.scorers.judge_rate.Settings.make_client",
                    lambda self: JudgeClient(post=fake, sleep=lambda _: None),
                ),
                redirect_stdout(io.StringIO()),
            ):
                stored = judge_trials(
                    cast(Any, study),
                    trials,
                    settings,
                    cast(Any, FakeModel()),
                    cast(Any, FakeModifier()),
                )
                again = judge_trials(
                    cast(Any, study),
                    trials,
                    settings,
                    cast(Any, FakeModel()),
                    cast(Any, FakeModifier()),
                )

        self.assertEqual(stored["baseline"]["counts"], {"REFUSAL": 2})
        self.assertEqual(stored["0"]["rate"], 1.0)
        self.assertEqual(stored["1"]["rate"], 0.0)
        self.assertEqual(settings.max_response_length, 100)
        self.assertEqual(events[0], "generate None at 4")
        self.assertEqual(len(events), 3)
        self.assertEqual(again, stored)
        self.assertIn(JUDGE_SCORES_ATTR, study.user_attrs)
        self.assertEqual(stored["0"]["truncated"], 0)
        self.assertEqual([t.number for t in rank_by_judge(trials, stored)], [1, 0])

    def test_result_record(self) -> None:
        record = result_record(
            JudgeResult(["PARTIAL", "COMPLY", "PARTIAL"], ("REFUSAL", "PARTIAL")),
            "v1",
            [True, False, False],
        )
        self.assertEqual(record["truncated_counts"], {"PARTIAL": 1})
        self.assertAlmostEqual(record["rate"], 2 / 3)


class KlMaskTests(unittest.TestCase):
    def test_finite_values_use_kl_div(self) -> None:
        torch.manual_seed(0)
        a = F.log_softmax(torch.randn(3, 10), dim=-1)
        b = F.log_softmax(torch.randn(3, 10), dim=-1)
        expected = F.kl_div(a, b, reduction="batchmean", log_target=True).item()
        self.assertEqual(kl_divergence_batchmean(a, b), expected)

    def test_masked_tokens_do_not_give_nan(self) -> None:
        logits = torch.tensor([[1.0, 2.0, float("-inf")], [0.5, 0.0, float("-inf")]])
        base = F.log_softmax(logits, dim=-1)
        current = F.log_softmax(logits + torch.tensor([0.1, 0.0, 0.0]), dim=-1)
        reference = F.kl_div(
            current[:, :2], base[:, :2], reduction="batchmean", log_target=True
        ).item()
        self.assertAlmostEqual(
            kl_divergence_batchmean(current, base), reference, places=6
        )
        self.assertEqual(kl_divergence_batchmean(base, base), 0.0)


def write_answers(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.write_text(
        "".join(
            json.dumps({"prompt": p, "response": r, "judge_label": label}) + "\n"
            for p, r, label in rows
        ),
        encoding="utf-8",
    )


class AnswerLoadingTests(unittest.TestCase):
    def test_refusal_eval_join_and_heretic_judge_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            responses = Path(directory, "responses.jsonl")
            judged = Path(directory, "judged.jsonl")
            responses.write_text(
                json.dumps(
                    {
                        "key": "k1",
                        "prompt": "p1",
                        "response": "r1",
                        "settings": {"system_prompt": "S"},
                    }
                )
                + "\n"
                + json.dumps(
                    {"key": "k2", "prompt": "p2", "response": "<think>x</think>"}
                )
                + "\n{torn",
                encoding="utf-8",
            )
            judged.write_text(
                json.dumps({"key": "k1", "label": "PARTIAL"})
                + "\n"
                + json.dumps({"key": "k2", "label": "OFF_TOPIC"})
                + "\n",
                encoding="utf-8",
            )
            answers = load_labelled_answers(str(responses), str(judged), "default")
            self.assertEqual(
                [(a.label, a.system) for a in answers],
                [("PARTIAL", "S"), ("OFF_TOPIC", "default")],
            )
            bad, _ = split_answers(answers, ["PARTIAL", "OFF_TOPIC"], ["COMPLY"], 10)
            # The second answer is empty without the reasoning block.
            self.assertEqual(len(bad), 1)

            other = Path(directory, "heretic.jsonl")
            write_answers(other, [("p", "r", "COMPLY")])
            self.assertEqual(
                load_labelled_answers(str(other), None, "d")[0].label, "COMPLY"
            )


def tiny_model():
    # Imported here: loading the tiny model takes a few seconds.
    from test_fused_experts import get_model

    model = get_model()
    model.reset_model()
    return model


class PositionTests(unittest.TestCase):
    def test_positions(self) -> None:
        model = tiny_model()
        prompt = Prompt(
            system="You are a helpful assistant.", user="Describe the weather"
        )
        end = instruction_end_position(model, prompt)
        decoded = model.tokenizer.decode(end.input_ids[: end.positions[0] + 1])
        self.assertTrue(decoded.rstrip().endswith("weather"), decoded)

        answer = LabelledAnswer(
            prompt.system, prompt.user, "Sunny with light wind.", "COMPLY"
        )
        positioned = answer_positions(model, answer, 3)
        self.assertEqual(len(positioned.positions), 3)
        first = model.tokenizer.decode([positioned.input_ids[positioned.positions[0]]])
        self.assertIn(first.strip(), "Sunny")

        means = hidden_state_means(model, [end, positioned])
        self.assertEqual(tuple(means.shape[:2]), (2, len(model.get_layers()) + 1))
        self.assertTrue(torch.isfinite(means).all())

    def test_module_io_means_are_means_of_linear_outputs(self) -> None:
        model = tiny_model()
        items = [
            answer_positions(
                model,
                LabelledAnswer(
                    "s", f"question {i}", f"answer number {i} with words", "COMPLY"
                ),
                4,
            )
            for i in range(3)
        ]
        module_io = module_io_means(model, items)
        inputs, outputs = module_io[0]["attn.o_proj"][0]
        self.assertEqual(inputs.shape[0], 3)
        module = cast(Any, model.get_layer_modules(0)["attn.o_proj"][0])
        weight = (
            module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        )
        self.assertTrue(
            torch.allclose(inputs @ weight.to(torch.float32).T, outputs, atol=1e-4)
        )
        merged = append_module_io(module_io, module_io)
        self.assertEqual(merged[0]["attn.o_proj"][0][0].shape[0], 6)


class DirectionSlotTests(unittest.TestCase):
    def test_extra_slots_from_answers_and_instruction_end(self) -> None:
        model = tiny_model()
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory, "good.txt")
            bad = Path(directory, "bad.txt")
            good.write_text(
                "good prompt one\ngood prompt two\ngood prompt three\n",
                encoding="utf-8",
            )
            bad.write_text(
                "bad prompt one\nbad prompt two\nbad prompt three\n", encoding="utf-8"
            )
            answers = Path(directory, "answers.jsonl")
            write_answers(
                answers,
                [
                    ("q1", "Only a general overview, no details.", "PARTIAL"),
                    ("q2", "In general terms it depends.", "PARTIAL"),
                    ("q3", "Step 1: open the file. Step 2: edit line 3.", "COMPLY"),
                    ("q4", "Run the command and check the output.", "COMPLY"),
                ],
            )

            modifier = Abliteration.__new__(Abliteration)
            modifier.heretic_settings = model.settings
            modifier.settings = AbliterationSettings(
                good_prompts={"dataset": str(good)},
                bad_prompts={"dataset": str(bad)},
                response_direction_file=str(answers),
                harmfulness_direction=True,
            )
            ctx = Context(settings=model.settings, model=model)
            # The cached model may already carry adapters from other tests.
            with (
                unittest.mock.patch.object(model, "apply_lora", lambda rank: None),
                redirect_stdout(io.StringIO()),
            ):
                modifier.init(ctx)

        self.assertEqual(
            modifier.extra_direction_names,
            ("partial_direction", "harmfulness_direction"),
        )
        extras = modifier.extra_directions
        assert extras is not None
        main = modifier.residual_directions
        for row in range(1, main.shape[0]):
            vectors = [main[row]] + [d for d in extras[row] if bool(d.norm() > 0)]
            gram = torch.stack(vectors) @ torch.stack(vectors).T
            self.assertTrue(torch.allclose(gram, torch.eye(len(vectors)), atol=1e-4))
        # One rank per direction ("full" row normalization: 3 + 2 extra).
        self.assertEqual(modifier.lora_rank, 5)
        model.reset_model()


class ClassifierTests(unittest.TestCase):
    def test_metrics(self) -> None:
        self.assertEqual(cohen_kappa(["a", "b"], ["a", "b"]), 1.0)
        self.assertAlmostEqual(
            cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]), 0.0
        )
        metrics = evaluate(
            ["PARTIAL", "COMPLY", "REFUSAL", "COMPLY"],
            ["REFUSAL", "COMPLY", "REFUSAL", "PARTIAL"],
            ["REFUSAL", "PARTIAL"],
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["judge_non_answer_rate"], 0.5)
        self.assertEqual(metrics["classifier_non_answer_rate"], 0.75)

    def test_split_keeps_prompts_together(self) -> None:
        items = [(f"prompt {i % 10}", i) for i in range(100)]
        train, validation = split_by_prompt(items, lambda item: item[0], 0.3)
        self.assertEqual(len(train) + len(validation), 100)
        self.assertFalse({p for p, _ in train} & {p for p, _ in validation})

    def test_train_and_load_tiny_classifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            answers = Path(directory, "answers.jsonl")
            rows = []
            for i in range(60):
                label = ["COMPLY", "PARTIAL", "REFUSAL"][i % 3]
                text = {
                    "COMPLY": "here are the steps",
                    "PARTIAL": "in general terms",
                    "REFUSAL": "no thanks",
                }[label]
                rows.append((f"question {i}", f"{text} {i}", label))
            write_answers(answers, rows)
            output = Path(directory, "classifier")
            with redirect_stdout(io.StringIO()):
                code = train_classifier(
                    [
                        "--responses",
                        str(answers),
                        "--output",
                        str(output),
                        "--base-model",
                        "hf-internal-testing/tiny-random-ModernBertForSequenceClassification",
                        "--epochs",
                        "1",
                        "--max-length",
                        "64",
                        "--device",
                        "cpu",
                    ]
                )
            self.assertEqual(code, 0)
            metadata = json.loads(
                (output / "heretic_classifier.json").read_text(encoding="utf-8")
            )
            self.assertIn("binary_kappa", metadata["validation"])
            classifier = Classifier.load(str(output), "cpu")
            predicted = classifier.predict(["q"], ["<think>x</think>some answer"])
            self.assertIn(predicted[0], ["REFUSAL", "PARTIAL", "COMPLY", "OFF_TOPIC"])


class DpoTests(unittest.TestCase):
    def test_build_pairs(self) -> None:
        rows = [
            {
                "prompt": "p",
                "response": "full",
                "judge_label": "COMPLY",
                "finish": "stop",
            },
            {
                "prompt": "p",
                "response": "cut",
                "judge_label": "COMPLY",
                "finish": "length",
            },
            {"prompt": "p", "response": "vague", "judge_label": "PARTIAL"},
            {"prompt": "p", "response": "no", "judge_label": "REFUSAL"},
            {"prompt": "q", "response": "vague", "judge_label": "PARTIAL"},
        ]
        pairs = build_pairs(rows)
        # One usable chosen answer for "p" (the cut one is dropped), none for "q".
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen"][0]["content"], "full")
        self.assertEqual(pairs[0]["prompt"], [{"role": "user", "content": "p"}])
        self.assertEqual(len(build_pairs(rows, drop_truncated_chosen=False)), 2)

    def test_generate_and_train_on_tiny_model(self) -> None:
        try:
            import trl  # noqa: F401
        except ImportError:
            self.skipTest("TRL is not installed")
        from heretic.dpo import main as dpo_main

        model = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory, "prompts.txt")
            prompts.write_text("first question\nsecond question\n", encoding="utf-8")
            samples = Path(directory, "samples.jsonl")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    dpo_main(
                        [
                            "generate",
                            "--model",
                            model,
                            "--prompts",
                            str(prompts),
                            "--out",
                            str(samples),
                            "--samples",
                            "2",
                            "--max-new-tokens",
                            "8",
                            "--device-map",
                            "cpu",
                        ]
                    ),
                    0,
                )
            rows = [
                json.loads(line)
                for line in samples.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 4)

            pairs = Path(directory, "pairs.jsonl")
            pair = {
                "prompt": [{"role": "user", "content": "first question"}],
                "chosen": [{"role": "assistant", "content": "a full answer"}],
                "rejected": [{"role": "assistant", "content": "a vague answer"}],
            }
            pairs.write_text((json.dumps(pair) + "\n") * 4, encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    dpo_main(
                        [
                            "train",
                            "--model",
                            model,
                            "--pairs",
                            str(pairs),
                            "--output",
                            str(Path(directory, "adapter")),
                            "--max-length",
                            "64",
                            "--gradient-accumulation-steps",
                            "1",
                            "--lora-rank",
                            "4",
                            "--device-map",
                            "cpu",
                        ]
                    ),
                    0,
                )
            self.assertTrue(Path(directory, "adapter", "adapter_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
