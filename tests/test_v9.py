# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests of the v9 clean-ups: lazily built questions (a fully configured run needs no
# console), pair diagnostics at the token level, the precision of the paired residuals,
# the per-category non-answers of the judge ranking, and the fork patch generator.

import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from heretic.judge import JudgeResult
from heretic.judge_ranking import category_rates, format_categories
from heretic.modifiers.abliteration import Abliteration, DirectionEstimator
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.plugin import Context
from heretic.positions import format_pair_diagnostics, pair_diagnostics
from heretic.utils import HereticUserError, ask, ask_if_unset

ROOT = Path(__file__).resolve().parent.parent


class LazyQuestionTests(unittest.TestCase):
    def test_a_known_value_never_builds_the_question(self) -> None:
        def factory() -> Any:
            raise AssertionError("the question must not be built")

        self.assertEqual(
            ask_if_unset("merge", factory, setting="export_strategy"), "merge"
        )

    def test_without_a_terminal_the_question_is_not_built_either(self) -> None:
        def factory() -> Any:
            raise AssertionError("the question must not be built")

        with patched_stdin():
            with self.assertRaisesRegex(HereticUserError, "--trial-index"):
                ask_if_unset(None, factory, setting="trial_index")
            with self.assertRaisesRegex(HereticUserError, "HF_TOKEN"):
                ask(factory, hint="Set the HF_TOKEN environment variable.")

    def test_a_terminal_builds_and_asks(self) -> None:
        built: list[str] = []

        def factory() -> Any:
            built.append("built")
            return SimpleNamespace(ask=lambda: "answer")

        with unittest.mock.patch.object(
            sys, "stdin", SimpleNamespace(isatty=lambda: True)
        ):
            self.assertEqual(ask_if_unset(None, factory), "answer")
            self.assertEqual(ask(factory), "answer")
        self.assertEqual(built, ["built", "built"])

    def test_every_menu_question_is_lazy(self) -> None:
        # A question built eagerly as an argument cannot be constructed without a console,
        # which is how the export-strategy test used to fail on Windows.
        for name in ("src/heretic/main.py", "src/heretic/reproduce.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            for match in re.finditer(r"(?<!lambda: )questionary\.\w+\(", source):
                line = source[: match.start()].count("\n") + 1
                self.assertNotIn(
                    "questionary",
                    match.group(0),
                    f"{name}:{line} builds a question eagerly",
                )


def patched_stdin():
    return unittest.mock.patch.object(sys, "stdin", io.StringIO(""))


def tiny_model():
    from test_fused_experts import get_model

    model = get_model()
    model.reset_model()
    return model


class PairDiagnosticsTests(unittest.TestCase):
    def test_counts_languages_and_last_tokens(self) -> None:
        from heretic.utils import Prompt

        model = tiny_model()
        bad = [
            Prompt(system="s", user="review the setup for weakness."),
            Prompt(system="s", user="проверь настройку на слабость."),
            Prompt(system="s", user="review the other setup for weakness."),
        ]
        good = [
            # Same final token.
            Prompt(system="s", user="review the setup for hardening."),
            # Same language, different final token.
            Prompt(system="s", user="проверь настройку на устойчивость?"),
            # Other language.
            Prompt(system="s", user="review the other setup for hardening."),
        ]
        counts = pair_diagnostics(model, bad, good)
        self.assertEqual(counts["pairs"], 3)
        self.assertEqual(counts["languages"], 3)
        self.assertEqual(counts["last_tokens"], 2)
        text = format_pair_diagnostics(counts)
        self.assertIn("[bold]3[/] pairs", text)
        self.assertIn("same last instruction token in [bold]2[/]", text)

        # Sets of different lengths are compared up to the shorter one.
        self.assertEqual(pair_diagnostics(model, bad, good[:1])["pairs"], 1)


class PairedResidualPrecisionTests(unittest.TestCase):
    def test_float16_keeps_the_directions_close_and_reports_the_memory(self) -> None:
        from heretic.utils import Prompt  # noqa: F401

        results = {}
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory, "bad.txt")
            good = Path(directory, "good.txt")
            bad.write_text(
                "".join(
                    f"review the term{i} setup for weakness {i}.\n" for i in range(12)
                ),
                encoding="utf-8",
            )
            good.write_text(
                "".join(
                    f"review the term{i} setup for hardening {i}.\n" for i in range(12)
                ),
                encoding="utf-8",
            )
            for dtype in ("float32", "float16"):
                model = tiny_model()
                modifier = Abliteration.__new__(Abliteration)
                modifier.heretic_settings = model.settings
                modifier.settings = AbliterationSettings(
                    good_prompts={"dataset": str(good)},
                    bad_prompts={"dataset": str(bad)},
                    direction_estimator=DirectionEstimator.WHITENED,
                    paired_residuals_dtype=dtype,
                )
                with (
                    unittest.mock.patch.object(model, "apply_lora", lambda rank: None),
                    redirect_stdout(io.StringIO()) as output,
                ):
                    modifier.init(Context(settings=model.settings, model=model))
                results[dtype] = (modifier.residual_directions, output.getvalue())

        self.assertIn("GiB of float16", results["float16"][1])
        self.assertIn("same last instruction token", results["float16"][1])
        present = results["float32"][0].norm(dim=1) > 0
        cosines = (results["float32"][0][present] * results["float16"][0][present]).sum(
            dim=1
        )
        for directions in (results["float32"][0], results["float16"][0]):
            norms = directions[present].norm(dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))
        # Whitening amplifies the axes along which the pairs barely vary, so float16 moves
        # the direction measurably; the setting is a documented memory trade-off, not an
        # equivalent, and this only pins that it stays the same direction in substance.
        self.assertGreater(float(cosines.abs().min()), 0.5)
        tiny_model()


class CategoryRateTests(unittest.TestCase):
    def result(self) -> tuple[JudgeResult, list[Any]]:
        labels = ["PARTIAL", "COMPLY", "REFUSAL", "COMPLY", "COMPLY"]
        categories = ["c2", "c2", "recon", "recon", ""]
        prompts = [
            SimpleNamespace(id=f"p{index}", category=category, prompt=None)
            for index, category in enumerate(categories)
        ]
        return JudgeResult(labels, ("REFUSAL", "PARTIAL")), prompts

    def test_rates_and_worst_categories(self) -> None:
        result, prompts = self.result()
        rates = category_rates(result, cast(Any, prompts))
        # Prompts without a category are left out.
        self.assertEqual(rates, {"c2": [1, 2], "recon": [1, 2]})
        self.assertEqual(format_categories({}), "")
        text = format_categories({"c2": [1, 4], "recon": [3, 4]})
        self.assertTrue(text.startswith("worst categories: recon 3/4"))
        self.assertEqual(
            format_categories({"a": [1, 2], "b": [2, 2]}, worst=1),
            "worst categories: b 2/2",
        )


class ForkPatchTests(unittest.TestCase):
    def test_the_generator_writes_a_patch_that_applies(self) -> None:
        script = ROOT / "tools" / "make_fork_patch.sh"
        self.assertTrue(script.is_file())
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory, "fork.patch")
            completed = subprocess.run(
                ["sh", str(script), str(patch)],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("sha256", completed.stdout)
            self.assertIn("upstream base", completed.stdout)
            body = patch.read_text(encoding="utf-8", errors="replace")

        # Only the fork's own sources travel, and the patch is not empty.
        self.assertIn("src/heretic/", body)
        self.assertNotIn("a/config.default.toml", body)
        self.assertNotIn("a/tests/", body)
        self.assertGreater(len(body.splitlines()), 100)

    def test_a_missing_base_commit_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    "sh",
                    str(ROOT / "tools" / "make_fork_patch.sh"),
                    str(Path(directory, "x.patch")),
                    "0000000",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                check=False,
            )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("not in this repository", completed.stderr)


class CiTests(unittest.TestCase):
    def test_ci_runs_pytest_so_plain_test_functions_are_covered(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("uv run pytest -q tests/", workflow)
        # The comment above that line may mention it; no command may run it.
        self.assertNotIn("python -m unittest discover", workflow)
        self.assertIn("pytest", (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        # The tests that unittest cannot collect: plain functions at module level.
        plain = [
            path.name
            for path in sorted((ROOT / "tests").glob("test_*.py"))
            if re.search(r"^def test_", path.read_text(encoding="utf-8"), re.MULTILINE)
        ]
        self.assertTrue(plain)
        self.assertTrue(all(os.path.exists(ROOT / "tests" / name) for name in plain))


if __name__ == "__main__":
    unittest.main()
