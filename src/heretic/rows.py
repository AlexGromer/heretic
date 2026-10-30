# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Answer rows in the format of refusal-eval's responses.jsonl (fork addition), so that the
# answers Heretic writes are judged, summarized and compared by refusal-eval itself
# (judge.py / judge_colab.py batch, compare_runs.py) instead of by a second copy of those
# tools here. See github.com/AlexGromer/refusal-eval, generate_colab.py.

from pathlib import Path
from typing import Any


def refusal_eval_key(
    model_name: str,
    trial: str,
    set_name: str,
    prompt_id: str,
    mode: str = "greedy",
    thinking: bool = False,
) -> str:
    # refusal-eval generate_colab.make_key.
    return f"{model_name}|{trial}|{mode}|{str(thinking).lower()}|{set_name}|{prompt_id}"


def model_label(model: str) -> str:
    """The model label of the rows: the last part of the model ID or path."""
    return Path(model).name


def response_row(
    *,
    model_name: str,
    trial: str,
    set_name: str,
    prompt_id: str,
    category: str,
    prompt: str,
    response: str,
    settings: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    """One greedy, thinking-off answer row; `extra` adds fields (finish, match, label, ...)."""
    return {
        "key": refusal_eval_key(model_name, trial, set_name, prompt_id),
        "model": model_name,
        "trial": trial,
        "mode": "greedy",
        "thinking": False,
        "set": set_name,
        "prompt_id": prompt_id,
        "category": category,
        "seed": 0,
        "prompt": prompt,
        "response": response,
        **extra,
        "settings": settings,
    }
