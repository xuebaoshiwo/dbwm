"""Offline tests: no model provider or network call is needed."""

from __future__ import annotations

import json
from collections import Counter

from bfcl_eval.consistency.data_generator.core import Call, build_trajectory
from bfcl_eval.consistency.data_generator.generate import make_trading_jobs
from bfcl_eval.consistency.data_generator.trading import TradingAdapter


def _fake_model(prompt: str, **_: object) -> str:
    request = json.loads(prompt)
    candidate_count = len(request["distractor_candidates"])
    choices = [index % candidate_count for index in range(request["required_distractor_count"])]
    return json.dumps({"distractor_ids": choices, "query_introduction": "Review the account over time."})


def test_thirty_trading_trajectories_are_grounded_and_solver_satisfiable() -> None:
    adapter = TradingAdapter()
    jobs = make_trading_jobs([5, 8, 12], [2, 5, 9], [1, 2, 3])
    assert Counter(job.length for job in jobs) == {5: 10, 8: 10, 12: 10}
    assert len({(job.length, job.scenario_name) for job in jobs}) == 30

    for job in jobs:
        trajectory = build_trajectory(adapter, job, _fake_model)
        assert trajectory["generation"]["model_proposal"]["source"] == "model_client.chat_text"
        assert len(trajectory["steps"]) == job.length
        assert trajectory["validation"]["actual_read_gap"] == job.read_gap
        assert trajectory["validation"]["actual_write_count"] == job.write_count
        assert trajectory["query"] and "observation" not in trajectory["query"]
        assert adapter.validate_trace(trajectory["steps"])["status"] == "sat"


def test_read_gap_can_leave_steps_after_second_read() -> None:
    adapter = TradingAdapter()
    job = make_trading_jobs([8], [3], [2])[0]
    trajectory = build_trajectory(adapter, job, _fake_model)
    assert trajectory["validation"]["target_read_steps"] == [2, 6]
    assert len(trajectory["steps"]) == 8
    assert trajectory["validation"]["actual_write_count"] == 2


def test_generated_order_history_schema_matches_backend() -> None:
    adapter = TradingAdapter()
    schema = next(item for item in adapter.tool_schema() if item["name"] == "get_order_history")
    assert set(schema["response"]["properties"]) == {"history"}
    observation, _ = adapter.rollout(
        make_trading_jobs([5], [2], [1])[0].initial_state,
        [
            Call("trading_login", {"username": "alice", "password": "demo"}),
            Call("get_order_history", {}),
        ],
    )
    assert set(observation[-1]) == {"history"}
