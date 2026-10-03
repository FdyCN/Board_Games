"""
对局记录（core/transcript.py）测试。
"""

from __future__ import annotations

import json
import random

from games.coup import CoupGame
from core.transcript import EpisodeRecorder, record_episode


def _random_controller(game, state, player, legal):
    return random.choice(legal)


def test_episode_recorder_stepwise():
    game = CoupGame(num_players=3, seed=7)
    rec = EpisodeRecorder(game, human_seats=[0], ai_labels={1: "random", 2: "random"}, seed=7)
    state = game.reset()
    while not game.is_terminal(state):
        p = game.get_current_player(state)
        legal = game.get_legal_actions(state)
        action = random.choice(legal)
        next_state, rewards, done, _ = game.step(action)
        rec.record_step(state, p, action, rewards[p], done)
        state = next_state
        if done:
            break
    rec.finish(state)
    d = rec.to_dict()

    assert d["game"] == "coup"
    assert d["num_players"] == 3
    assert d["human_seats"] == [0]
    assert len(d["steps"]) > 0
    assert "winner" in d["result"]
    # 每步关键字段齐全
    s0 = d["steps"][0]
    for key in ("player", "is_human", "action_idx", "obs", "reward", "done"):
        assert key in s0, f"缺字段 {key}"
    assert len(s0["obs"]) == game.observation_shape[0]


def test_record_episode_full_loop():
    game = CoupGame(num_players=3, seed=7)
    controllers = {i: _random_controller for i in range(3)}
    ep = record_episode(game, controllers, human_seats=[0], seed=7)
    assert ep["type"] == "episode"
    assert len(ep["steps"]) > 0
    assert ep["result"]["winner"] >= 0


def test_save_jsonl_append(tmp_path):
    game = CoupGame(num_players=2, seed=1)
    out = tmp_path / "t.jsonl"
    for seed in (1, 2):
        rec = EpisodeRecorder(game, human_seats=[0], seed=seed)
        state = game.reset()
        while not game.is_terminal(state):
            p = game.get_current_player(state)
            action = random.choice(game.get_legal_actions(state))
            state, rewards, done, _ = game.step(action)
        rec.finish(state)
        rec.save(out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert json.loads(line)["type"] == "episode"
