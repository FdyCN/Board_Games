"""
情书（Love Letter）游戏引擎测试。
"""

import random

import pytest

from games.love_letter import LoveLetterGame, LoveLetterState, PlayCardAction
from games.love_letter.constants import (
    BARON,
    COUNTESS,
    GUARD,
    HANDMAID,
    KING,
    PRIEST,
    PRINCE,
    PRINCESS,
)


def _make_state(n=3):
    return LoveLetterState(
        num_players=n,
        tokens=[0] * n,
        target_tokens=5,
        deck=[],
        hands=[[] for _ in range(n)],
        discards=[[] for _ in range(n)],
        eliminated=[False] * n,
        protected=[False] * n,
        current_player=0,
    )


def test_game_registered():
    """测试游戏注册（对全局注册表清理保持健壮）。"""
    from games.registry import create_game, list_games, unregister_game, is_registered

    if is_registered("love_letter"):
        unregister_game("love_letter")

    # 重新导入以触发注册
    import games.love_letter.game
    import importlib
    importlib.reload(games.love_letter.game)

    assert "love_letter" in list_games()
    game = create_game("love_letter", num_players=3)
    assert game.__class__.__name__ == "LoveLetterGame"
    assert game.num_players == 3


def test_action_space_size():
    game = LoveLetterGame(num_players=3)
    assert game.action_space_size == 11 * 3 - 3  # 30


def test_observation_shape():
    game = LoveLetterGame(num_players=3)
    assert game.observation_shape == (18 + 11 * 3,)  # 51


def test_initial_state():
    game = LoveLetterGame(num_players=3, seed=42)
    state = game.reset()
    # 16 张 - 移除 1 - 发 3 - 起始玩家抽 1 = 11
    assert len(state.deck) == 11
    assert len(state.hands[0]) == 2  # 起始玩家已抽第二张
    assert len(state.hands[1]) == 1
    assert len(state.hands[2]) == 1
    assert state.current_player == 0
    assert not state.game_over


def test_action_index_roundtrip():
    game = LoveLetterGame(num_players=3, seed=1)
    state = game.reset()
    for _ in range(20):
        for action in game.get_legal_actions(state):
            idx = game.action_to_index(action, state)
            back = game.index_to_action(idx, state)
            assert back == action
        legal = game.get_legal_actions(state)
        state, _, done, _ = game.step(random.choice(legal))
        if done:
            break


def test_legal_actions_match_hand():
    game = LoveLetterGame(num_players=3, seed=2)
    state = game.reset()
    hand = state.hands[0]
    cards_played = {a.card for a in game.get_legal_actions(state)}
    assert cards_played == set(hand)


def test_observation_fixed_dim():
    game = LoveLetterGame(num_players=3, seed=3)
    state = game.reset()
    for _ in range(30):
        obs = game.state_to_observation(state, state.current_player)
        assert obs.shape == game.observation_shape
        legal = game.get_legal_actions(state)
        state, _, done, _ = game.step(random.choice(legal))
        if done:
            break


def test_full_game_random_reaches_terminal():
    """随机策略完整玩完一局，必须终局且产生唯一胜者（无死锁）。"""
    game = LoveLetterGame(num_players=3, seed=7)
    state = game.reset()
    steps = 0
    while not game.is_terminal(state) and steps < 1000:
        legal = game.get_legal_actions(state)
        assert legal, f"死锁：第 {steps} 步无合法动作"
        state, _, done, _ = game.step(random.choice(legal))
        steps += 1
    assert game.is_terminal(state)
    assert game.get_winner(state) >= 0
    rewards = game.get_final_rewards(state)
    assert sum(rewards) == pytest.approx(1.0)
    assert rewards[game.get_winner(state)] == pytest.approx(1.0)


def test_forced_countess():
    game = LoveLetterGame(num_players=3, seed=4)
    state = _make_state()
    state.hands[0] = [COUNTESS, KING]  # 手上有女伯爵 + 国王 → 必须打女伯爵
    legal = game.get_legal_actions(state)
    assert all(a.card == COUNTESS for a in legal)
    assert len(legal) == 1


def test_guard_guesses():
    game = LoveLetterGame(num_players=3, seed=5)
    state = _make_state()
    state.hands[0] = [GUARD, PRINCESS]
    state.hands[1] = [KING]
    state.hands[2] = [PRIEST]
    legal = game.get_legal_actions(state)
    guard_actions = [a for a in legal if a.card == GUARD]
    # 两个目标 × 7 种猜测
    assert len(guard_actions) == 2 * 7
    assert {a.guess for a in guard_actions} == set(range(2, 9))


def test_princess_play_eliminates_self():
    game = LoveLetterGame(num_players=3)
    state = _make_state()
    state.hands[0] = [PRINCESS, GUARD]
    game._play_card(state, 0, PlayCardAction(PRINCESS))
    assert state.eliminated[0]
    # 出局后揭示并弃掉剩余手牌
    assert state.hands[0] == []


def test_baron_eliminates_lower():
    game = LoveLetterGame(num_players=3)
    state = _make_state()
    state.hands[0] = [BARON, GUARD]  # 打出男爵后，剩余手牌 = 卫兵(1)
    state.hands[1] = [KING]          # 对手 = 国王(6)
    game._play_card(state, 0, PlayCardAction(BARON, target=1))
    assert state.eliminated[0]
    assert not state.eliminated[1]


def test_baron_tie_no_elimination():
    game = LoveLetterGame(num_players=3)
    state = _make_state()
    state.hands[0] = [BARON, GUARD]  # 剩余 = 卫兵(1)
    state.hands[1] = [GUARD]         # 对手 = 卫兵(1)
    game._play_card(state, 0, PlayCardAction(BARON, target=1))
    assert not state.eliminated[0]
    assert not state.eliminated[1]


def test_guard_correct_guess_eliminates():
    game = LoveLetterGame(num_players=3)
    state = _make_state()
    state.hands[0] = [GUARD, HANDMAID]
    state.hands[1] = [PRIEST]  # 神父 = 2
    game._play_card(state, 0, PlayCardAction(GUARD, target=1, guess=PRIEST))
    assert state.eliminated[1]
    assert state.hands[1] == []  # 揭示弃掉


def test_guard_wrong_guess():
    game = LoveLetterGame(num_players=3)
    state = _make_state()
    state.hands[0] = [GUARD, HANDMAID]
    state.hands[1] = [PRIEST]
    game._play_card(state, 0, PlayCardAction(GUARD, target=1, guess=BARON))
    assert not state.eliminated[1]
    assert state.hands[1] == [PRIEST]


def test_king_swaps_hands():
    game = LoveLetterGame(num_players=3)
    state = _make_state()
    state.hands[0] = [KING, GUARD]  # 打出国王后剩余 = 卫兵(1)
    state.hands[1] = [PRINCESS]
    game._play_card(state, 0, PlayCardAction(KING, target=1))
    assert state.hands[0] == [PRINCESS]
    assert state.hands[1] == [GUARD]


def test_handmaid_protection_blocks_targeting():
    game = LoveLetterGame(num_players=3)
    state = _make_state()
    state.hands[0] = [PRIEST, HANDMAID]
    state.hands[1] = [KING]
    state.hands[2] = [GUARD]
    state.protected[1] = True  # P1 受保护
    legal = game.get_legal_actions(state)
    targets = {a.target for a in legal if a.card == PRIEST}
    assert targets == {2}  # 只能选 P2，不能选受保护的 P1


def test_prince_target_discards_and_draws():
    game = LoveLetterGame(num_players=3, seed=6)
    state = _make_state()
    state.deck = [KING]  # 牌堆有牌可抽
    state.hands[0] = [PRINCE, HANDMAID]
    state.hands[1] = [PRIEST]
    game._play_card(state, 0, PlayCardAction(PRINCE, target=1))
    assert PRIEST in state.discards[1]  # P1 弃掉神父
    assert state.hands[1] == [KING]     # 重抽到国王


def test_prince_on_princess_eliminates():
    game = LoveLetterGame(num_players=3)
    state = _make_state()
    state.hands[0] = [PRINCE, HANDMAID]
    state.hands[1] = [PRINCESS]
    game._play_card(state, 0, PlayCardAction(PRINCE, target=1))
    assert state.eliminated[1]
    assert state.hands[1] == []


def test_relative_target_mapping():
    """相对目标：当前玩家为 p 时，target=r 指向绝对玩家 (p+r)%n。"""
    game = LoveLetterGame(num_players=3)
    state = _make_state()
    state.current_player = 1
    state.hands[1] = [BARON, HANDMAID]  # 玩家1 打出男爵，剩侍女(4)
    state.hands[2] = [PRINCESS]         # 相对 target=1 → 绝对 (1+1)%3 = 2
    game._play_card(state, 1, PlayCardAction(BARON, target=1))
    # 侍女(4) < 公主(8) → 玩家1 出局
    assert state.eliminated[1]
    assert not state.eliminated[2]


def test_eliminate_opponent_reward():
    """消灭对手应给予 eliminate_opponent 稠密奖励。"""
    game = LoveLetterGame(num_players=3, reward_config={
        "round_win": 0.5, "eliminate_opponent": 0.3, "step_penalty": -0.01,
    })
    state = _make_state()
    state.deck = [1, 2, 3]  # 保证不会因牌堆空结束本轮
    state.hands[0] = [GUARD, HANDMAID]
    state.hands[1] = [PRIEST]  # 神父 = 2
    state.hands[2] = [KING]
    game._state = state
    new_state, rewards, done, info = game.step(PlayCardAction(GUARD, target=1, guess=PRIEST))
    assert new_state.eliminated[1]
    assert info["dense_reward"] == pytest.approx(-0.01 + 0.3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
