"""
政变疑云（Coup）引擎测试。

覆盖：重置/动作空间/观察编码、多阶段状态机（质疑/反制/扣血/换牌）、
终局判定，以及随机 rollout 能正常收敛。
"""

from __future__ import annotations

import random

import numpy as np

from games.coup import CoupGame
from games.coup.actions import CoupAction
from games.coup.constants import (
    DUKE, ASSASSIN, CAPTAIN, AMBASSADOR, CONTESSA,
    PHASE_ACTION, PHASE_CHALLENGE, PHASE_BLOCK, PHASE_BLOCK_CHALLENGE,
    PHASE_REVEAL, PHASE_EXCHANGE,
    INITIAL_COINS, FORCED_COUP_COINS, COUP_COST, ASSASSINATE_COST,
)


def _game(n=3, seed=0):
    # 直接构造，避免受 registry 被 clear 的隔离影响（与 love_letter 测试一致）
    return CoupGame(num_players=n, seed=seed)


def _set_hand(g, p, cards):
    g._state.hands[p] = list(cards)


def _set_coins(g, p, c):
    g._state.coins[p] = c


def _pass_challenges(g):
    """在当前 CHALLENGE / BLOCK_CHALLENGE 阶段让所有候选依次 pass。"""
    st = g._state
    while st.phase in (PHASE_CHALLENGE, PHASE_BLOCK_CHALLENGE):
        g.step(CoupAction("pass"))
        st = g._state


# ===== 基础 =====

def test_reset_basics():
    g = _game(3)
    st = g.reset()
    assert st.num_players == 3
    assert all(len(h) == 2 for h in st.hands)
    assert all(c == INITIAL_COINS for c in st.coins)
    assert len(st.deck) == 9  # 15 - 3*2
    assert all(st.alive)
    assert st.phase == PHASE_ACTION
    assert st.current_player == 0


def test_action_space_and_observation():
    for n, obs_dim in [(2, 1066), (3, 1135), (4, 1204)]:
        g = _game(n)
        assert g.action_space_size == 3 * n + 16
        st = g.reset()
        obs = g.state_to_observation(st, 0)
        assert obs.shape == g.observation_shape == (obs_dim,)
        assert g.auxiliary_shape == ((n - 1) * 5,)


def test_action_index_roundtrip():
    g = _game(3)
    st = g.reset()
    for idx in range(g.action_space_size):
        a = g.index_to_action(idx, st)
        assert g.action_to_index(a, st) == idx, (idx, a)


def test_random_rollout_terminates():
    for n in (2, 3, 4):
        for seed in range(10):
            g = _game(n, seed=seed)
            st = g.reset()
            rng = random.Random(seed)
            steps = 0
            while not g.is_terminal(st) and steps < 3000:
                legal = g.get_legal_actions(st)
                st, _, done, _ = g.step(rng.choice(legal))
                steps += 1
                if done:
                    break
            assert g.is_terminal(st), f"{n}p seed={seed} 未收敛"
            alive = [i for i in range(n) if st.alive[i]]
            assert len(alive) == 1
            assert g.get_winner(st) == alive[0]


# ===== 基础行动 =====

def test_income_gives_one_coin():
    g = _game(3)
    g.reset()
    g.step(CoupAction("income"))
    assert g._state.coins[0] == INITIAL_COINS + 1
    assert g._state.phase == PHASE_ACTION
    assert g._state.turn_player == 1  # 回合推进


def test_coup_deducts_and_target_loses_influence():
    g = _game(3)
    g.reset()
    _set_coins(g, 0, COUP_COST)
    _set_hand(g, 1, [DUKE, CAPTAIN])
    g.step(CoupAction("coup", target=1))
    assert g._state.phase == PHASE_REVEAL  # 目标两张手牌 → 需选翻哪张
    assert g._state.current_player == 1
    g.step(CoupAction("reveal", reveal_slot=0))
    assert len(g._state.hands[1]) == 1
    assert g._state.coins[0] == 0  # 扣 7


def test_forced_coup_at_10_coins():
    g = _game(3)
    g.reset()
    _set_coins(g, 0, FORCED_COUP_COINS)
    legal = g.get_legal_actions(g._state)
    assert all(a.kind == "coup" for a in legal)
    assert len(legal) == 2  # 3 人 → 2 个目标


# ===== 质疑 =====

def test_challenge_bluff_caught():
    g = _game(3)
    g.reset()
    _set_hand(g, 0, [CAPTAIN, AMBASSADOR])  # 没有公爵
    g.step(CoupAction("tax"))  # 宣称公爵 → 说谎
    assert g._state.phase == PHASE_CHALLENGE
    assert g._state.current_player == 1
    g.step(CoupAction("challenge"))  # 质疑成功
    assert g._state.phase == PHASE_REVEAL
    assert g._state.current_player == 0  # 说谎者扣血
    g.step(CoupAction("reveal", reveal_slot=0))
    assert len(g._state.hands[0]) == 1
    assert g._state.coins[0] == INITIAL_COINS  # 行动取消，未征税


def test_challenge_defended():
    g = _game(3)
    g.reset()
    _set_hand(g, 0, [DUKE, CAPTAIN])  # 真有公爵
    g.step(CoupAction("tax"))
    assert g._state.phase == PHASE_CHALLENGE
    g.step(CoupAction("challenge"))  # 质疑失败
    assert g._state.phase == PHASE_REVEAL
    assert g._state.current_player == 1  # 质疑者扣血
    g.step(CoupAction("reveal", reveal_slot=0))
    assert len(g._state.hands[1]) == 1
    assert len(g._state.hands[0]) == 2  # 洗回重抽，不掉血
    assert g._state.coins[0] == INITIAL_COINS + 3  # 征税生效


# ===== 反制 =====

def test_block_assassinate_with_contessa():
    g = _game(3)
    g.reset()
    _set_hand(g, 0, [ASSASSIN, CAPTAIN])
    _set_coins(g, 0, 5)
    _set_hand(g, 1, [CONTESSA, AMBASSADOR])
    g.step(CoupAction("assassinate", target=1))
    _pass_challenges(g)  # p1、p2 都不质疑
    assert g._state.phase == PHASE_BLOCK
    assert g._state.current_player == 1  # 目标是反制决策者
    g.step(CoupAction("block", role=CONTESSA))
    assert g._state.phase == PHASE_BLOCK_CHALLENGE
    _pass_challenges(g)  # 无人质疑反制
    # 反制成立：行动取消，目标不掉血，刺客不扣钱
    assert g._state.phase == PHASE_ACTION
    assert len(g._state.hands[1]) == 2
    assert g._state.coins[0] == 5


def test_block_bluff_caught():
    g = _game(3)
    g.reset()
    _set_hand(g, 0, [ASSASSIN, CAPTAIN])
    _set_coins(g, 0, 5)
    _set_hand(g, 1, [AMBASSADOR, DUKE])  # 没有女伯爵
    g.step(CoupAction("assassinate", target=1))
    _pass_challenges(g)
    assert g._state.phase == PHASE_BLOCK
    g.step(CoupAction("block", role=CONTESSA))  # 说谎反制
    assert g._state.phase == PHASE_BLOCK_CHALLENGE
    g.step(CoupAction("challenge"))  # 质疑反制成功
    assert g._state.phase == PHASE_REVEAL
    assert g._state.current_player == 1  # 说谎的反制者扣血
    g.step(CoupAction("reveal", reveal_slot=0))
    # 反制取消 → 刺杀结算：扣 3 币，目标再掉血（1 张 → 自动翻开并淘汰）
    assert len(g._state.hands[1]) == 0
    assert g._state.alive[1] is False
    assert g._state.coins[0] == 2


# ===== 偷窃 / 换牌 =====

def test_steal_transfers_coins():
    g = _game(3)
    g.reset()
    _set_hand(g, 0, [CAPTAIN, DUKE])
    _set_coins(g, 0, 2)
    _set_coins(g, 1, 5)
    g.step(CoupAction("steal", target=1))
    _pass_challenges(g)
    assert g._state.phase == PHASE_BLOCK
    g.step(CoupAction("pass_block"))  # 目标不反制
    assert g._state.coins[0] == 4  # 偷 2
    assert g._state.coins[1] == 3


def test_exchange_keeps_two_cards():
    g = _game(3)
    g.reset()
    _set_hand(g, 0, [AMBASSADOR, CAPTAIN])
    deck_before = len(g._state.deck)
    g.step(CoupAction("exchange"))
    _pass_challenges(g)
    assert g._state.phase == PHASE_EXCHANGE
    assert g._state.current_player == 0
    assert len(g._state.exchange_hand) == 4
    g.step(CoupAction("exchange_keep", keep=(0, 2)))
    assert len(g._state.hands[0]) == 2
    assert AMBASSADOR in g._state.hands[0]
    # 牌堆：抽走 2 张后又归还 2 张 → 数量不变
    assert len(g._state.deck) == deck_before


# ===== 淘汰 / 终局 / 辅助 =====

def test_elimination_and_win():
    g = _game(3)
    g.reset()
    _set_hand(g, 1, [DUKE])  # 只剩 1 张
    _set_coins(g, 0, COUP_COST)
    g.step(CoupAction("coup", target=1))
    # 目标 1 张 → 自动翻开并淘汰，无需 reveal 决策
    assert g._state.alive[1] is False
    assert len(g._state.hands[1]) == 0


def test_auxiliary_labels():
    g = _game(3)
    st = g.reset()
    _set_hand(g, 0, [DUKE, CAPTAIN])
    _set_hand(g, 1, [AMBASSADOR, CONTESSA])
    _set_hand(g, 2, [DUKE, AMBASSADOR])
    labels = g.get_auxiliary_labels(st, 0)
    assert labels.shape == (2,)
    assert labels[0] == AMBASSADOR   # 相对对手 1：min([AMBASSADOR, CONTESSA])
    assert labels[1] == DUKE         # 相对对手 2：min([DUKE, AMBASSADOR])


def test_observation_changes_after_step():
    g = _game(3)
    st = g.reset()
    obs0 = g.state_to_observation(st, 0)
    st, _, _, _ = g.step(CoupAction("income"))
    obs1 = g.state_to_observation(st, 0)
    assert not np.array_equal(obs0, obs1)
