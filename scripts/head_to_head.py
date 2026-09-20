#!/usr/bin/env python3
"""
本地 PPO 模型 vs 开源 AlphaZero 3 人预训练模型，直接 head-to-head。

做法：在开源引擎（numba Board）里对局，3 个座位分别是 [本地模型, 开源模型, 随机]，
每局随机洗牌座位。统计本地模型 vs 开源模型的胜局数。

桥接层：
   1. 开源棋盘(71x7) -> 本地 SplendorState -> 本地观察(384)
   2. 本地动作(46) -> 开源动作(81)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "alpha-zero-general"))

from splendor.SplendorLogicNumba import Board
from splendor.SplendorLogic import np_different_gems_up_to_3

from games.splendor.state import SplendorState, PlayerState
from games.splendor.cards import DevelopmentCard, NobleTile
from games.splendor.constants import CardTier, GemColor
from games.splendor.encoder import SplendorEncoder

# ===== 颜色置换 =====
# 他们的颜色顺序: [white=0, blue=1, green=2, red=3, black=4, gold=5]
# 我的颜色顺序:   [red=0, green=1, blue=2, white=3, black=4, gold=5]
# my[i] = their[THEIR_TO_MY[i]]
THEIR_TO_MY = [3, 2, 1, 0, 4, 5]
# their[i] = my[MY_TO_THEIR[i]]  (对 5 种基础颜色)
MY_TO_THEIR = [3, 2, 1, 0, 4]


def their_gems6_to_my(their6):
    return [their6[3], their6[2], their6[1], their6[0], their6[4], their6[5]]


def their_cost5_to_my(their5):
    return [their5[3], their5[2], their5[1], their5[0], their5[4]]


def their_bonus_to_my(their_gain5):
    their_b = int(np.argmax(their_gain5))
    return {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}[their_b]


_dummy_id = [0]


def _mk_card(tier, points, bonus_color, cost):
    _dummy_id[0] += 1
    return DevelopmentCard(
        card_id=_dummy_id[0] % 100000,
        tier=CardTier(tier),
        points=int(points),
        bonus_color=GemColor(bonus_color),
        cost=tuple(cost),
    )


def their_board_to_my_state(board):
    """把开源 numba 棋盘转成我的 SplendorState（卡牌用 dummy id，只保留编码所需字段）。"""
    n = board.num_players

    # 宝石堆
    my_gem_bank = their_gems6_to_my([int(x) for x in board.bank[0][:6]])

    # 公开卡牌（12 张，3 tier × 4 位置）
    open_cards = {CardTier.TIER_1: [], CardTier.TIER_2: [], CardTier.TIER_3: []}
    for tier in range(3):
        for pos in range(4):
            cost_row = board.cards_tiers[8 * tier + 2 * pos]
            gain_row = board.cards_tiers[8 * tier + 2 * pos + 1]
            if int(cost_row[:5].sum()) == 0:
                continue  # 空位
            card = _mk_card(tier + 1, gain_row[6], their_bonus_to_my(gain_row[:5]),
                            their_cost5_to_my([int(x) for x in cost_row[:5]]))
            open_cards[CardTier(tier + 1)].append(card)

    # 贵族（4 张，3 人局）
    nobles = []
    for i in range(n + 1):
        req = board.nobles[i][:5]
        if int(req.sum()) > 0:
            nobles.append(NobleTile(
                noble_id=i, name=str(i),
                requirements=tuple(their_cost5_to_my([int(x) for x in req])),
            ))

    # 玩家
    players = []
    for p in range(n):
        my_gems = their_gems6_to_my([int(x) for x in board.players_gems[p][:6]])
        # 加成 + 分数
        their_bonuses = [int(x) for x in board.players_cards[p][:5]]
        my_bonuses = their_cost5_to_my(their_bonuses)
        total_points = int(board.players_cards[p][6])

        # 重构卡牌列表（bonus 数量 + 总分），卡牌具体内容对编码不重要。
        # 每张卡最多 5 分，把 total_points 分散到各张卡上（总分 <= 5*卡数，总能分完）。
        num_cards = int(sum(my_bonuses))
        remaining_points = total_points
        cards = []
        for color, cnt in enumerate(my_bonuses):
            for _ in range(int(cnt)):
                pts = min(5, remaining_points)
                remaining_points -= pts
                cards.append(_mk_card(1, pts, color, (0, 0, 0, 0, 0)))

        # 保留卡（3 张）
        reserved = []
        for r in range(3):
            cost_row = board.players_reserved[6 * p + 2 * r]
            gain_row = board.players_reserved[6 * p + 2 * r + 1]
            if int(cost_row[:5].sum()) == 0:
                continue
            card = _mk_card(1, gain_row[6], their_bonus_to_my(gain_row[:5]),
                            their_cost5_to_my([int(x) for x in cost_row[:5]]))
            reserved.append(card)

        # 贵族数量
        their_nobles = board.players_nobles[p * (n + 1):(p + 1) * (n + 1)]
        noble_cnt = int(sum(1 for nb in their_nobles if int(nb[:5].sum()) > 0))
        my_nobles = [NobleTile(noble_id=999, name="x", requirements=(8, 0, 0, 0, 0))
                     for _ in range(noble_cnt)]

        players.append(PlayerState(
            player_id=p, gems=my_gems, cards=cards, reserved_cards=reserved, nobles=my_nobles,
        ))

    state = SplendorState(
        num_players=n, players=players, gem_bank=my_gem_bank, nobles=nobles,
        open_cards=open_cards, decks={CardTier.TIER_1: [], CardTier.TIER_2: [], CardTier.TIER_3: []},
        current_player=int(board.current_player_index) if hasattr(board, "current_player_index") else 0,
        turn_number=int(board.get_round()),
    )
    return state


# ===== 我的 46 动作 -> 他们的 81 动作 =====
# 构建 take3 映射
_MY_TAKE3 = [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
             (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]


def _build_take3_map():
    m = {}
    for my_i, my_combo in enumerate(_MY_TAKE3):
        their_combo = tuple(sorted(MY_TO_THEIR[c] for c in my_combo))
        for their_i in range(15, 25):
            row = np_different_gems_up_to_3[their_i][:5]
            their_colors = tuple(np.where(row == 1)[0])
            if their_colors == their_combo:
                m[my_i] = 30 + their_i  # 他们的动作
                break
    return m


_MY_TAKE3_TO_THEIR = _build_take3_map()


def my_action_to_their_action(my_idx):
    """我的 46 槽位 -> 他们的 81 槽位。返回 -1 表示无映射。"""
    if my_idx < 5:  # 拿2同色
        return 55 + MY_TO_THEIR[my_idx]
    elif my_idx < 15:  # 拿3不同
        return _MY_TAKE3_TO_THEIR[my_idx - 5]
    elif my_idx < 27:  # 保留明牌 (tier 1-3, pos 0-3)
        offset = my_idx - 15
        return 12 + offset  # 他们的 reserve visible 也是 tier-major，12 个
    elif my_idx < 30:  # 保留牌堆
        return 24 + (my_idx - 27)
    elif my_idx < 42:  # 买明牌
        offset = my_idx - 30
        return offset  # 他们的 buy visible 是 [0:12]
    elif my_idx < 45:  # 买保留
        return 27 + (my_idx - 42)
    elif my_idx == 45:  # pass
        return 80
    return -1


def load_models():
    # 开源模型
    ckpt = torch.load(project_root / "third_party/alpha-zero-general/splendor/pretrained_3players.pt",
                      map_location="cpu", weights_only=False)
    open_model = ckpt["full_model"]
    open_model.eval()

    # 本地模型
    from models.model_factory import create_model
    my_model = create_model(obs_dim=384, action_size=46, encoder_type="mlp", config="medium")
    my_ckpt = torch.load(project_root / "data/splendor/checkpoints/mlp_medium_3p_v1/latest.pth",
                         map_location="cpu", weights_only=False)
    my_model.load_state_dict(my_ckpt["model_state_dict"])
    my_model.eval()

    encoder = SplendorEncoder()
    return open_model, my_model, encoder


def my_model_action(board, player, my_model, encoder):
    """本地模型选动作：转成我的 obs，argmax 策略，映射回他们的动作（并保证合法）。"""
    state = their_board_to_my_state(board)
    obs = encoder.encode(state, player)

    legal_my = []
    # 生成我的合法动作（简化：按 46 槽位逐个检查是否有映射且开源侧合法）
    their_valid = board.valid_moves(player)
    for my_idx in range(46):
        their_idx = my_action_to_their_action(my_idx)
        if their_idx >= 0 and their_valid[their_idx]:
            legal_my.append(my_idx)

    if not legal_my:
        # 兜底：pass
        return 80

    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    mask = torch.zeros(1, 46, dtype=torch.bool)
    mask[0, legal_my] = True
    with torch.no_grad():
        logits, _ = my_model(obs_t, mask)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    probs = probs * mask[0].numpy()
    my_idx = int(np.argmax(probs))
    return my_action_to_their_action(my_idx)


def open_model_action(board, player, open_model):
    """开源模型选动作（贪婪 argmax）。"""
    valid = board.valid_moves(player)
    state = board.get_state()
    inp = torch.FloatTensor(state.astype(np.float32)).unsqueeze(0)
    val = torch.BoolTensor(valid.astype(bool)).unsqueeze(0)
    with torch.no_grad():
        log_pi, _ = open_model(inp, val)
    probs = torch.exp(log_pi).cpu().numpy()[0] * valid.astype(np.float32)
    return int(np.argmax(probs))


def random_action(board, player):
    idxs = np.where(board.valid_moves(player))[0]
    return int(np.random.choice(idxs))


def play_one_game(open_model, my_model, encoder, role_map):
    """role_map: {seat: 'my'|'open'|'random'}"""
    board = Board(3)
    current = 0
    while True:
        result = board.check_end_game()
        if np.any(result != 0.0):
            winner = int(np.argmax(result))
            return winner, role_map[winner]
        role = role_map[current]
        if role == "my":
            a = my_model_action(board, current, my_model, encoder)
        elif role == "open":
            a = open_model_action(board, current, open_model)
        else:
            a = random_action(board, current)
        current = board.make_move(a, current, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-games", type=int, default=60)
    args = ap.parse_args()

    open_model, my_model, encoder = load_models()
    print("模型加载完成")

    roles = ["my", "open", "random"]
    my_wins = open_wins = random_wins = 0

    for g in range(args.num_games):
        role_map = dict(zip(range(3), np.random.permutation(roles)))
        winner, winner_role = play_one_game(open_model, my_model, encoder, role_map)
        if winner_role == "my":
            my_wins += 1
        elif winner_role == "open":
            open_wins += 1
        else:
            random_wins += 1
        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{args.num_games} 局 | 本地胜 {my_wins}, 开源胜 {open_wins}, 随机胜 {random_wins}")

    print(f"\n=== Head-to-Head 结果 ({args.num_games} 局) ===")
    print(f"本地 PPO 模型胜: {my_wins} ({my_wins/args.num_games:.1%})")
    print(f"开源 AlphaZero 胜: {open_wins} ({open_wins/args.num_games:.1%})")
    print(f"随机胜: {random_wins} ({random_wins/args.num_games:.1%})")


if __name__ == "__main__":
    main()
