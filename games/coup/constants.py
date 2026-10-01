"""
政变疑云（Coup）游戏常量。

角色（0-4）：
    DUKE       公爵：征税(拿 3 币)，反制「外援」
    ASSASSIN   刺客：刺杀(付 3 币，令目标掉血)，被女伯爵反制
    CAPTAIN    队长：偷窃(拿目标 2 币)，反制「偷窃」
    AMBASSADOR 大使：换牌(抽 2 留 2)，反制「偷窃」
    CONTESSA   女伯爵：反制「刺杀」
"""

from __future__ import annotations

# ===== 角色 =====
DUKE = 0
ASSASSIN = 1
CAPTAIN = 2
AMBASSADOR = 3
CONTESSA = 4

NUM_ROLES = 5
ROLE_COPIES = 3          # 每种角色 3 张
DECK_SIZE = NUM_ROLES * ROLE_COPIES  # 15

ROLE_NAMES: dict[int, str] = {
    DUKE: "公爵",
    ASSASSIN: "刺客",
    CAPTAIN: "队长",
    AMBASSADOR: "大使",
    CONTESSA: "女伯爵",
}

# ===== 金币 / 费用 =====
INITIAL_COINS = 2
INCOME_GAIN = 1
FOREIGN_AID_GAIN = 2
TAX_GAIN = 3
STEAL_GAIN = 2
ASSASSINATE_COST = 3
COUP_COST = 7
FORCED_COUP_COINS = 10     # 金币 >= 此值必须政变

# ===== 阶段 =====
PHASE_ACTION = 0           # 宣告行动
PHASE_CHALLENGE = 1        # 质疑（顺时针依次）
PHASE_BLOCK = 2            # 反制宣告
PHASE_BLOCK_CHALLENGE = 3  # 质疑反制
PHASE_REVEAL = 4           # 扣血：翻开弃掉一张手牌
PHASE_EXCHANGE = 5         # 大使换牌：抽 2 留 2
NUM_PHASES = 6

PHASE_NAMES: dict[int, str] = {
    PHASE_ACTION: "action",
    PHASE_CHALLENGE: "challenge",
    PHASE_BLOCK: "block",
    PHASE_BLOCK_CHALLENGE: "block_challenge",
    PHASE_REVEAL: "reveal",
    PHASE_EXCHANGE: "exchange",
}

# ===== 事件类型（序列编码 + 富日志）=====
EV_INCOME = 0
EV_FOREIGN_AID = 1
EV_COUP = 2
EV_CLAIM = 3        # 宣称角色行动（role=宣称的角色）
EV_CHALLENGE = 4    # 质疑（role=-1）
EV_BLOCK = 5        # 反制（role=反制宣称角色）
EV_REVEAL = 6       # 扣血翻牌弃掉（role=翻开的角色）
EV_SWAP = 7         # 质疑失败后洗回重抽（role=证明的角色）
EV_EXCHANGE = 8     # 换牌（role=-1）
NUM_EVENT_TYPES = 9

# ===== 行动类型（用于 pending 编码）=====
ACT_INCOME = 0
ACT_FOREIGN_AID = 1
ACT_COUP = 2
ACT_TAX = 3
ACT_ASSASSINATE = 4
ACT_STEAL = 5
ACT_EXCHANGE = 6
NUM_ACTION_TYPES = 7

# 宣称角色行动 → 对应角色
CLAIM_ROLE_OF_ACTION: dict[str, int] = {
    "tax": DUKE,
    "assassinate": ASSASSIN,
    "steal": CAPTAIN,
    "exchange": AMBASSADOR,
}

# 可被反制的行动 → 允许的反制角色列表
BLOCKABLE: dict[str, list[int]] = {
    "foreign_aid": [DUKE],
    "assassinate": [CONTESSA],
    "steal": [CAPTAIN, AMBASSADOR],
}

# 换牌「从 4 张中保留 2 张」的 6 种组合（C(4,2)，字典序）
EXCHANGE_KEEP_COMBOS: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
]

# ===== 动作空间槽位布局（详见 game.py）=====
# 0..3           income / foreign_aid / tax / exchange
# 4..            coup 目标 (n-1)
# ...            assassinate 目标 (n-1)
# ...            steal 目标 (n-1)
# base+0         pass（质疑）
# base+1         challenge
# base+2         pass_block（不反制）
# base+3..base+6 block 角色（contessa/duke/captain/ambassador）
# base+7..base+8 reveal 槽 0/1
# base+9..base+14 exchange 保留组合
# 总槽位 = 3n + 16


def make_deck() -> list[int]:
    """构造一副完整 15 张牌（未洗牌，按角色排列）。"""
    return [role for role in range(NUM_ROLES) for _ in range(ROLE_COPIES)]


__all__ = [
    "DUKE", "ASSASSIN", "CAPTAIN", "AMBASSADOR", "CONTESSA",
    "NUM_ROLES", "ROLE_COPIES", "DECK_SIZE", "ROLE_NAMES",
    "INITIAL_COINS", "INCOME_GAIN", "FOREIGN_AID_GAIN", "TAX_GAIN",
    "STEAL_GAIN", "ASSASSINATE_COST", "COUP_COST", "FORCED_COUP_COINS",
    "PHASE_ACTION", "PHASE_CHALLENGE", "PHASE_BLOCK", "PHASE_BLOCK_CHALLENGE",
    "PHASE_REVEAL", "PHASE_EXCHANGE", "NUM_PHASES", "PHASE_NAMES",
    "EV_INCOME", "EV_FOREIGN_AID", "EV_COUP", "EV_CLAIM", "EV_CHALLENGE",
    "EV_BLOCK", "EV_REVEAL", "EV_SWAP", "EV_EXCHANGE", "NUM_EVENT_TYPES",
    "ACT_INCOME", "ACT_FOREIGN_AID", "ACT_COUP", "ACT_TAX", "ACT_ASSASSINATE",
    "ACT_STEAL", "ACT_EXCHANGE", "NUM_ACTION_TYPES", "CLAIM_ROLE_OF_ACTION",
    "BLOCKABLE", "EXCHANGE_KEEP_COMBOS", "make_deck",
]
