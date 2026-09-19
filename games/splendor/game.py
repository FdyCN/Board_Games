"""
Splendor 游戏引擎

完整实现 Splendor 游戏规则，包括：
- 游戏状态初始化
- 动作执行
- 合法动作生成
- 贵族拜访机制
- 游戏结束判定
"""

import random
from copy import deepcopy
from typing import Any

import numpy as np

from core.game_interface import GameInterface
from core.exceptions import IllegalActionError
from games.registry import register_game
from games.splendor.actions import (
    BuyCardAction,
    PassAction,
    ReserveCardAction,
    SplendorAction,
    TakeGemsAction,
)
from games.splendor.cards import (
    DevelopmentCard,
    NobleTile,
    load_development_cards,
    load_nobles,
)
from games.splendor.constants import (
    BASE_GEM_COLORS,
    CardTier,
    GamePhase,
    GemColor,
    MAX_GEMS_IN_HAND,
    MAX_RESERVED_CARDS,
    NUM_GEM_COLORS,
    NUM_GOLD_GEMS,
    NUM_OPEN_CARDS_PER_TIER,
    WINNING_SCORE,
    get_gems_count,
    get_nobles_count,
)
from games.splendor.state import PlayerState, SplendorState


@register_game("splendor")
class SplendorGame(GameInterface):
    """
    Splendor 游戏实现

    支持 2-4 名玩家的宝石商人策略游戏
    """

    # 默认奖励系数
    DEFAULT_REWARDS = {
        "take_gem": 0.01,
        "discard_gem": -0.05,
        "reserve_card": 0.02,
        "get_gold": 0.03,
        "buy_card_points": 0.15,
        "buy_card_bonus": 0.05,
        "noble_visit": 0.3,
        "win": 1.0,
        # 每一步的小惩罚（鼓励尽快结束游戏，避免"囤宝石刷奖励"的无限拖延）
        "step_penalty": 0.0,
    }

    # ===== 规范（固定）动作空间 =====
    # 关键：每个动作槽位有**稳定语义**，策略头每个神经元含义固定。
    # 这是参考 alpha-zero-general Splendor 实现（action_size=81）后，
    # 针对本游戏简化出的规范空间。
    #
    # 槽位布局（共 46 个）：
    #   0-4   : 拿 2 个同色宝石（5 种颜色）
    #   5-14  : 拿 3 个不同色宝石（C(5,3)=10 种组合）
    #   15-26 : 保留明牌（3 tier × 4 位置）
    #   27-29 : 从牌堆保留（3 tier）
    #   30-41 : 购买明牌（3 tier × 4 位置）
    #   42-44 : 购买保留牌（3 位置）
    #   45    : pass（空过）
    ACTION_SPACE_SIZE = 46
    PASS_INDEX = 45

    # 拿 3 个不同色的 10 种组合（字典序）
    TAKE_3_COMBOS = [
        (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
        (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
    ]
    _TAKE_3_INDEX = {combo: i for i, combo in enumerate(TAKE_3_COMBOS)}

    def __init__(self, num_players: int = 4, seed: int | None = None, reward_config: dict | None = None):
        """
        初始化游戏

        Args:
            num_players: 玩家数量 (2-4)
            seed: 随机种子
            reward_config: 奖励配置字典（可选，用于 PPO 训练）
        """
        if not 2 <= num_players <= 4:
            raise ValueError(f"玩家数量必须在 2-4 之间，当前: {num_players}")

        self._num_players = num_players
        self._seed = seed
        self._rng = random.Random(seed)

        # 回合上限（总动作数），防止 pass 导致无限对局
        self._max_total_turns = 62 * num_players

        # 设置奖励配置
        self._rewards = self.DEFAULT_REWARDS.copy()
        if reward_config:
            self._rewards.update(reward_config)

        # 加载卡牌数据
        self._all_cards = load_development_cards()
        self._all_nobles = load_nobles()

        # 当前状态
        self._state: SplendorState | None = None

    # ===== 核心游戏接口 =====

    def reset(self) -> SplendorState:
        """
        重置游戏到初始状态

        Returns:
            初始游戏状态
        """
        # 初始化玩家
        players = [PlayerState(player_id=i) for i in range(self._num_players)]

        # 初始化宝石堆
        gems_per_color = get_gems_count(self._num_players)
        gem_bank = [gems_per_color] * NUM_GEM_COLORS + [NUM_GOLD_GEMS]

        # 初始化卡牌
        cards_by_tier = {
            CardTier.TIER_1: [c for c in self._all_cards if c.tier == CardTier.TIER_1],
            CardTier.TIER_2: [c for c in self._all_cards if c.tier == CardTier.TIER_2],
            CardTier.TIER_3: [c for c in self._all_cards if c.tier == CardTier.TIER_3],
        }

        # 洗牌
        for tier_cards in cards_by_tier.values():
            self._rng.shuffle(tier_cards)

        # 放置公开卡牌
        open_cards = {}
        decks = {}
        for tier, tier_cards in cards_by_tier.items():
            open_cards[tier] = tier_cards[:NUM_OPEN_CARDS_PER_TIER]
            decks[tier] = tier_cards[NUM_OPEN_CARDS_PER_TIER:]

        # 随机选择贵族
        nobles_count = get_nobles_count(self._num_players)
        nobles = self._rng.sample(self._all_nobles, nobles_count)

        # 创建状态
        self._state = SplendorState(
            num_players=self._num_players,
            players=players,
            gem_bank=gem_bank,
            nobles=nobles,
            open_cards=open_cards,
            decks=decks,
            current_player=0,
            turn_number=0,
            phase=GamePhase.PLAYING,
        )

        return self._state

    def step(
        self, action: SplendorAction
    ) -> tuple[SplendorState, list[float], bool, dict[str, Any]]:
        """
        执行一个动作

        Args:
            action: 玩家动作

        Returns:
            new_state: 新状态
            rewards: 每个玩家的奖励
            done: 游戏是否结束
            info: 额外信息

        Raises:
            IllegalActionError: 如果动作非法
        """
        if self._state is None:
            raise RuntimeError("游戏未初始化，请先调用 reset()")

        if self._state.is_terminal():
            raise RuntimeError("游戏已结束")

        # 验证动作合法性
        legal_actions = self.get_legal_actions(self._state)
        if not self._is_action_legal(action, legal_actions):
            raise IllegalActionError(f"非法动作: {action}")

        # 快速浅拷贝状态（卡牌/贵族是 frozen 不可变对象，可安全共享引用）
        new_state = self._state.clone()
        current_player = new_state.get_current_player_state()
        player_id = new_state.current_player

        # 初始化稠密奖励（先加上每步小惩罚，鼓励尽快推进游戏）
        dense_reward = self._rewards.get("step_penalty", 0.0)

        # 执行动作
        info = {"action_type": action.action_type, "player_id": player_id}

        if isinstance(action, TakeGemsAction):
            gems_taken, gems_discarded = self._execute_take_gems(new_state, current_player, action)
            info["gems_taken"] = action.gems
            info["gems_discarded"] = gems_discarded

            # 稠密奖励：拿取宝石
            dense_reward += self._rewards["take_gem"] * gems_taken
            # 惩罚：丢弃宝石
            dense_reward += self._rewards["discard_gem"] * gems_discarded

        elif isinstance(action, ReserveCardAction):
            card, got_gold = self._execute_reserve_card(new_state, current_player, action)
            info["card_reserved"] = card.card_id if card else None
            info["got_gold"] = got_gold

            # 稠密奖励：保留卡牌
            dense_reward += self._rewards["reserve_card"]
            # 稠密奖励：获得金宝石
            if got_gold:
                dense_reward += self._rewards["get_gold"]

        elif isinstance(action, BuyCardAction):
            card = self._execute_buy_card(new_state, current_player, action)
            info["card_bought"] = card.card_id

            # 稠密奖励：购买卡牌（根据分数）
            dense_reward += self._rewards["buy_card_points"] * card.points
            # 稠密奖励：获得永久宝石加成
            dense_reward += self._rewards["buy_card_bonus"]

        elif isinstance(action, PassAction):
            # 空过：什么都不做（只有每步小惩罚）
            info["passed"] = True

        # 检查贵族拜访
        visiting_nobles = self._check_noble_visits(new_state, current_player)
        if visiting_nobles:
            # 玩家获得第一个满足条件的贵族
            noble = visiting_nobles[0]
            current_player.nobles.append(noble)
            new_state.nobles.remove(noble)
            info["noble_visit"] = noble.noble_id

            # 稠密奖励：获得贵族
            dense_reward += self._rewards["noble_visit"]

        # 检查游戏结束条件
        # 规则：有人首次达到 15 分后，把当前这一轮走完（所有玩家行动次数相等），
        # 然后游戏结束。之前实现要求"最后一位玩家也必须达到 15 分"才结束，
        # 导致游戏在 FINAL_ROUND 阶段无限拖长，是"100+ 回合"的直接原因之一。
        if current_player.get_score() >= WINNING_SCORE:
            if new_state.phase == GamePhase.PLAYING:
                # 首次达到 15 分：进入最后一轮，记录本轮剩余需要行动的玩家数
                new_state.phase = GamePhase.FINAL_ROUND
                # 本轮还剩多少玩家要行动（含当前触发玩家）：
                # 触发玩家 p 及之后的 p+1...n-1，共 n-p 人
                new_state.final_round_turns_remaining = (
                    self._num_players - new_state.current_player
                )
                info["final_round_triggered"] = True

        # 切换玩家
        new_state.current_player = (new_state.current_player + 1) % self._num_players
        new_state.turn_number += 1

        # 最后一轮：每个玩家行动一次后递减，减到 0 则游戏结束
        if (
            new_state.phase == GamePhase.FINAL_ROUND
            and new_state.final_round_turns_remaining is not None
        ):
            new_state.final_round_turns_remaining -= 1
            if new_state.final_round_turns_remaining <= 0:
                new_state.phase = GamePhase.ENDED

        # 回合数上限：防止"pass 恒可执行"在银行枯竭且无牌可买时形成无限对局。
        # 参考 alpha-zero-general Splendor 实现（max_moves = 62 * num_players）。
        if new_state.turn_number >= self._max_total_turns:
            new_state.phase = GamePhase.ENDED

        # 计算最终奖励（稠密奖励 + 稀疏终局奖励）
        rewards = [0.0] * self._num_players
        rewards[player_id] = dense_reward

        done = new_state.is_terminal()
        if done:
            winner = new_state.get_winner()
            rewards[winner] += self._rewards["win"]  # 胜利奖励叠加到稠密奖励上

        # 记录稠密奖励到 info
        info["dense_reward"] = dense_reward

        self._state = new_state
        return new_state, rewards, done, info

    def get_legal_actions(self, state: SplendorState | None = None) -> list[SplendorAction]:
        """
        获取当前状态下的所有合法动作

        Args:
            state: 游戏状态（None 则使用当前状态）

        Returns:
            合法动作列表
        """
        if state is None:
            state = self._state

        if state is None or state.is_terminal():
            return []

        player = state.get_current_player_state()
        legal_actions = []

        # 1. 拿宝石动作
        legal_actions.extend(self._get_take_gems_actions(state, player))

        # 2. 保留卡牌动作
        legal_actions.extend(self._get_reserve_card_actions(state, player))

        # 3. 购买卡牌动作
        legal_actions.extend(self._get_buy_card_actions(state, player))

        # 4. pass（始终合法，避免"无合法动作"死锁）
        legal_actions.append(PassAction())

        return legal_actions

    def state_to_observation(
        self, state: SplendorState | None = None, player_id: int = 0
    ) -> np.ndarray:
        """
        将游戏状态编码为观察向量

        Args:
            state: 游戏状态（None 则使用当前状态）
            player_id: 观察者玩家 ID

        Returns:
            观察向量 (observation_shape,)
        """
        if state is None:
            state = self._state

        if state is None:
            raise RuntimeError("状态为空")

        # 使用状态编码器
        from games.splendor.encoder import SplendorEncoder

        encoder = SplendorEncoder()
        return encoder.encode(state, player_id)

    # ===== 游戏属性 =====

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """观察空间形状 (384 维向量)"""
        # 详细计算见 encoder.py
        return (384,)

    @property
    def action_space_size(self) -> int:
        """
        动作空间大小（固定规范槽位）

        固定为 46 个槽位（见类常量 ACTION_SPACE_SIZE 的注释），
        每个槽位语义稳定，便于策略网络学习稳定动作偏好。
        """
        return self.ACTION_SPACE_SIZE

    @property
    def current_player(self) -> int:
        if self._state is None:
            return 0
        return self._state.current_player

    def get_current_player(self, state: SplendorState | None = None) -> int:
        """
        获取当前玩家 ID

        Args:
            state: 游戏状态

        Returns:
            当前玩家 ID
        """
        if state is None:
            state = self._state
        if state is None:
            return 0
        return state.current_player

    def is_terminal(self, state: SplendorState | None = None) -> bool:
        """
        检查游戏是否结束

        Args:
            state: 游戏状态

        Returns:
            是否结束
        """
        if state is None:
            state = self._state
        if state is None:
            return False
        return state.is_terminal()

    def get_final_rewards(self, state: SplendorState | None = None) -> list[float]:
        """
        获取游戏最终奖励

        Args:
            state: 游戏状态

        Returns:
            每个玩家的最终奖励
        """
        return self.get_result(state)

    def action_to_index(self, action: SplendorAction, state: SplendorState | None = None) -> int:
        """
        将动作对象转换为**规范动作索引**（固定槽位，语义稳定）

        规范动作空间见类常量 ACTION_SPACE_SIZE 的注释。该映射需要 state 来
        确定"明牌/保留牌的位置"（槽位按位置而非 card_id），因此每个槽位的
        含义跨状态恒定。

        Args:
            action: 动作对象
            state: 游戏状态（None 则使用当前状态）

        Returns:
            规范动作索引 (0 到 action_space_size-1)
        """
        if state is None:
            state = self._state
        if state is None:
            raise RuntimeError("状态为空，无法计算规范动作索引")

        player = state.get_current_player_state()

        if isinstance(action, TakeGemsAction):
            gems = action.gems
            # 拿 2 个同色
            for c in range(NUM_GEM_COLORS):
                if gems[c] == 2:
                    return c  # 0-4
            # 拿 3 个不同色
            colors = tuple(sorted(c for c in range(NUM_GEM_COLORS) if gems[c] == 1))
            return 5 + self._TAKE_3_INDEX[colors]  # 5-14

        elif isinstance(action, ReserveCardAction):
            if action.card_id is None:
                # 从牌堆保留
                return 27 + (action.tier - 1)  # 27-29
            # 保留明牌：按位置
            pos = self._find_open_card_position(state, action.tier, action.card_id)
            return 15 + (action.tier - 1) * 4 + pos  # 15-26

        elif isinstance(action, BuyCardAction):
            if action.from_reserved:
                pos = self._find_reserved_card_position(player, action.card_id)
                return 42 + pos  # 42-44
            else:
                # BuyCardAction 不携带 tier，需要跨所有 tier 查找位置
                tier, pos = self._find_card_tier_position(state, action.card_id)
                return 30 + (tier - 1) * 4 + pos  # 30-41

        elif isinstance(action, PassAction):
            return self.PASS_INDEX

        raise ValueError(f"未知动作类型: {type(action)}")

    def index_to_action(self, index: int, state: SplendorState | None = None) -> SplendorAction:
        """
        将**规范动作索引**转换为动作对象

        Args:
            index: 规范动作索引 (0 到 action_space_size-1)
            state: 游戏状态（None 则使用当前状态）

        Returns:
            动作对象

        Raises:
            IndexError: 如果索引超出范围
        """
        if index < 0 or index >= self.ACTION_SPACE_SIZE:
            raise IndexError(f"动作索引 {index} 超出范围 [0, {self.ACTION_SPACE_SIZE})")
        if state is None:
            state = self._state
        if state is None:
            raise RuntimeError("状态为空，无法转换规范动作索引")

        player = state.get_current_player_state()

        if index < 5:
            # 拿 2 个同色
            gems = [0] * NUM_GEM_COLORS
            gems[index] = 2
            return TakeGemsAction(gems=tuple(gems))

        elif index < 15:
            # 拿 3 个不同色
            colors = self.TAKE_3_COMBOS[index - 5]
            gems = [0] * NUM_GEM_COLORS
            for c in colors:
                gems[c] = 1
            return TakeGemsAction(gems=tuple(gems))

        elif index < 27:
            # 保留明牌
            offset = index - 15
            tier = CardTier(offset // 4 + 1)
            pos = offset % 4
            card = state.open_cards[tier][pos]
            return ReserveCardAction(tier=tier, card_id=card.card_id)

        elif index < 30:
            # 从牌堆保留
            tier = CardTier(index - 27 + 1)
            return ReserveCardAction(tier=tier, card_id=None)

        elif index < 42:
            # 购买明牌
            offset = index - 30
            tier = CardTier(offset // 4 + 1)
            pos = offset % 4
            card = state.open_cards[tier][pos]
            return BuyCardAction(card_id=card.card_id, from_reserved=False)

        elif index < 45:
            # 购买保留牌
            pos = index - 42
            card = player.reserved_cards[pos]
            return BuyCardAction(card_id=card.card_id, from_reserved=True)

        else:
            # pass
            return PassAction()

    def _find_open_card_position(self, state: SplendorState, tier: CardTier, card_id: int) -> int:
        """在公开卡牌中查找卡牌位置（用于规范槽位）"""
        for i, c in enumerate(state.open_cards[tier]):
            if c.card_id == card_id:
                return i
        raise ValueError(f"卡牌 {card_id} 不在 tier {tier} 的明牌中")

    def _find_card_tier_position(self, state: SplendorState, card_id: int) -> tuple[int, int]:
        """跨所有 tier 查找卡牌所在 (tier, position)，用于 BuyCardAction"""
        for tier in [CardTier.TIER_1, CardTier.TIER_2, CardTier.TIER_3]:
            for i, c in enumerate(state.open_cards[tier]):
                if c.card_id == card_id:
                    return tier, i
        raise ValueError(f"卡牌 {card_id} 不在明牌中")

    def _find_reserved_card_position(self, player: PlayerState, card_id: int) -> int:
        """在保留卡中查找卡牌位置（用于规范槽位）"""
        for i, c in enumerate(player.reserved_cards):
            if c.card_id == card_id:
                return i
        raise ValueError(f"卡牌 {card_id} 不在保留区中")

    def get_result(self, state: SplendorState | None = None) -> list[float]:
        """
        获取游戏结果（每个玩家的得分）

        Args:
            state: 游戏状态

        Returns:
            每个玩家的得分列表
        """
        if state is None:
            state = self._state

        if state is None or not state.is_terminal():
            return [0.0] * self._num_players

        winner = state.get_winner()
        results = [0.0] * self._num_players
        results[winner] = 1.0
        return results

    def clone(self) -> "SplendorGame":
        """克隆游戏实例"""
        new_game = SplendorGame(
            num_players=self._num_players,
            seed=self._seed,
            reward_config=self._rewards.copy()
        )
        if self._state is not None:
            new_game._state = self._state.clone()
        return new_game

    def render(self, state: SplendorState | None = None) -> str:
        """渲染游戏状态为字符串"""
        if state is None:
            state = self._state

        if state is None:
            return "游戏未初始化"

        lines = []
        lines.append("=" * 60)
        lines.append(f"Splendor - 回合 {state.turn_number} - 当前玩家: {state.current_player}")
        lines.append("=" * 60)

        # 宝石堆
        lines.append("\n💎 宝石堆:")
        from games.splendor.constants import GEM_COLORED_NAMES, GEM_COLORED_SYMBOLS

        for i, count in enumerate(state.gem_bank[:NUM_GEM_COLORS]):
            gem_display = f"{GEM_COLORED_SYMBOLS[GemColor(i)]} {GEM_COLORED_NAMES[GemColor(i)]}"
            lines.append(f"  {gem_display}: {count}")
        gold_display = f"{GEM_COLORED_SYMBOLS[GemColor.GOLD]} {GEM_COLORED_NAMES[GemColor.GOLD]}"
        lines.append(f"  {gold_display}: {state.gem_bank[GemColor.GOLD]}")

        # 贵族
        lines.append(f"\n👑 贵族卡 ({len(state.nobles)} 张):")
        for noble in state.nobles:
            lines.append(f"  {noble}")

        # 公开卡牌
        lines.append("\n📋 公开卡牌:")
        for tier in [CardTier.TIER_3, CardTier.TIER_2, CardTier.TIER_1]:
            cards = state.open_cards.get(tier, [])
            lines.append(f"  Tier {tier}: {len(cards)} 张")
            for card in cards:
                lines.append(f"    {card}")

        # 玩家状态
        lines.append("\n👥 玩家状态:")
        for player in state.players:
            marker = "→" if player.player_id == state.current_player else " "
            lines.append(
                f"{marker} 玩家 {player.player_id}: "
                f"{player.get_score()}分 | "
                f"宝石:{player.total_gems()} | "
                f"卡牌:{len(player.cards)} | "
                f"保留:{len(player.reserved_cards)}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    # ===== 内部辅助方法 =====

    def _execute_take_gems(self, state: SplendorState, player: PlayerState, action: TakeGemsAction) -> tuple[int, int]:
        """
        执行拿宝石动作

        Returns:
            (gems_taken, gems_discarded): 拿取的宝石数量和丢弃的宝石数量
        """
        gems_taken = 0
        for color in range(NUM_GEM_COLORS):
            gems_to_take = action.get_gem_count(GemColor(color))
            if gems_to_take > 0:
                player.gems[color] += gems_to_take
                state.gem_bank[color] -= gems_to_take
                gems_taken += gems_to_take

        # 如果超过 10 个宝石，需要丢弃（这里简化处理，自动丢弃多余的）
        gems_discarded = 0
        while player.total_gems() > MAX_GEMS_IN_HAND:
            # 找到数量最多的宝石并丢弃 1 个
            max_color = max(range(NUM_GEM_COLORS), key=lambda c: player.gems[c])
            if player.gems[max_color] > 0:
                player.gems[max_color] -= 1
                state.gem_bank[max_color] += 1
                gems_discarded += 1

        return gems_taken, gems_discarded

    def _execute_reserve_card(
        self, state: SplendorState, player: PlayerState, action: ReserveCardAction
    ) -> tuple[DevelopmentCard | None, bool]:
        """
        执行保留卡牌动作

        Returns:
            (card, got_gold): 保留的卡牌和是否获得金宝石
        """
        card = None
        got_gold = False

        if action.is_from_deck():
            # 从牌堆顶保留
            deck = state.decks[action.tier]
            if deck:
                card = deck.pop(0)
        else:
            # 从公开卡牌保留
            open_cards = state.open_cards[action.tier]
            for i, c in enumerate(open_cards):
                if c.card_id == action.card_id:
                    card = open_cards.pop(i)
                    # 补充新卡
                    if state.decks[action.tier]:
                        open_cards.append(state.decks[action.tier].pop(0))
                    break

        if card:
            player.reserved_cards.append(card)

            # 获得 1 个金宝石
            if state.gem_bank[GemColor.GOLD] > 0:
                player.gems[GemColor.GOLD] += 1
                state.gem_bank[GemColor.GOLD] -= 1
                got_gold = True

        return card, got_gold

    def _execute_buy_card(
        self, state: SplendorState, player: PlayerState, action: BuyCardAction
    ) -> DevelopmentCard:
        """执行购买卡牌动作"""
        card = None

        if action.is_from_hand():
            # 从保留区购买
            for i, c in enumerate(player.reserved_cards):
                if c.card_id == action.card_id:
                    card = player.reserved_cards.pop(i)
                    break
        else:
            # 从公开卡牌购买
            for tier, open_cards in state.open_cards.items():
                for i, c in enumerate(open_cards):
                    if c.card_id == action.card_id:
                        card = open_cards.pop(i)
                        # 补充新卡
                        if state.decks[tier]:
                            open_cards.append(state.decks[tier].pop(0))
                        break
                if card:
                    break

        if not card:
            raise IllegalActionError(f"找不到卡牌 {action.card_id}")

        # 计算支付
        payment = self._calculate_payment(player, card)

        # 扣除宝石并归还银行（包括金宝石）
        for color in range(NUM_GEM_COLORS + 1):  # 包括金色
            if payment[color] > 0:
                player.gems[color] -= payment[color]
                state.gem_bank[color] += payment[color]

        # 添加卡牌到玩家手中
        player.cards.append(card)

        return card

    def _calculate_payment(self, player: PlayerState, card: DevelopmentCard) -> list[int]:
        """计算购买卡牌的支付方案"""
        payment = [0] * (NUM_GEM_COLORS + 1)  # [R,G,B,W,Bl,Gold]
        bonuses = player.get_total_bonuses()

        gold_needed = 0

        for color in range(NUM_GEM_COLORS):
            cost = card.cost[color]
            bonus = bonuses[color]
            needed = max(0, cost - bonus)

            # 先用对应颜色的宝石
            paid_from_color = min(needed, player.gems[color])
            payment[color] = paid_from_color
            needed -= paid_from_color

            # 剩余用金宝石
            gold_needed += needed

        payment[GemColor.GOLD] = gold_needed
        return payment

    def _check_noble_visits(self, state: SplendorState, player: PlayerState) -> list[NobleTile]:
        """检查哪些贵族会拜访玩家"""
        visiting = []
        bonuses = player.get_total_bonuses()

        for noble in state.nobles:
            if all(bonuses[color] >= noble.requirements[color] for color in range(NUM_GEM_COLORS)):
                visiting.append(noble)

        return visiting

    def _get_take_gems_actions(
        self, state: SplendorState, player: PlayerState
    ) -> list[TakeGemsAction]:
        """生成所有拿宝石的合法动作（仅标准动作，配合规范动作空间）"""
        actions = []

        # 1. 拿 3 个不同颜色
        available_colors = [c for c in BASE_GEM_COLORS if state.gem_bank[c] > 0]
        if len(available_colors) >= 3:
            from itertools import combinations

            for colors in combinations(available_colors, 3):
                gems = [0] * NUM_GEM_COLORS
                for color in colors:
                    gems[color] = 1
                actions.append(TakeGemsAction(gems=tuple(gems)))

        # 2. 拿 2 个相同颜色（宝石堆中至少有 4 个）
        for color in BASE_GEM_COLORS:
            if state.gem_bank[color] >= 4:
                gems = [0] * NUM_GEM_COLORS
                gems[color] = 2
                actions.append(TakeGemsAction(gems=tuple(gems)))

        # 注：移除了"拿 2 个不同 / 拿 1 个"的退化兜底动作，
        # 它们不是标准 Splendor 动作且无法映射到稳定槽位。
        # 银行枯竭时由始终合法的 pass 动作兜底，避免死锁。

        return actions

    def _get_reserve_card_actions(
        self, state: SplendorState, player: PlayerState
    ) -> list[ReserveCardAction]:
        """生成所有保留卡牌的合法动作"""
        if len(player.reserved_cards) >= MAX_RESERVED_CARDS:
            return []

        actions = []

        # 保留公开卡牌
        for tier, open_cards in state.open_cards.items():
            for card in open_cards:
                actions.append(ReserveCardAction(tier=tier, card_id=card.card_id))

        # 从牌堆顶保留
        for tier, deck in state.decks.items():
            if deck:
                actions.append(ReserveCardAction(tier=tier, card_id=None))

        return actions

    def _get_buy_card_actions(
        self, state: SplendorState, player: PlayerState
    ) -> list[BuyCardAction]:
        """生成所有购买卡牌的合法动作"""
        actions = []

        # 购买公开卡牌
        for open_cards in state.open_cards.values():
            for card in open_cards:
                if player.can_afford(card):
                    actions.append(BuyCardAction(card_id=card.card_id, from_reserved=False))

        # 购买保留卡牌
        for card in player.reserved_cards:
            if player.can_afford(card):
                actions.append(BuyCardAction(card_id=card.card_id, from_reserved=True))

        return actions

    def _is_action_legal(self, action: SplendorAction, legal_actions: list[SplendorAction]) -> bool:
        """检查动作是否合法"""
        for legal_action in legal_actions:
            if self._actions_equal(action, legal_action):
                return True
        return False

    def _actions_equal(self, action1: SplendorAction, action2: SplendorAction) -> bool:
        """检查两个动作是否相等"""
        if type(action1) != type(action2):
            return False

        if isinstance(action1, TakeGemsAction):
            return action1.gems == action2.gems
        elif isinstance(action1, ReserveCardAction):
            return action1.tier == action2.tier and action1.card_id == action2.card_id
        elif isinstance(action1, BuyCardAction):
            return (
                action1.card_id == action2.card_id
                and action1.from_reserved == action2.from_reserved
            )
        elif isinstance(action1, PassAction):
            return True
        return False


# ===== 导出 =====
__all__ = ["SplendorGame"]
