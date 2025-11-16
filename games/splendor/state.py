"""
Splendor 游戏状态

定义游戏状态和玩家状态的数据结构。
"""

from dataclasses import dataclass, field

from games.splendor.cards import DevelopmentCard, NobleTile
from games.splendor.constants import (
    CardTier,
    GamePhase,
    GemColor,
    MAX_GEMS_IN_HAND,
    MAX_RESERVED_CARDS,
    NUM_GEM_COLORS,
)


@dataclass
class PlayerState:
    """
    玩家状态

    Attributes:
        player_id: 玩家 ID
        gems: 持有的宝石 [R, G, B, W, Bl, Gold]
        cards: 已购买的发展卡
        reserved_cards: 保留的卡牌
        nobles: 拥有的贵族
    """

    player_id: int
    gems: list[int] = field(default_factory=lambda: [0] * 6)  # [R,G,B,W,Bl,Gold]
    cards: list[DevelopmentCard] = field(default_factory=list)
    reserved_cards: list[DevelopmentCard] = field(default_factory=list)
    nobles: list[NobleTile] = field(default_factory=list)

    def total_gems(self) -> int:
        """总宝石数"""
        return sum(self.gems)

    def get_gem_count(self, color: GemColor) -> int:
        """获取指定颜色宝石数量"""
        return self.gems[color]

    def get_bonus(self, color: GemColor) -> int:
        """获取指定颜色的卡牌加成"""
        if color == GemColor.GOLD:
            return 0
        return sum(1 for card in self.cards if card.bonus_color == color)

    def get_total_bonuses(self) -> list[int]:
        """获取所有颜色的加成 [R, G, B, W, Bl]"""
        bonuses = [0] * NUM_GEM_COLORS
        for card in self.cards:
            bonuses[card.bonus_color] += 1
        return bonuses

    def get_score(self) -> int:
        """计算总分"""
        card_points = sum(card.points for card in self.cards)
        noble_points = len(self.nobles) * 3
        return card_points + noble_points

    def can_afford(self, card: DevelopmentCard) -> bool:
        """检查是否能购买卡牌"""
        bonuses = self.get_total_bonuses()
        gold_needed = 0

        for color in range(NUM_GEM_COLORS):
            cost = card.cost[color]
            bonus = bonuses[color]
            needed = max(0, cost - bonus)

            # 优先使用对应颜色的宝石
            available_from_color = self.gems[color]
            if needed > available_from_color:
                # 不足的部分需要金宝石
                gold_needed += needed - available_from_color

        # 检查金宝石是否足够
        return gold_needed <= self.gems[GemColor.GOLD]


@dataclass
class SplendorState:
    """
    Splendor 游戏状态

    Attributes:
        num_players: 玩家数量
        players: 玩家状态列表
        gem_bank: 宝石堆 [R, G, B, W, Bl, Gold]
        nobles: 场上的贵族卡
        open_cards: 公开卡牌 {tier: [cards]}
        decks: 卡牌堆 {tier: [cards]}
        current_player: 当前玩家索引
        turn_number: 回合数
        phase: 游戏阶段
    """

    num_players: int
    players: list[PlayerState]
    gem_bank: list[int]
    nobles: list[NobleTile]
    open_cards: dict[CardTier, list[DevelopmentCard]]
    decks: dict[CardTier, list[DevelopmentCard]]
    current_player: int = 0
    turn_number: int = 0
    phase: GamePhase = GamePhase.PLAYING

    def get_current_player_state(self) -> PlayerState:
        """获取当前玩家状态"""
        return self.players[self.current_player]

    def is_terminal(self) -> bool:
        """判断游戏是否结束"""
        return self.phase == GamePhase.ENDED

    def get_winner(self) -> int:
        """获取获胜者"""
        if not self.is_terminal():
            return -1

        max_score = max(p.get_score() for p in self.players)
        winners = [i for i, p in enumerate(self.players) if p.get_score() == max_score]

        if len(winners) == 1:
            return winners[0]

        # 平局：拥有卡牌最少的玩家获胜
        min_cards = min(len(self.players[i].cards) for i in winners)
        for i in winners:
            if len(self.players[i].cards) == min_cards:
                return i

        return winners[0]
