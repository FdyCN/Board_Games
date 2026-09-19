"""
Splendor 游戏引擎单元测试
"""

import pytest
import numpy as np

from games.splendor import (
    SplendorGame,
    TakeGemsAction,
    ReserveCardAction,
    BuyCardAction,
    GemColor,
    CardTier,
    GamePhase,
    create_take_three_different,
    create_take_two_same,
)
from core.exceptions import IllegalActionError


class TestSplendorGameInit:
    """测试游戏初始化"""

    def test_init_valid_players(self):
        """测试有效的玩家数量"""
        for num_players in [2, 3, 4]:
            game = SplendorGame(num_players=num_players)
            assert game.num_players == num_players

    def test_init_invalid_players(self):
        """测试无效的玩家数量"""
        with pytest.raises(ValueError):
            SplendorGame(num_players=1)
        with pytest.raises(ValueError):
            SplendorGame(num_players=5)

    def test_game_properties(self):
        """测试游戏属性"""
        game = SplendorGame(num_players=4)
        assert game.num_players == 4
        assert game.observation_shape == (384,)
        assert game.action_space_size == 46


class TestSplendorGameReset:
    """测试游戏重置"""

    def test_reset_creates_initial_state(self):
        """测试重置创建初始状态"""
        game = SplendorGame(num_players=4)
        state = game.reset()

        assert state is not None
        assert state.num_players == 4
        assert len(state.players) == 4
        assert state.current_player == 0
        assert state.turn_number == 0
        assert state.phase == GamePhase.PLAYING

    def test_reset_initializes_gems(self):
        """测试重置初始化宝石堆"""
        game = SplendorGame(num_players=4)
        state = game.reset()

        # 4 人游戏每种颜色 7 个
        for i in range(5):  # 5 种基础颜色
            assert state.gem_bank[i] == 7

        # 金宝石固定 5 个
        assert state.gem_bank[GemColor.GOLD] == 5

    def test_reset_initializes_cards(self):
        """测试重置初始化卡牌"""
        game = SplendorGame(num_players=4)
        state = game.reset()

        # 每个等级 4 张公开卡
        for tier in [CardTier.TIER_1, CardTier.TIER_2, CardTier.TIER_3]:
            assert len(state.open_cards[tier]) == 4

        # 确保有牌堆
        assert len(state.decks[CardTier.TIER_1]) == 36  # 40 - 4
        assert len(state.decks[CardTier.TIER_2]) == 26  # 30 - 4
        assert len(state.decks[CardTier.TIER_3]) == 16  # 20 - 4

    def test_reset_initializes_nobles(self):
        """测试重置初始化贵族"""
        game = SplendorGame(num_players=4)
        state = game.reset()

        # 4 人游戏 5 张贵族
        assert len(state.nobles) == 5

    def test_reset_players_start_empty(self):
        """测试玩家初始状态为空"""
        game = SplendorGame(num_players=4)
        state = game.reset()

        for player in state.players:
            assert player.total_gems() == 0
            assert len(player.cards) == 0
            assert len(player.reserved_cards) == 0
            assert len(player.nobles) == 0
            assert player.get_score() == 0


class TestSplendorGameActions:
    """测试游戏动作"""

    def test_take_three_different_gems(self):
        """测试拿 3 个不同颜色宝石"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 拿 3 个不同颜色
        action = create_take_three_different([GemColor.RED, GemColor.GREEN, GemColor.BLUE])
        new_state, rewards, done, info = game.step(action)

        # 检查玩家宝石
        assert new_state.players[0].gems[GemColor.RED] == 1
        assert new_state.players[0].gems[GemColor.GREEN] == 1
        assert new_state.players[0].gems[GemColor.BLUE] == 1
        assert new_state.players[0].total_gems() == 3

        # 检查宝石堆
        assert new_state.gem_bank[GemColor.RED] == 6
        assert new_state.gem_bank[GemColor.GREEN] == 6
        assert new_state.gem_bank[GemColor.BLUE] == 6

        # 检查游戏状态
        assert new_state.current_player == 1
        assert new_state.turn_number == 1
        assert not done

    def test_take_two_same_gems(self):
        """测试拿 2 个相同颜色宝石"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 拿 2 个红宝石
        action = create_take_two_same(GemColor.RED)
        new_state, rewards, done, info = game.step(action)

        # 检查玩家宝石
        assert new_state.players[0].gems[GemColor.RED] == 2
        assert new_state.players[0].total_gems() == 2

        # 检查宝石堆
        assert new_state.gem_bank[GemColor.RED] == 5

    def test_reserve_card_from_open(self):
        """测试从公开卡保留"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 获取第一张公开卡
        card = state.open_cards[CardTier.TIER_1][0]
        action = ReserveCardAction(tier=CardTier.TIER_1, card_id=card.card_id)

        new_state, rewards, done, info = game.step(action)

        # 检查保留卡
        assert len(new_state.players[0].reserved_cards) == 1
        assert new_state.players[0].reserved_cards[0].card_id == card.card_id

        # 检查金宝石
        assert new_state.players[0].gems[GemColor.GOLD] == 1

        # 检查公开卡被补充
        assert len(new_state.open_cards[CardTier.TIER_1]) == 4

    def test_buy_card(self):
        """测试购买卡牌"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 找一张便宜的卡（3 个宝石）
        cheap_card = None
        for card in state.open_cards[CardTier.TIER_1]:
            if sum(card.cost) == 3:
                cheap_card = card
                break

        if cheap_card:
            # 给玩家足够的宝石
            player = state.players[0]
            for i, cost in enumerate(cheap_card.cost):
                player.gems[i] = cost

            action = BuyCardAction(card_id=cheap_card.card_id, from_reserved=False)
            new_state, rewards, done, info = game.step(action)

            # 检查卡牌被购买
            assert len(new_state.players[0].cards) == 1
            assert new_state.players[0].cards[0].card_id == cheap_card.card_id

    def test_illegal_action_raises_error(self):
        """测试非法动作抛出异常"""
        game = SplendorGame(num_players=2, seed=42)
        state = game.reset()

        # 尝试拿 2 个相同颜色，但宝石堆只有 4 个（2 人游戏）
        # 实际上 2 人游戏时宝石堆只有 4 个，所以可以拿 2 个
        # 让我们尝试购买一张买不起的卡
        expensive_card = state.open_cards[CardTier.TIER_3][0]
        action = BuyCardAction(card_id=expensive_card.card_id, from_reserved=False)

        with pytest.raises(IllegalActionError):
            game.step(action)


class TestSplendorGameLegalActions:
    """测试合法动作生成"""

    def test_get_legal_actions_initial_state(self):
        """测试初始状态的合法动作"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        legal_actions = game.get_legal_actions(state)

        # 初始状态应该有拿宝石和保留卡的动作
        assert len(legal_actions) > 0

        # 应该有拿 3 个不同颜色的动作
        take_three_actions = [a for a in legal_actions if isinstance(a, TakeGemsAction)]
        assert len(take_three_actions) > 0

        # 应该有拿 2 个相同颜色的动作
        take_two_actions = [
            a for a in legal_actions if isinstance(a, TakeGemsAction) and max(a.gems) == 2
        ]
        assert len(take_two_actions) == 5  # 5 种颜色

        # 应该有保留卡动作
        reserve_actions = [a for a in legal_actions if isinstance(a, ReserveCardAction)]
        assert len(reserve_actions) > 0

    def test_no_legal_actions_when_terminal(self):
        """测试游戏结束时没有合法动作"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()
        state.phase = GamePhase.ENDED

        legal_actions = game.get_legal_actions(state)
        assert len(legal_actions) == 0


class TestSplendorGameNobleVisit:
    """测试贵族拜访机制"""

    def test_noble_visits_when_requirements_met(self):
        """测试满足条件时贵族拜访"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 找一个贵族
        if state.nobles:
            noble = state.nobles[0]
            player = state.players[0]

            # 给玩家足够的卡牌加成
            from games.splendor.cards import DevelopmentCard

            for i, req in enumerate(noble.requirements):
                for _ in range(req):
                    # 创建一张提供对应加成的卡
                    card = DevelopmentCard(
                        card_id=1000 + i,
                        tier=CardTier.TIER_1,
                        points=0,
                        bonus_color=GemColor(i),
                        cost=(0, 0, 0, 0, 0),
                    )
                    player.cards.append(card)

            # 执行任意动作触发贵族检查
            action = create_take_three_different([GemColor.RED, GemColor.GREEN, GemColor.BLUE])
            new_state, rewards, done, info = game.step(action)

            # 检查贵族是否拜访
            assert len(new_state.players[0].nobles) == 1
            assert "noble_visit" in info


class TestSplendorGameWinCondition:
    """测试胜利条件"""

    def test_game_ends_when_player_reaches_15_points(self):
        """测试玩家达到 15 分时游戏进入最后一轮"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 给玩家足够的分数
        from games.splendor.cards import DevelopmentCard

        player = state.players[0]
        for i in range(3):
            card = DevelopmentCard(
                card_id=2000 + i,
                tier=CardTier.TIER_3,
                points=5,
                bonus_color=GemColor.RED,
                cost=(0, 0, 0, 0, 0),
            )
            player.cards.append(card)

        # 玩家应该有 15 分
        assert player.get_score() == 15

        # 执行动作
        action = create_take_three_different([GemColor.RED, GemColor.GREEN, GemColor.BLUE])
        new_state, rewards, done, info = game.step(action)

        # 应该触发最后一轮
        assert new_state.phase == GamePhase.FINAL_ROUND
        assert "final_round_triggered" in info


class TestSplendorGameStateEncoding:
    """测试状态编码"""

    def test_state_to_observation_shape(self):
        """测试观察向量形状"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        obs = game.state_to_observation(state, player_id=0)

        assert obs.shape == (384,)
        assert obs.dtype == np.float32

    def test_state_to_observation_normalized(self):
        """测试观察向量归一化"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        obs = game.state_to_observation(state, player_id=0)

        # 所有值应该在 [0, 1] 范围内
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_state_to_observation_different_players(self):
        """测试不同玩家的观察向量不同"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 给玩家 0 一些宝石
        state.players[0].gems[GemColor.RED] = 3

        obs0 = game.state_to_observation(state, player_id=0)
        obs1 = game.state_to_observation(state, player_id=1)

        # 观察向量应该不同
        assert not np.array_equal(obs0, obs1)


class TestSplendorGameClone:
    """测试游戏克隆"""

    def test_clone_creates_independent_copy(self):
        """测试克隆创建独立副本"""
        game1 = SplendorGame(num_players=4, seed=42)
        state1 = game1.reset()

        game2 = game1.clone()

        # 修改 game1
        action = create_take_three_different([GemColor.RED, GemColor.GREEN, GemColor.BLUE])
        game1.step(action)

        # game2 不应该受影响
        assert game2._state.turn_number == 0
        assert game2._state.players[0].total_gems() == 0


class TestSplendorDenseRewards:
    """测试稠密奖励系统"""

    def test_take_gems_reward(self):
        """测试拿宝石的稠密奖励"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 拿 3 个不同颜色宝石
        action = create_take_three_different([GemColor.RED, GemColor.GREEN, GemColor.BLUE])
        new_state, rewards, done, info = game.step(action)

        # 应该获得 0.01 * 3 = 0.03 的奖励
        assert rewards[0] == pytest.approx(0.03, rel=1e-5)
        assert info["dense_reward"] == pytest.approx(0.03, rel=1e-5)

    def test_take_two_same_gems_reward(self):
        """测试拿 2 个相同颜色宝石的奖励"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 拿 2 个红宝石
        action = create_take_two_same(GemColor.RED)
        new_state, rewards, done, info = game.step(action)

        # 应该获得 0.01 * 2 = 0.02 的奖励
        assert rewards[0] == pytest.approx(0.02, rel=1e-5)

    def test_reserve_card_reward(self):
        """测试保留卡牌的稠密奖励"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 获取第一张公开卡
        card = state.open_cards[CardTier.TIER_1][0]
        action = ReserveCardAction(tier=CardTier.TIER_1, card_id=card.card_id)

        new_state, rewards, done, info = game.step(action)

        # 应该获得 0.02 (保留) + 0.03 (金宝石) = 0.05 的奖励
        assert rewards[0] == pytest.approx(0.05, rel=1e-5)
        assert info["got_gold"] is True

    def test_reserve_card_without_gold(self):
        """测试没有金宝石时保留卡牌的奖励"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 清空金宝石
        state.gem_bank[GemColor.GOLD] = 0

        card = state.open_cards[CardTier.TIER_1][0]
        action = ReserveCardAction(tier=CardTier.TIER_1, card_id=card.card_id)

        new_state, rewards, done, info = game.step(action)

        # 只有保留奖励 0.02
        assert rewards[0] == pytest.approx(0.02, rel=1e-5)
        assert info["got_gold"] is False

    def test_buy_card_reward(self):
        """测试购买卡牌的稠密奖励"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 找一张有分数的便宜卡
        target_card = None
        for card in state.open_cards[CardTier.TIER_1]:
            if card.points > 0:
                target_card = card
                break

        if target_card is None:
            # 如果没找到，使用第一张卡
            target_card = state.open_cards[CardTier.TIER_1][0]

        # 给玩家足够的宝石
        player = state.players[0]
        for i, cost in enumerate(target_card.cost):
            player.gems[i] = cost

        action = BuyCardAction(card_id=target_card.card_id, from_reserved=False)
        new_state, rewards, done, info = game.step(action)

        # 奖励 = 0.15 * points + 0.05 (加成)
        expected_reward = 0.15 * target_card.points + 0.05
        assert rewards[0] == pytest.approx(expected_reward, rel=1e-5)

    def test_noble_visit_reward(self):
        """测试获得贵族的稠密奖励"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 找一个贵族并给玩家足够的卡牌加成
        if state.nobles:
            noble = state.nobles[0]
            player = state.players[0]

            from games.splendor.cards import DevelopmentCard

            for i, req in enumerate(noble.requirements):
                for _ in range(req):
                    card = DevelopmentCard(
                        card_id=1000 + i,
                        tier=CardTier.TIER_1,
                        points=0,
                        bonus_color=GemColor(i),
                        cost=(0, 0, 0, 0, 0),
                    )
                    player.cards.append(card)

            # 执行拿宝石动作触发贵族检查
            action = create_take_three_different([GemColor.RED, GemColor.GREEN, GemColor.BLUE])
            new_state, rewards, done, info = game.step(action)

            # 奖励 = 0.03 (拿宝石) + 0.3 (贵族)
            expected_reward = 0.03 + 0.3
            assert rewards[0] == pytest.approx(expected_reward, rel=1e-5)
            assert "noble_visit" in info

    def test_discard_gems_penalty(self):
        """测试丢弃宝石的惩罚"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 给玩家 9 个宝石（接近上限）
        player = state.players[0]
        player.gems[GemColor.RED] = 3
        player.gems[GemColor.GREEN] = 3
        player.gems[GemColor.BLUE] = 3  # 总共 9 个

        # 拿 3 个宝石，会超过 10 个上限，需要丢弃 2 个
        action = create_take_three_different([GemColor.WHITE, GemColor.BLACK, GemColor.RED])
        new_state, rewards, done, info = game.step(action)

        # 玩家现在应该有 10 个宝石
        assert new_state.players[0].total_gems() == 10
        # 丢弃了 2 个宝石
        assert info["gems_discarded"] == 2

        # 奖励 = 0.01 * 3 (拿取) + (-0.05) * 2 (丢弃) = 0.03 - 0.1 = -0.07
        expected_reward = 0.01 * 3 + (-0.05) * 2
        assert rewards[0] == pytest.approx(expected_reward, rel=1e-5)
        assert rewards[0] < 0  # 应该是负奖励

    def test_win_reward_stacks_with_dense(self):
        """测试胜利奖励与稠密奖励叠加"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        # 给玩家 3 足够的分数
        from games.splendor.cards import DevelopmentCard

        player = state.players[3]
        for i in range(3):
            card = DevelopmentCard(
                card_id=2000 + i,
                tier=CardTier.TIER_3,
                points=5,
                bonus_color=GemColor.RED,
                cost=(0, 0, 0, 0, 0),
            )
            player.cards.append(card)

        # 进入最后一轮，且玩家 3 已是本轮最后一位行动者（行动后游戏结束）
        state.phase = GamePhase.FINAL_ROUND
        state.current_player = 3
        state.final_round_turns_remaining = 0

        # 执行动作
        action = create_take_three_different([GemColor.RED, GemColor.GREEN, GemColor.BLUE])
        new_state, rewards, done, info = game.step(action)

        # 游戏应该结束，玩家 3 获得稠密奖励 + 胜利奖励
        assert done
        # 稠密奖励 0.03 + 胜利奖励 1.0
        assert rewards[3] == pytest.approx(1.03, rel=1e-5)


class TestSplendorInfoFields:
    """测试 info 字段的完整性"""

    def test_info_contains_dense_reward(self):
        """测试 info 包含 dense_reward 字段"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        action = create_take_three_different([GemColor.RED, GemColor.GREEN, GemColor.BLUE])
        new_state, rewards, done, info = game.step(action)

        assert "dense_reward" in info
        assert isinstance(info["dense_reward"], float)

    def test_info_contains_gems_discarded(self):
        """测试拿宝石动作的 info 包含 gems_discarded"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        action = create_take_three_different([GemColor.RED, GemColor.GREEN, GemColor.BLUE])
        new_state, rewards, done, info = game.step(action)

        assert "gems_discarded" in info
        assert info["gems_discarded"] == 0  # 正常情况不丢弃

    def test_info_contains_got_gold(self):
        """测试保留卡牌动作的 info 包含 got_gold"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        card = state.open_cards[CardTier.TIER_1][0]
        action = ReserveCardAction(tier=CardTier.TIER_1, card_id=card.card_id)

        new_state, rewards, done, info = game.step(action)

        assert "got_gold" in info
        assert info["got_gold"] is True


class TestSplendorGameRender:
    """测试游戏渲染"""

    def test_render_returns_string(self):
        """测试渲染返回字符串"""
        game = SplendorGame(num_players=4, seed=42)
        state = game.reset()

        output = game.render(state)

        assert isinstance(output, str)
        assert len(output) > 0
        assert "Splendor" in output


class TestSplendorGameIntegration:
    """集成测试"""

    def test_full_game_playthrough(self):
        """测试完整游戏流程"""
        game = SplendorGame(num_players=2, seed=42)
        state = game.reset()

        max_turns = 100
        turn_count = 0

        while not state.is_terminal() and turn_count < max_turns:
            legal_actions = game.get_legal_actions(state)
            assert len(legal_actions) > 0

            # 随机选择一个动作
            import random

            action = random.choice(legal_actions)

            state, rewards, done, info = game.step(action)
            turn_count += 1

            if done:
                # 应该有赢家
                winner = state.get_winner()
                assert 0 <= winner < game.num_players
                # 赢家应该得到 >= 1.0 奖励（1.0 胜利奖励 + 稠密奖励）
                assert rewards[winner] >= 1.0
                break

    def test_game_registration(self):
        """测试游戏注册"""
        from games.registry import create_game, list_games, unregister_game, is_registered

        # 如果已注册则先取消（避免其他测试影响）
        if is_registered("splendor"):
            unregister_game("splendor")

        # 重新导入以触发注册
        import games.splendor.game
        import importlib
        importlib.reload(games.splendor.game)

        # Splendor 应该已注册
        assert "splendor" in list_games()

        # 应该可以创建游戏
        game = create_game("splendor", num_players=4)
        assert game.__class__.__name__ == "SplendorGame"
        assert game.num_players == 4


# ===== 运行测试 =====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
