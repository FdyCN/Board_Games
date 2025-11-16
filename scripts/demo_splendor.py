"""
Splendor 游戏演示脚本

演示四个随机 Agent 进行完整的游戏对弈
"""

import time
from typing import Optional

from agents.random_agent import RandomAgent
from games.splendor import SplendorGame


def play_game(
    num_players: int = 4,
    seed: Optional[int] = None,
    render: bool = True,
    delay: float = 0.0,
    verbose: bool = True,
) -> dict:
    """
    运行一局完整的 Splendor 游戏

    Args:
        num_players: 玩家数量 (2-4)
        seed: 随机种子
        render: 是否渲染游戏状态
        delay: 每步之间的延迟（秒）
        verbose: 是否输出详细信息

    Returns:
        游戏结果字典
    """
    # 创建游戏
    game = SplendorGame(num_players=num_players, seed=seed)

    # 创建随机 Agents
    agents = [RandomAgent(player_id=i, seed=seed + i if seed else None) for i in range(num_players)]

    # 重置游戏
    state = game.reset()

    if render:
        print("\n" + "=" * 80)
        print("🎮 Splendor 游戏开始")
        print("=" * 80)
        print(game.render(state))
        time.sleep(delay)

    # 游戏循环
    turn_count = 0
    max_turns = 1000  # 防止无限循环

    while not state.is_terminal() and turn_count < max_turns:
        current_player = state.current_player

        # 获取合法动作
        legal_actions = game.get_legal_actions(state)

        if not legal_actions:
            print(f"⚠️  警告: 玩家 {current_player} 没有合法动作！")
            print(f"   回合数: {turn_count}")
            player = state.players[current_player]
            print(f"   玩家宝石: {player.gems} (总计: {player.total_gems()})")
            print(f"   玩家加成: {player.get_total_bonuses()}")
            print(f"   玩家分数: {player.get_score()}")
            print(f"   保留卡: {len(player.reserved_cards)}")
            print(f"   宝石堆: {state.gem_bank}")
            print(f"   公开卡数量: {sum(len(cards) for cards in state.open_cards.values())}")

            # 检查是否有可购买的卡
            affordable_cards = []
            for open_cards in state.open_cards.values():
                for card in open_cards:
                    if player.can_afford(card):
                        affordable_cards.append(card.card_id)
            for card in player.reserved_cards:
                if player.can_afford(card):
                    affordable_cards.append(f"R{card.card_id}")
            print(f"   可购买卡牌: {affordable_cards}")

            # 打印所有玩家的宝石持有情况
            print(f"\n   所有玩家宝石分布:")
            total_in_players = [0] * 6
            for p in state.players:
                print(f"     玩家 {p.player_id}: {p.gems} (总计: {p.total_gems()})")
                for i in range(6):
                    total_in_players[i] += p.gems[i]
            print(f"   玩家总持有: {total_in_players} (总计: {sum(total_in_players)})")
            print(f"   银行剩余: {state.gem_bank} (总计: {sum(state.gem_bank)})")
            print(f"   系统总宝石: {[total_in_players[i] + state.gem_bank[i] for i in range(6)]}")
            break

        # Agent 选择动作
        legal_action_objects = legal_actions
        action_indices = list(range(len(legal_actions)))
        action_index, info = agents[current_player].select_action(
            observation=None, legal_actions=action_indices, deterministic=False
        )
        action = legal_action_objects[action_index]

        if verbose:
            print(f"\n回合 {turn_count + 1} - 玩家 {current_player} 的动作: {action}")

        # 执行动作
        state, rewards, done, step_info = game.step(action)
        turn_count += 1

        # 渲染状态
        if render and (turn_count % 4 == 0 or done):  # 每 4 回合渲染一次，或游戏结束时
            print(game.render(state))
            time.sleep(delay)

        # 显示特殊事件
        if "noble_visit" in step_info:
            print(f"👑 贵族拜访了玩家 {current_player}！")

        if "final_round_triggered" in step_info:
            print(f"🏁 玩家 {current_player} 达到 15 分，进入最后一轮！")

    # 游戏结束
    if state.is_terminal():
        winner = state.get_winner()
        final_scores = [player.get_score() for player in state.players]

        if render:
            print("\n" + "=" * 80)
            print("🏆 游戏结束")
            print("=" * 80)
            print(f"获胜者: 玩家 {winner}")
            print(f"最终分数: {final_scores}")
            print(f"总回合数: {turn_count}")
            print("=" * 80)

        return {
            "winner": winner,
            "scores": final_scores,
            "turns": turn_count,
            "finished": True,
        }
    else:
        print(f"\n⚠️  游戏未完成（达到最大回合数 {max_turns}）")
        return {
            "winner": -1,
            "scores": [player.get_score() for player in state.players],
            "turns": turn_count,
            "finished": False,
        }


def run_multiple_games(num_games: int = 10, num_players: int = 4, seed: int = 42) -> None:
    """
    运行多局游戏并统计结果

    Args:
        num_games: 游戏局数
        num_players: 玩家数量
        seed: 初始随机种子
    """
    print(f"\n{'=' * 80}")
    print(f"🎲 运行 {num_games} 局 {num_players} 人 Splendor 游戏")
    print(f"{'=' * 80}\n")

    results = []
    win_counts = [0] * num_players
    total_turns = 0

    for i in range(num_games):
        print(f"\n--- 第 {i + 1}/{num_games} 局 ---")
        result = play_game(
            num_players=num_players, seed=seed + i, render=False, verbose=False
        )
        results.append(result)

        if result["finished"]:
            win_counts[result["winner"]] += 1
            total_turns += result["turns"]

    # 统计结果
    finished_games = sum(1 for r in results if r["finished"])

    print(f"\n{'=' * 80}")
    print("📊 统计结果")
    print(f"{'=' * 80}")
    print(f"完成游戏数: {finished_games}/{num_games}")
    print(f"平均回合数: {total_turns / finished_games if finished_games > 0 else 0:.1f}")
    print(f"\n各玩家胜率:")
    for i in range(num_players):
        win_rate = win_counts[i] / finished_games * 100 if finished_games > 0 else 0
        print(f"  玩家 {i}: {win_counts[i]:2d} 胜 ({win_rate:5.1f}%)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Splendor 游戏演示")
    parser.add_argument(
        "--players", type=int, default=4, choices=[2, 3, 4], help="玩家数量 (2-4)"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--games", type=int, default=1, help="游戏局数")
    parser.add_argument(
        "--delay", type=float, default=0.0, help="每步延迟（秒）"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="不渲染游戏状态"
    )

    args = parser.parse_args()

    if args.games == 1:
        # 单局游戏，详细显示
        play_game(
            num_players=args.players,
            seed=args.seed,
            render=not args.no_render,
            delay=args.delay,
            verbose=True,
        )
    else:
        # 多局游戏，统计结果
        run_multiple_games(
            num_games=args.games, num_players=args.players, seed=args.seed
        )
