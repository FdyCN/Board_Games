#!/usr/bin/env python3
"""
诊断训练异常问题

检查:
1. 游戏是否正常结束 (winner 分布)
2. 熵计算是否正确
3. 概率分布是否有数值问题
"""

import sys
from pathlib import Path
import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from games.splendor import SplendorGame
from models.model_factory import create_splendor_model
from agents.neural_agent import NeuralAgent
from training.self_play_worker import collect_episode


def test_episode_winners():
    """测试 episode 的 winner 分布"""
    print("=" * 60)
    print("测试 1: 检查 Episode Winner 分布")
    print("=" * 60)

    game = SplendorGame(num_players=4, seed=42)
    model = create_splendor_model(encoder_type="mlp", config="medium")
    agents = [NeuralAgent(model=model) for _ in range(4)]

    winners = []
    for i in range(20):
        episode = collect_episode(
            game=game,
            agents=agents,
            deterministic=False,
            verbose=False,
        )
        winners.append(episode.winner)
        print(f"Episode {i+1}: winner={episode.winner}, steps={episode.num_steps}, rewards={episode.total_rewards}")

    print(f"\nWinner 分布:")
    print(f"  玩家 0 获胜: {winners.count(0)}/20 ({winners.count(0)/20:.1%})")
    print(f"  玩家 1 获胜: {winners.count(1)}/20 ({winners.count(1)/20:.1%})")
    print(f"  玩家 2 获胜: {winners.count(2)}/20 ({winners.count(2)/20:.1%})")
    print(f"  玩家 3 获胜: {winners.count(3)}/20 ({winners.count(3)/20:.1%})")
    print(f"  平局 (-1): {winners.count(-1)}/20 ({winners.count(-1)/20:.1%})")

    total_wins = winners.count(0) + winners.count(1) + winners.count(2) + winners.count(3)
    print(f"\n总计有胜者的游戏: {total_wins}/20 ({total_wins/20:.1%})")


def test_entropy_calculation():
    """测试熵计算"""
    print("\n" + "=" * 60)
    print("测试 2: 检查熵计算")
    print("=" * 60)

    game = SplendorGame(num_players=4, seed=42)
    model = create_splendor_model(encoder_type="mlp", config="medium")
    model.eval()

    # 获取一个观察
    state = game.reset()
    observation = game.state_to_observation(state, 0)
    legal_actions = game.get_legal_actions(state)
    legal_action_indices = [game.action_to_index(a, legal_actions) for a in legal_actions]

    # 创建合法动作掩码
    legal_actions_mask = torch.zeros(game.action_space_size)
    legal_actions_mask[legal_action_indices] = 1.0

    # 获取模型输出
    obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
    legal_mask_tensor = legal_actions_mask.unsqueeze(0)

    with torch.no_grad():
        logits, values = model(obs_tensor, legal_mask_tensor)
        probs = torch.softmax(logits, dim=-1)

        # 计算熵
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        print(f"Logits 统计:")
        print(f"  min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
        print(f"\nProbs 统计:")
        print(f"  min={probs.min().item():.6f}, max={probs.max().item():.6f}, sum={probs.sum().item():.6f}")
        print(f"  非零概率数: {(probs > 0.001).sum().item()}")
        print(f"\nEntropy: {entropy.item():.4f}")

        # 检查是否有异常值
        if entropy.item() < 0:
            print("\n⚠️ 警告: 熵为负值!")
            print(f"检查概率分布:")
            top_probs, top_indices = torch.topk(probs[0], k=10)
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                print(f"  Top {i+1}: 动作 {idx.item()}, 概率 {prob.item():.6f}, log {torch.log(prob).item():.6f}")

        # 测试多个样本
        print(f"\n测试 100 个随机状态的熵:")
        entropies = []
        for _ in range(100):
            state = game.reset()
            obs = game.state_to_observation(state, 0)
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                entropies.append(entropy.item())

        entropies = np.array(entropies)
        print(f"  熵统计: min={entropies.min():.4f}, max={entropies.max():.4f}, mean={entropies.mean():.4f}")
        print(f"  负熵数量: {(entropies < 0).sum()}/100")


def test_ppo_entropy():
    """测试 PPO 更新中的熵计算"""
    print("\n" + "=" * 60)
    print("测试 3: 检查 PPO 更新中的熵")
    print("=" * 60)

    from training.experience import split_episodes_by_player, ExperienceBatch

    game = SplendorGame(num_players=4, seed=42)
    model = create_splendor_model(encoder_type="mlp", config="medium")
    agents = [NeuralAgent(model=model) for _ in range(4)]

    # 收集一些经验
    print("收集 5 个 episodes...")
    episodes = []
    for i in range(5):
        episode = collect_episode(game, agents, deterministic=False, verbose=False)
        episodes.append(episode)
        print(f"  Episode {i+1}: {len(episode.experiences)} 经验, winner={episode.winner}")

    # 分割经验
    player_experiences = split_episodes_by_player(episodes)

    # 合并所有玩家的经验
    all_experiences = []
    for player_exps in player_experiences:
        all_experiences.extend(player_exps)

    # 创建批次
    batch = ExperienceBatch.from_experiences(
        all_experiences,
        device="cpu",
        normalize_advantages=True,
    )

    print(f"\n经验批次大小: {batch.batch_size}")

    # 使用模型评估动作
    model.train()
    obs = batch.observations if isinstance(batch.observations, torch.Tensor) else torch.from_numpy(batch.observations).float()
    actions = batch.actions if isinstance(batch.actions, torch.Tensor) else torch.from_numpy(batch.actions).long()

    values, log_probs, entropy = model.evaluate_actions(obs, actions)

    print(f"\nEntropy 统计:")
    print(f"  min={entropy.min().item():.4f}, max={entropy.max().item():.4f}, mean={entropy.mean().item():.4f}")
    print(f"  负熵数量: {(entropy < 0).sum().item()}/{batch.batch_size}")

    if (entropy < 0).any():
        print(f"\n⚠️ 警告: 发现负熵值!")
        neg_indices = (entropy < 0).nonzero(as_tuple=True)[0][:5]  # 显示前 5 个
        print(f"负熵样本索引: {neg_indices.tolist()}")
        for idx in neg_indices:
            print(f"\n  样本 {idx.item()}:")
            print(f"    observation: {obs[idx][:10].tolist()}...")  # 显示前 10 维
            print(f"    action: {actions[idx].item()}")
            print(f"    entropy: {entropy[idx].item():.4f}")


def main():
    print("开始诊断训练异常问题...\n")

    test_episode_winners()
    test_entropy_calculation()
    test_ppo_entropy()

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
