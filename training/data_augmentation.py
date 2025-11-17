"""
数据增强模块

提供位置旋转等数据增强功能，用于减少位置偏差。
"""

import numpy as np
from typing import List
from core.types import Experience


def rotate_splendor_observation(
    obs: np.ndarray,
    rotation: int,
    num_players: int = 4,
) -> np.ndarray:
    """
    旋转 Splendor 观察（改变玩家视角）

    **重要说明**：由于 Splendor 的观察编码是不对称的（当前玩家 60 维 vs 其他玩家 15 维），
    我们采用简化的数据增强方案：**只随机化"当前玩家 ID"字段**，而不完整旋转所有玩家状态。

    这样可以：
    1. 防止模型学习到"我总是在位置 X"
    2. 保持观察的其他信息完整性
    3. 避免不对称编码带来的信息丢失

    观察结构 (384 维，实际使用 326 维)：
    - [0:60]     当前玩家状态
    - [60:105]   其他玩家 (3 × 15 = 45 维)
    - [105:285]  公开卡牌 (12 × 15 = 180 维)
    - [285:315]  贵族 (5 × 6 = 30 维)
    - [315:321]  宝石堆 (6 维)
    - [321:325]  当前玩家 ID (4 维 one-hot) ← **这里被随机化**
    - [325:326]  回合数 (1 维)
    - [326:384]  未使用

    Args:
        obs: 原始观察 (384 维)
        rotation: 旋转步数 (0-3)，0 表示不旋转
        num_players: 玩家数量

    Returns:
        旋转后的观察

    Examples:
        >>> obs = encoder.encode(state, player_id=0)
        >>> # 随机化"当前玩家 ID"字段
        >>> rotated_obs = rotate_splendor_observation(obs, rotation=1)
    """
    if rotation == 0:
        return obs.copy()

    # 检查维度
    if len(obs) != 384:
        raise ValueError(f"Splendor 观察维度应为 384，实际为 {len(obs)}")

    rotated_obs = obs.copy()

    # 只随机化"当前玩家 ID"编码 (321:325)
    # 找到原来的"当前玩家 ID"
    original_current_id = np.argmax(obs[321:325])
    # 新的"当前玩家 ID"应该是 (original - rotation) % num_players
    # 这样模型无法学习到"我总是在位置 X"
    new_current_id = (original_current_id - rotation) % num_players
    rotated_obs[321:325] = 0
    rotated_obs[321 + new_current_id] = 1.0

    return rotated_obs


def augment_experiences_with_rotation(
    experiences: List[Experience],
    num_players: int = 4,
    augmentation_factor: int = 2,
) -> List[Experience]:
    """
    对经验列表进行位置旋转数据增强

    为每个经验生成多个旋转版本，增加训练数据的多样性。

    Args:
        experiences: 原始经验列表
        num_players: 玩家数量
        augmentation_factor: 增强倍数 (1-4)
            - 1: 不增强，返回原始经验
            - 2: 每个经验增强 1 次（随机旋转）
            - 4: 每个经验增强 3 次（生成所有旋转版本）

    Returns:
        增强后的经验列表

    Examples:
        >>> experiences = [exp1, exp2, exp3, ...]
        >>> augmented = augment_experiences_with_rotation(
        ...     experiences, num_players=4, augmentation_factor=2
        ... )
        >>> print(f"原始: {len(experiences)}, 增强后: {len(augmented)}")
    """
    if augmentation_factor == 1:
        return experiences

    augmented_experiences = []

    for exp in experiences:
        # 保留原始经验
        augmented_experiences.append(exp)

        # 生成旋转版本
        if augmentation_factor == 2:
            # 随机旋转 1 次
            rotations = [np.random.randint(1, num_players)]
        elif augmentation_factor >= 4:
            # 生成所有旋转版本 (1, 2, 3)
            rotations = list(range(1, num_players))
        else:
            # augmentation_factor = 3: 随机选择 2 个旋转
            rotations = np.random.choice(
                range(1, num_players), size=augmentation_factor - 1, replace=False
            ).tolist()

        for rotation in rotations:
            # 旋转观察
            rotated_obs = rotate_splendor_observation(
                exp.observation, rotation=rotation, num_players=num_players
            )

            # 旋转 next_observation（如果存在）
            rotated_next_obs = None
            if exp.next_observation is not None:
                rotated_next_obs = rotate_splendor_observation(
                    exp.next_observation, rotation=rotation, num_players=num_players
                )

            # 旋转 legal_actions_mask（如果存在）
            # 注意：Splendor 的动作空间可能是位置无关的，所以不需要旋转
            # 但如果动作包含"针对特定玩家"的操作，则需要调整
            rotated_mask = exp.legal_actions_mask

            # 创建新的经验
            rotated_exp = Experience(
                player_id=exp.player_id,  # 保持玩家 ID 不变（这是训练时的标签）
                observation=rotated_obs,
                action=exp.action,  # 动作索引保持不变（假设动作空间位置无关）
                reward=exp.reward,
                next_observation=rotated_next_obs,
                done=exp.done,
                log_prob=exp.log_prob,
                value=exp.value,
                advantage=exp.advantage,
                returns=exp.returns,
                legal_actions_mask=rotated_mask,
            )

            augmented_experiences.append(rotated_exp)

    return augmented_experiences


def apply_random_rotation_to_batch(
    experiences: List[Experience],
    num_players: int = 4,
    rotation_probability: float = 0.5,
) -> List[Experience]:
    """
    对经验批次应用随机旋转

    以一定概率对每个经验进行随机旋转。
    这是一个轻量级的数据增强方法，不会增加数据量。

    Args:
        experiences: 原始经验列表
        num_players: 玩家数量
        rotation_probability: 旋转概率 (0-1)

    Returns:
        旋转后的经验列表（长度不变）

    Examples:
        >>> experiences = [exp1, exp2, exp3, ...]
        >>> rotated = apply_random_rotation_to_batch(
        ...     experiences, num_players=4, rotation_probability=0.5
        ... )
    """
    rotated_experiences = []

    for exp in experiences:
        # 以一定概率进行旋转
        if np.random.random() < rotation_probability:
            rotation = np.random.randint(1, num_players)

            # 旋转观察
            rotated_obs = rotate_splendor_observation(
                exp.observation, rotation=rotation, num_players=num_players
            )

            # 旋转 next_observation
            rotated_next_obs = None
            if exp.next_observation is not None:
                rotated_next_obs = rotate_splendor_observation(
                    exp.next_observation, rotation=rotation, num_players=num_players
                )

            # 创建新经验
            rotated_exp = Experience(
                player_id=exp.player_id,
                observation=rotated_obs,
                action=exp.action,
                reward=exp.reward,
                next_observation=rotated_next_obs,
                done=exp.done,
                log_prob=exp.log_prob,
                value=exp.value,
                advantage=exp.advantage,
                returns=exp.returns,
                legal_actions_mask=exp.legal_actions_mask,
            )
            rotated_experiences.append(rotated_exp)
        else:
            # 保持原样
            rotated_experiences.append(exp)

    return rotated_experiences


# ===== 导出 =====
__all__ = [
    "rotate_splendor_observation",
    "augment_experiences_with_rotation",
    "apply_random_rotation_to_batch",
]
