"""
设备性能基准测试

对比 CPU 和 MPS (Metal Performance Shaders) 的推理性能。
"""

import time
import torch
import numpy as np
from games.registry import create_game
from models.model_factory import create_splendor_model
from agents.neural_agent import NeuralAgent
from training.self_play_worker import collect_episode


def benchmark_inference(device: str, num_iterations: int = 100) -> dict:
    """
    基准测试：单次推理延迟

    Args:
        device: 设备 ("cpu" 或 "mps")
        num_iterations: 迭代次数

    Returns:
        性能统计字典
    """
    print(f"\n{'='*60}")
    print(f"基准测试：单次推理延迟 ({device.upper()})")
    print(f"{'='*60}")

    # 创建模型
    model = create_splendor_model(encoder_type='mlp', config='small')
    model.to(device)
    model.eval()

    # 准备输入
    obs_dim = 384
    action_size = 50

    # 预热
    print("预热中...")
    for _ in range(10):
        obs = torch.randn(1, obs_dim, device=device)
        mask = torch.ones(1, action_size, dtype=torch.bool, device=device)
        with torch.no_grad():
            _ = model(obs, mask)

    # 同步（确保预热完成）
    if device == "mps":
        torch.mps.synchronize()

    # 基准测试
    print(f"运行 {num_iterations} 次推理...")
    start_time = time.time()

    for _ in range(num_iterations):
        obs = torch.randn(1, obs_dim, device=device)
        mask = torch.ones(1, action_size, dtype=torch.bool, device=device)
        with torch.no_grad():
            _ = model(obs, mask)

    # 同步
    if device == "mps":
        torch.mps.synchronize()

    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_iterations * 1000  # ms

    print(f"✓ 完成")
    print(f"  总耗时: {elapsed_time:.3f}s")
    print(f"  平均延迟: {avg_time:.3f}ms")
    print(f"  吞吐量: {num_iterations / elapsed_time:.1f} samples/s")

    return {
        "device": device,
        "total_time": elapsed_time,
        "avg_latency_ms": avg_time,
        "throughput": num_iterations / elapsed_time,
    }


def benchmark_with_data_transfer(device: str, num_iterations: int = 100) -> dict:
    """
    基准测试：包含数据传输的完整流程

    模拟实际使用场景：numpy -> GPU -> 推理 -> CPU

    Args:
        device: 设备 ("cpu" 或 "mps")
        num_iterations: 迭代次数

    Returns:
        性能统计字典
    """
    print(f"\n{'='*60}")
    print(f"基准测试：包含数据传输 ({device.upper()})")
    print(f"{'='*60}")

    # 创建模型
    model = create_splendor_model(encoder_type='mlp', config='small')
    model.to(device)
    model.eval()

    obs_dim = 384
    action_size = 50

    # 预热
    print("预热中...")
    for _ in range(10):
        obs_np = np.random.randn(obs_dim).astype(np.float32)
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        mask = torch.ones(1, action_size, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, value = model(obs, mask)
        _ = logits.cpu().numpy()

    # 同步
    if device == "mps":
        torch.mps.synchronize()

    # 基准测试
    print(f"运行 {num_iterations} 次推理（含数据传输）...")
    start_time = time.time()

    for _ in range(num_iterations):
        # 模拟实际流程
        obs_np = np.random.randn(obs_dim).astype(np.float32)
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        mask = torch.ones(1, action_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            logits, value = model(obs, mask)

        # 转回 CPU（实际使用中必须的）
        _ = logits.cpu().numpy()
        _ = value.cpu().item()

    # 同步
    if device == "mps":
        torch.mps.synchronize()

    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_iterations * 1000  # ms

    print(f"✓ 完成")
    print(f"  总耗时: {elapsed_time:.3f}s")
    print(f"  平均延迟: {avg_time:.3f}ms")
    print(f"  吞吐量: {num_iterations / elapsed_time:.1f} samples/s")

    return {
        "device": device,
        "total_time": elapsed_time,
        "avg_latency_ms": avg_time,
        "throughput": num_iterations / elapsed_time,
    }


def benchmark_self_play(device: str, num_episodes: int = 10) -> dict:
    """
    基准测试：自对弈性能

    Args:
        device: 设备 ("cpu" 或 "mps")
        num_episodes: Episode 数量

    Returns:
        性能统计字典
    """
    print(f"\n{'='*60}")
    print(f"基准测试：自对弈性能 ({device.upper()})")
    print(f"{'='*60}")

    # 创建游戏和模型
    game = create_game('splendor', num_players=4)
    model = create_splendor_model(encoder_type='mlp', config='small')
    agents = [NeuralAgent(model=model, device=device) for _ in range(4)]

    print(f"运行 {num_episodes} 个 episodes...")
    start_time = time.time()

    total_steps = 0
    for i in range(num_episodes):
        episode = collect_episode(
            game=game,
            agents=agents,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=False,
        )
        total_steps += episode.num_steps
        if (i + 1) % 5 == 0:
            print(f"  进度: {i + 1}/{num_episodes}")

    elapsed_time = time.time() - start_time

    print(f"✓ 完成")
    print(f"  总耗时: {elapsed_time:.3f}s")
    print(f"  总步数: {total_steps}")
    print(f"  平均步数/episode: {total_steps / num_episodes:.1f}")
    print(f"  吞吐量: {total_steps / elapsed_time:.1f} steps/s")
    print(f"  速度: {num_episodes / elapsed_time * 3600:.1f} episodes/hour")

    return {
        "device": device,
        "total_time": elapsed_time,
        "total_steps": total_steps,
        "avg_steps_per_episode": total_steps / num_episodes,
        "throughput_steps": total_steps / elapsed_time,
        "throughput_episodes": num_episodes / elapsed_time,
    }


def main():
    """主函数"""
    print("\n" + "="*60)
    print("设备性能基准测试")
    print("="*60)

    # 检查可用设备
    has_mps = torch.backends.mps.is_available()

    print(f"\n可用设备:")
    print(f"  CPU: ✓")
    print(f"  MPS: {'✓' if has_mps else '✗'}")

    if not has_mps:
        print("\n⚠️  MPS 不可用，仅测试 CPU")
        devices = ["cpu"]
    else:
        devices = ["cpu", "mps"]

    # 测试 1: 纯推理性能
    print("\n" + "="*60)
    print("测试 1: 纯推理性能（无数据传输）")
    print("="*60)

    inference_results = {}
    for device in devices:
        result = benchmark_inference(device, num_iterations=100)
        inference_results[device] = result

    # 对比
    if len(devices) > 1:
        cpu_time = inference_results["cpu"]["avg_latency_ms"]
        mps_time = inference_results["mps"]["avg_latency_ms"]
        speedup = cpu_time / mps_time
        print(f"\n📊 对比:")
        print(f"  CPU: {cpu_time:.3f}ms")
        print(f"  MPS: {mps_time:.3f}ms")
        print(f"  加速比: {speedup:.2f}x")

    # 测试 2: 包含数据传输
    print("\n" + "="*60)
    print("测试 2: 包含数据传输的完整流程")
    print("="*60)

    transfer_results = {}
    for device in devices:
        result = benchmark_with_data_transfer(device, num_iterations=100)
        transfer_results[device] = result

    # 对比
    if len(devices) > 1:
        cpu_time = transfer_results["cpu"]["avg_latency_ms"]
        mps_time = transfer_results["mps"]["avg_latency_ms"]
        speedup = cpu_time / mps_time
        print(f"\n📊 对比:")
        print(f"  CPU: {cpu_time:.3f}ms")
        print(f"  MPS: {mps_time:.3f}ms")
        print(f"  加速比: {speedup:.2f}x")

        if speedup < 1.0:
            print(f"\n⚠️  警告: MPS 比 CPU 慢 {1/speedup:.2f}x")
            print("  原因: 数据传输开销超过计算收益")

    # 测试 3: 自对弈性能
    print("\n" + "="*60)
    print("测试 3: 自对弈性能（实际训练场景）")
    print("="*60)

    selfplay_results = {}
    for device in devices:
        result = benchmark_self_play(device, num_episodes=10)
        selfplay_results[device] = result

    # 对比
    if len(devices) > 1:
        cpu_throughput = selfplay_results["cpu"]["throughput_steps"]
        mps_throughput = selfplay_results["mps"]["throughput_steps"]
        speedup = mps_throughput / cpu_throughput
        print(f"\n📊 对比:")
        print(f"  CPU: {cpu_throughput:.1f} steps/s")
        print(f"  MPS: {mps_throughput:.1f} steps/s")
        print(f"  加速比: {speedup:.2f}x")

        if speedup < 1.0:
            print(f"\n⚠️  警告: MPS 比 CPU 慢 {1/speedup:.2f}x")

    # 总结
    print("\n" + "="*60)
    print("性能总结与建议")
    print("="*60)

    if len(devices) > 1:
        mps_faster_inference = inference_results["mps"]["avg_latency_ms"] < inference_results["cpu"]["avg_latency_ms"]
        mps_faster_transfer = transfer_results["mps"]["avg_latency_ms"] < transfer_results["cpu"]["avg_latency_ms"]
        mps_faster_selfplay = selfplay_results["mps"]["throughput_steps"] > selfplay_results["cpu"]["throughput_steps"]

        if mps_faster_inference and not mps_faster_transfer:
            print("\n📊 诊断: MPS 计算快但总体慢")
            print("  ✓ 纯计算: MPS 更快")
            print("  ✗ 含传输: MPS 更慢")
            print("\n💡 建议:")
            print("  1. 使用 CPU 训练（当前模型太小，GPU 优势不明显）")
            print("  2. 或者增大模型规模（使用 'medium' 或 'large' 配置）")
            print("  3. 或者增大 batch_size（批量处理多个样本）")
            print("  4. 考虑使用 pin_memory 加速数据传输")
        elif mps_faster_selfplay:
            print("\n✅ MPS 训练更快，建议使用 MPS")
        else:
            print("\n✅ CPU 训练更快，建议使用 CPU")
            print("\n💡 原因:")
            print("  - 模型规模较小（~150K 参数）")
            print("  - Batch size = 1（无法利用 GPU 并行性）")
            print("  - 频繁的 CPU-GPU 数据传输")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
