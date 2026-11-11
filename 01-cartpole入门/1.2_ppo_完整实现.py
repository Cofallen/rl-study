"""
1.2 CartPole 原理 —— 把书里零散的代码片段补成一份可以直接运行的完整实现

对应 https://walkinglabs.github.io/hands-on-modern-rl/chapter01_cartpole/metrics

页面里只给了片段（ActorCritic、裁剪目标、collect_rollout 的循环、GAE 公式、
训练主循环……），这里补齐成纯 PyTorch 的 PPO，并记录页面讲的 5 个指标：

    mean_episode_reward        最近 100 个完整回合的滑动平均（SB3 默认窗口）
    mean_episode_reward_rollout 本轮结束回合的平均值
    value_loss                 Critic 的预测误差
    entropy                    策略熵（动作选择有多确定）
    approx_kl                  新旧策略的近似 KL
    clip_fraction              触发裁剪的样本占比

运行方式（在仓库根目录）：

    # 默认：训练 40 轮 × 2048 步，存 CSV 并画图
    python 01-cartpole入门/1.2_ppo_完整实现.py

    # 快速验证（几十秒）
    python 01-cartpole入门/1.2_ppo_完整实现.py --iterations 5 --steps-per-rollout 512

    # 训练完弹窗看小车（需要图形界面）
    python 01-cartpole入门/1.2_ppo_完整实现.py --demo
"""

import argparse
import csv
import os
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib

matplotlib.use("Agg")  # 先不要弹窗，最后统一 savefig
import matplotlib.pyplot as plt  # noqa: E402

# ==========================================================
# 第一部分：环境信息（对应"观测：环境每步返回四个数"）
# ==========================================================


def print_env_info(env):
    print("=" * 56)
    print("CartPole-v1 环境信息")
    print("=" * 56)
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  观测上限: {env.observation_space.high}")
    print(f"  观测下限: {env.observation_space.low}")
    print(
        f"  终止条件: 小车位置 > ±{env.unwrapped.x_threshold}，"
        f"杆子角度 > ±{env.unwrapped.theta_threshold_radians:.4f} rad"
        f"（约 ±{np.degrees(env.unwrapped.theta_threshold_radians):.0f}°）"
    )
    print("  说明: 速度维度的边界是 inf，表示 Gymnasium 没有给速度声明有限边界。")
    print("=" * 56)


# ==========================================================
# 第二部分：Actor-Critic 网络
# ==========================================================
class ActorCritic(nn.Module):
    """Actor 输出两个动作的分数，Critic 输出状态价值 V(s)。"""

    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """正交初始化：actor 输出层用小 gain，让初始策略接近均匀分布。"""
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0.0)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# ==========================================================
# 第三部分：收集轨迹（rollout）
# ==========================================================
def collect_rollout(
    model,
    env,
    obs,
    device,
    episode_reward=0.0,
    episode_length=0,
    num_steps=2048,
):
    """
    采样 num_steps 步，正确处理 terminated / truncated：
      - terminated=True （杆子倒下/小车出界）：V(s') = 0
      - truncated=True  （500 步上限）：仍用 V(s') 做自举
      - rollout 在回合中间结束 ：同样用 V(s') 做自举
    跨 rollout 边界的回合会继续累计奖励，等回合真正结束才记账。
    """
    transitions = []
    completed_rewards = []
    completed_lengths = []

    for _ in range(num_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, log_prob, value = model.get_action(obs_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())

        with torch.no_grad():
            if terminated:
                next_value = 0.0
            else:
                next_tensor = torch.as_tensor(
                    next_obs, dtype=torch.float32, device=device
                )
                _, next_value = model(next_tensor)
                next_value = next_value.item()

        transitions.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "action": int(action.item()),
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                # 存下 V(s')，让终止、截断、rollout 边界共用同一套 GAE 公式
                "next_value": float(next_value),
            }
        )

        episode_reward += float(reward)
        episode_length += 1
        obs = next_obs

        if terminated or truncated:
            completed_rewards.append(episode_reward)
            completed_lengths.append(episode_length)
            episode_reward = 0.0
            episode_length = 0
            obs, _ = env.reset()

    return (
        transitions,
        obs,
        completed_rewards,
        completed_lengths,
        episode_reward,
        episode_length,
    )


# ==========================================================
# 第四部分：TD 误差与 GAE 优势
# ==========================================================
def compute_gae(transitions, gamma=0.99, lam=0.95):
    """GAE 优势；在每次 reset 处切断，优势不跨回合传播。"""
    raw_advantages = []
    gae = 0.0

    for step in reversed(range(len(transitions))):
        t = transitions[step]
        episode_end = t["terminated"] or t["truncated"]
        delta = t["reward"] + gamma * t["next_value"] - t["value"]
        gae = delta + gamma * lam * (1.0 - float(episode_end)) * gae
        raw_advantages.insert(0, gae)

    raw_advantages = torch.tensor(raw_advantages, dtype=torch.float32)
    values = torch.tensor(
        [t["value"] for t in transitions], dtype=torch.float32
    )

    # Critic 学的是未归一化的回报目标，归一化只用于策略损失
    returns = raw_advantages + values
    advantages = (raw_advantages - raw_advantages.mean()) / (
        raw_advantages.std(unbiased=False) + 1e-8
    )
    return advantages, returns


# ==========================================================
# 第五部分：PPO 裁剪目标更新
# ==========================================================
def ppo_update(
    model,
    optimizer,
    transitions,
    advantages,
    returns,
    device,
    clip_eps=0.2,
    epochs=10,
    batch_size=64,
    value_coef=0.5,
    entropy_coef=0.0,
    max_grad_norm=0.5,
):
    obs = np.array([t["obs"] for t in transitions], dtype=np.float32)
    actions = np.array([t["action"] for t in transitions], dtype=np.int64)
    old_log_probs = np.array(
        [t["log_prob"] for t in transitions], dtype=np.float32
    )

    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(
        old_log_probs, dtype=torch.float32, device=device
    )
    advantages = advantages.to(device)
    returns = returns.to(device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_clip_frac = 0.0
    n_updates = 0

    for _ in range(epochs):
        indices = np.random.permutation(len(transitions))
        for start in range(0, len(transitions), batch_size):
            idx = indices[start:start + batch_size]

            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]

            logits, values = model(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)

            # 概率比与裁剪目标
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = (
                torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                * batch_advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Critic 的价值损失
            value_loss = ((values - batch_returns) ** 2).mean()

            # 策略熵：鼓励探索
            entropy = dist.entropy().mean()

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = new_log_probs - batch_old_log_probs
                # 非负的 KL 近似（与 SB3 的 approx_kl 一致）
                total_kl += ((log_ratio.exp() - 1) - log_ratio).mean().item()
                clipped = (ratio - 1.0).abs() > clip_eps
                total_clip_frac += clipped.float().mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
        "approx_kl": total_kl / n_updates,
        "clip_fraction": total_clip_frac / n_updates,
    }


# ==========================================================
# 第六部分：训练主循环
# ==========================================================
def resolve_device(choice="auto"):
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_available = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="纯 PyTorch PPO 训练 CartPole-v1"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--iterations", type=int, default=40, help="PPO 迭代轮数"
    )
    parser.add_argument(
        "--steps-per-rollout",
        type=int,
        default=2048,
        help="每轮采样步数",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="GAE 的 lambda"
    )
    parser.add_argument("--clip-eps", type=float, default=0.2, help="裁剪范围")
    parser.add_argument(
        "--epochs", type=int, default=10, help="每轮数据复用次数"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="小批次大小")
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="初始学习率（线性衰减到 0）"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=20, help="训练后的独立评估回合数"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="训练设备",
    )
    parser.add_argument(
        "--log-csv",
        default="01-cartpole入门/output/training_metrics.csv",
        help="训练指标 CSV 保存路径",
    )
    parser.add_argument(
        "--output-dir", default="01-cartpole入门/output", help="模型与曲线图保存目录"
    )
    parser.add_argument("--no-plot", action="store_true", help="不要画曲线图")
    parser.add_argument("--demo", action="store_true", help="训练后弹窗演示 5 个回合")
    return parser.parse_args()


def evaluate(model, env, device, episodes, seed):
    """独立评估：确定性策略（argmax），从新的初始状态跑 episodes 回合。"""
    scores = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        done = False
        truncated = False
        score = 0.0
        while not (done or truncated):
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                action, _, _ = model.get_action(
                    obs_tensor, deterministic=True
                )
            obs, reward, done, truncated, _ = env.step(action.item())
            score += float(reward)
        scores.append(score)
    return np.array(scores, dtype=np.float64)


def train():
    args = parse_args()
    device = resolve_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    obs, _ = env.reset(seed=args.seed)

    print_env_info(env)
    print(f"训练设备: {device}")

    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    csv_path = Path(args.log_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pytorch_ppo_cartpole.pth"

    metric_rows = []
    total_timesteps = 0
    ongoing_episode_reward = 0.0
    ongoing_episode_length = 0

    # 与 SB3 的 stats_window_size=100 一致：滑动窗口跨轮次累计
    ep_rew_buffer = deque(maxlen=100)
    ep_len_buffer = deque(maxlen=100)

    print(f"开始训练：{args.iterations} 轮 × {args.steps_per_rollout} 步")
    print("-" * 64)

    for iteration in range(args.iterations):
        # 1) 收集轨迹
        (
            transitions,
            obs,
            ep_rewards,
            ep_lengths,
            ongoing_episode_reward,
            ongoing_episode_length,
        ) = collect_rollout(
            model,
            env,
            obs,
            device,
            ongoing_episode_reward,
            ongoing_episode_length,
            args.steps_per_rollout,
        )
        total_timesteps += len(transitions)

        # 2) 计算优势与回报
        advantages, returns = compute_gae(
            transitions, gamma=args.gamma, lam=args.gae_lambda
        )

        # 学习率线性衰减
        frac = 1.0 - iteration / args.iterations
        lr = args.lr * frac
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 3) 更新 Actor 与 Critic
        metrics = ppo_update(
            model,
            optimizer,
            transitions,
            advantages,
            returns,
            device,
            clip_eps=args.clip_eps,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # Critic 解释了多少回报方差
        return_values = returns.numpy()
        rollout_values = np.array([t["value"] for t in transitions])
        var_returns = float(np.var(return_values))
        if var_returns < 1e-6:
            explained_variance = 0.0
        else:
            residual_var = float(np.var(return_values - rollout_values))
            explained_variance = 1.0 - residual_var / var_returns

        # 统计
        mean_reward_rollout = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_ep_len_rollout = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        ep_rew_buffer.extend(ep_rewards)
        ep_len_buffer.extend(ep_lengths)
        mean_reward = float(np.mean(ep_rew_buffer))
        mean_ep_len = float(np.mean(ep_len_buffer))

        metric_rows.append(
            {
                "seed": args.seed,
                "iteration": iteration + 1,
                "total_timesteps": total_timesteps,
                "completed_episodes": len(ep_rewards),
                "mean_episode_reward": mean_reward,
                "mean_episode_length": mean_ep_len,
                "mean_episode_reward_rollout": mean_reward_rollout,
                "mean_episode_length_rollout": mean_ep_len_rollout,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "approx_kl": metrics["approx_kl"],
                "clip_fraction": metrics["clip_fraction"],
                "explained_variance": explained_variance,
                "learning_rate": lr,
            }
        )

        print(
            f"  迭代 {iteration + 1:2d}/{args.iterations} | "
            f"回合数: {len(ep_rewards):3d} | "
            f"平均奖励: {mean_reward:6.1f} | "
            f"KL: {metrics['approx_kl']:.4f} | "
            f"clip%: {metrics['clip_fraction']:.1%}"
        )

    print("-" * 64)

    # 写 CSV（先写临时文件再原子替换）
    fieldnames = list(metric_rows[0].keys())
    temporary_csv = f"{csv_path}.tmp"
    with open(temporary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    os.replace(temporary_csv, csv_path)
    print(f"训练指标已保存: {csv_path}")

    # 独立评估
    eval_scores = evaluate(model, env, device, args.eval_episodes, args.seed)
    print(
        f"\n训练完成！{args.eval_episodes} 回合评估: "
        f"{eval_scores.mean():.1f} +/- {eval_scores.std():.1f}"
    )

    torch.save(model.state_dict(), model_path)
    print(f"模型已保存: {model_path}")

    # 画图
    if not args.no_plot:
        plot_curves(metric_rows, output_dir)

    # 演示
    if args.demo:
        try:
            vis_env = gym.make("CartPole-v1", render_mode="human")
            print("\n演示训练后的策略（5 回合）...")
            for ep in range(5):
                obs, _ = vis_env.reset(seed=args.seed + 20_000 + ep)
                done = False
                truncated = False
                score = 0.0
                while not (done or truncated):
                    obs_tensor = torch.as_tensor(
                        obs, dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        action, _, _ = model.get_action(
                            obs_tensor, deterministic=True
                        )
                    obs, reward, done, truncated, _ = vis_env.step(
                        action.item()
                    )
                    score += float(reward)
                print(f"  演示回合 {ep + 1} 得分: {score}")
            vis_env.close()
        except Exception as exc:  # 没有图形界面时直接跳过
            print(f"（跳过演示，无可用显示: {exc}）")

    env.close()
    print("\n提示: 加 --demo 可以弹窗看小车；CSV 与曲线图在 " + str(output_dir))


# ==========================================================
# 第七部分：画训练曲线（奖励 + 四个辅助指标）
# ==========================================================
def _setup_cjk_font():
    """尽量找一个中文字体，避免图里的中文变成方框。"""
    from matplotlib import font_manager

    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_curves(metric_rows, output_dir):
    _setup_cjk_font()
    iterations = [row["iteration"] for row in metric_rows]
    rewards = [row["mean_episode_reward"] for row in metric_rows]
    rewards_rollout = [
        row["mean_episode_reward_rollout"] for row in metric_rows
    ]

    # 图 1：回合奖励
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        iterations, rewards, marker="o", markersize=3, label="100 回合滑动平均"
    )
    ax.plot(
        iterations,
        rewards_rollout,
        marker="s",
        markersize=3,
        linestyle="--",
        alpha=0.6,
        label="本轮结束回合平均",
    )
    ax.axhline(500, color="gray", linestyle=":", label="单回合 500 步上限")
    ax.set_xlabel("迭代轮次")
    ax.set_ylabel("回合奖励")
    ax.set_title("CartPole-v1 PPO 训练奖励")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    reward_path = Path(output_dir) / "cartpole_reward.png"
    fig.savefig(reward_path, dpi=150)
    plt.close(fig)
    print(f"奖励曲线已保存: {reward_path}")

    # 图 2：四个辅助指标
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    specs = [
        ("value_loss", "价值损失", axes[0][0]),
        ("entropy", "策略熵", axes[0][1]),
        ("approx_kl", "近似 KL", axes[1][0]),
        ("clip_fraction", "裁剪比例", axes[1][1]),
    ]
    for key, title, ax in specs:
        ax.plot(
            iterations,
            [row[key] for row in metric_rows],
            marker="o",
            markersize=3,
        )
        ax.set_title(title)
        ax.set_xlabel("迭代轮次")
        ax.grid(alpha=0.3)
    axes[1][1].yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    )
    fig.suptitle("PPO 训练诊断指标")
    fig.tight_layout()
    diag_path = Path(output_dir) / "cartpole_diagnostics.png"
    fig.savefig(diag_path, dpi=150)
    plt.close(fig)
    print(f"诊断曲线已保存: {diag_path}")


if __name__ == "__main__":
    train()
