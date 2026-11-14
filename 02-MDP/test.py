import numpy as np


class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q = np.zeros(n_arms)   # 每个臂的价值估计
        self.n = np.zeros(n_arms)   # 每个臂被拉次数

    def select(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)   # 探索
        return int(np.argmax(self.q))               # 利用

    def update(self, arm, reward):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]


class Bandit:
    """每个老虎机有真实中奖概率 probs[i]；拉一次，以该概率返回 1，否则 0"""
    def __init__(self, probs):
        self.probs = np.array(probs, dtype=float)

    @property
    def n_arms(self):
        return len(self.probs)

    def pull(self, arm):
        return 1.0 if np.random.random() < self.probs[arm] else 0.0


def main():
    probs = [0.2, 0.5, 0.8]          # ← 3 台老虎机，真实概率
    bandit = Bandit(probs)
    agent = EpsilonGreedy(n_arms=bandit.n_arms, epsilon=0.1)

    n_steps = 1000
    total = 0.0
    for _ in range(n_steps):
        arm = agent.select()          # 1. 选臂
        reward = bandit.pull(arm)     # 2. 拉，得到奖励
        agent.update(arm, reward)     # 3. 更新估计
        total += reward

    print("真实概率 :", probs)
    print("估计价值 :", np.round(agent.q, 3))
    print("各臂次数 :", agent.n.astype(int))
    print(f"总奖励 {total:.0f}，平均 {total / n_steps:.3f}")


if __name__ == "__main__":
    main()