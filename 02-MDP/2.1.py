import numpy as np
import time

from warp import float32


class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q = np.zeros(n_arms)  # 每个臂 价值估计  
        self.n = np.zeros(n_arms)  # 每个臂 被拉次数

    def select(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q)

    def update(self, arm, reward):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]


class Bandit:
    def __init__(self, probs):
        self.probs = np.array(probs, dtype=float32)

    @property
    

def main():
    g = EpsilonGreedy(2)

    for _ in range(100):
        g.select()
        g.update(0, 0)

        print("K")
        time.sleep(1)


if __name__ == "__main__":
    main()