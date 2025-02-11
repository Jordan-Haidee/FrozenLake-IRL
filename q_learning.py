import gymnasium as gym
import numpy as np
from tqdm import trange, tqdm


def q_learning(
    env: gym.Env,
    reward_fn: np.ndarray | None = None,
    log: bool = False,
) -> np.ndarray:
    # Q-Learning的超参数
    alpha = 0.50  # 学习率
    gamma = 0.95  # 折扣因子
    epsilon_init = 1.00  # 初始探索率
    num_episodes = 2000  # 训练的回合数

    # 初始化Q表
    q = np.zeros([env.observation_space.n, env.action_space.n])

    # Q-Learning算法
    for episode in trange(num_episodes, leave=False, disable=not log):
        state, _ = env.reset()  # 重置环境
        episode_reward = 0

        # 衰减epislon
        epsilon = epsilon_init * (1 - 0.90 * episode / num_episodes)

        while True:
            # epsilon-greedy策略
            if np.random.rand() < epsilon:
                # 探索: 随机选择动作
                action = np.random.choice(env.action_space.n)
            else:
                # 利用: 选择Q值最大的动作
                action = np.argmax(q[state])

            # 执行动作，获取下一个状态、奖励、是否结束等信息
            next_state, original_reward, t1, t2, _ = env.step(action)
            if reward_fn is None:
                reward = original_reward
            else:
                reward = reward_fn[next_state]

            # 计算回合奖励
            episode_reward += reward

            # 更新Q值
            q_loss = (
                reward + gamma * np.max(q[next_state]) * (1 - t1) - q[state, action]
            )
            q[state, action] += alpha * q_loss

            # 更新状态
            state = next_state

            # 判断是否结束
            if t1 or t2:
                break

        # 打印回合信息
        if log:
            tqdm.write(
                f"回合: {episode + 1:5d}, 回合奖励: {episode_reward}, epsilon: {epsilon:.4f}, q_loss: {q_loss:.4f}"
            )

    return q
