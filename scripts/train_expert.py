import gymnasium as gym
import numpy as np
import pickle

from src.q_learning import q_learning


if __name__ == "__main__":
    # 初始化环境
    env = gym.make("FrozenLake-v1", is_slippery=False)

    # 训练Q-Learning
    q = q_learning(env)

    # 打印学习到的Q表
    print("Table of max Q:\n", q)

    # 采样学习到的策略
    trajs = []
    for _ in range(50):
        traj = []
        s, _ = env.reset()
        while True:
            # 选择Q值最大的动作
            # q[s][np.abs(q[s].max() - q[s]) > 1e-2] = -np.inf
            # p = np.exp(q[s]) / np.exp(q[s]).sum()
            # a = np.random.choice(env.action_space.n, p=p)
            a = np.argmax(q[s])
            s, r, t1, t2, _ = env.step(a)
            traj.append(s)
            if t1 or t2:
                break
        trajs.append(traj)

    # 保存采样的轨迹
    with open("expert/expert.pkl", "wb") as f:
        pickle.dump(trajs, f)

    # 测试学习到的策略
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
    state, info = env.reset()
    while True:
        env.render()
        # 选择Q值最大的动作
        action = np.argmax(q[state])
        state, reward, t1, t2, info = env.step(action)
        print(state)
        if t1 or t2:
            break
