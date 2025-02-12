import gymnasium as gym
import numpy as np
from tqdm import tqdm, trange
from q_learning import q_learning


class MaxEntIRL:
    def __init__(self, env: gym.Env, expert_trajectories: np.ndarray):
        self.env = env
        self.features = np.eye(env.observation_space.n).astype(
            np.float32
        )  # one-hot encoding
        self.expert_expectations = self.compute_feature_expectations(
            expert_trajectories
        )

    def state_to_feature(self, state: int):
        # Feature map: one-hot encoding of the state (zero overhead)
        return self.features[state]

    def train(self, num: int = 50) -> tuple[np.ndarray, list[float]]:
        # Initialize theta randomly
        theta = np.random.uniform(-1, 1, (self.features.shape[1],)).astype(np.float32)
        # Perform IRL and record the true rewards
        true_rewards_list = []
        for i in trange(num):
            reward_fn = (
                np.array(
                    [
                        self.state_to_feature(s)
                        for s in range(self.env.observation_space.n)
                    ]
                )
                @ theta
            )
            q = q_learning(self.env, reward_fn=reward_fn)
            trajs, true_rewards = self.generate_trajectories(q)
            feature_expectations = self.compute_feature_expectations(trajs)
            theta += 0.05 * (self.expert_expectations - feature_expectations)
            true_rewards_list.append(np.mean(true_rewards).item())
            tqdm.write(f"{i} -> {true_rewards_list[-1]}")

        return theta, true_rewards_list

    def generate_trajectories(
        self, q: np.ndarray, num: int = 64
    ) -> tuple[list[np.ndarray], list[float]]:
        # sample via advantage A(s,a) = Q(s,a) - V(s)
        advantage = q - q.max(axis=1, keepdims=True)
        policy = np.exp(advantage) / np.exp(advantage).sum(axis=1, keepdims=True)
        trajs = []
        true_rewards = []
        for _ in range(num):
            s, _ = self.env.reset()
            traj = [(s, 0)]
            while True:
                a = np.random.choice(self.env.action_space.n, p=policy[s])
                s, true_r, t1, t2, _ = self.env.step(a)
                traj.append((s, true_r))
                if t1 or t2:
                    break
            trajs.append([t[0] for t in traj])
            true_rewards.append(sum([t[1] for t in traj]))
        return trajs, true_rewards

    def compute_feature_expectations(self, trajectories: list[np.ndarray]):
        return np.array(
            [self.state_to_feature(s) for traj in trajectories for s in traj]
        ).sum(axis=0) / len(trajectories)
