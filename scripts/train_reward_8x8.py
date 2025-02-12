from matplotlib import pyplot as plt
import numpy as np
from max_ent import MaxEntIRL
from q_learning import q_learning
import gymnasium as gym


def get_expert_trajs() -> list[np.ndarray]:
    """ Generate expert trajectories """
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    q = q_learning(env, num_episodes=50000, log=True, alpha=0.50, gamma=0.90)
    expert_trajectories = []
    n = 0
    while n < 128:
        s, _ = env.reset()
        traj = [(s, 0)]
        while True:
            adv = q[s] - q[s].max()
            candidate_actions = np.where(adv > -0.02)[0]
            a = np.random.choice(candidate_actions)
            s, r, t1, t2, _ = env.step(a)
            traj.append((s, r))
            if t1 or t2:
                break
        if r == 1 and len(traj) < 20:
            expert_trajectories.append(np.array([t[0] for t in traj]))
            n += 1
    return expert_trajectories


env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
trajs = get_expert_trajs()
# Train the reward function
model = MaxEntIRL(env, expert_trajectories=trajs)
theta, rewards = model.train(iters=100, q_episodes=20000, gamma=0.80, lr=0.20)
with open("theta_8x8.npy", "wb") as f:
    np.save(f, theta)
# Plot the true rewards
plt.plot(rewards)
plt.xlabel("Iterations")
plt.ylabel("Total rewards")
plt.grid()
plt.show()
