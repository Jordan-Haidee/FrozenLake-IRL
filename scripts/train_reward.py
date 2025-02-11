import gymnasium as gym
import numpy as np
import random
from matplotlib import pyplot as plt

from max_ent import MaxEntIRL


def optimal_policy(s: int) -> int:
    # Optimal policy for FrozenLake-v1
    match s:
        case 0:
            return random.choice([1, 2])
        case 1:
            return 2
        case 2:
            return 1
        case 3:
            return 0
        case 4:
            return 1
        case 5:
            return random.choice([1, 2])
        case 6:
            return 1
        case 7:
            return 0
        case 8:
            return 2
        case 9:
            return random.choice([1, 2])
        case 10:
            return 1
        case 11:
            return 0
        case 12:
            return 2
        case 13:
            return 2
        case 14:
            return 2
        case 15:
            return random.choice([1, 2])


def get_expert_trajs(num: int = 64) -> np.ndarray:
    """ Generate expert trajectories """
    env = gym.make("FrozenLake-v1", is_slippery=False)
    trajs = []
    for _ in range(num):
        s, _ = env.reset()
        traj = []
        while True:
            a = optimal_policy(s)
            s, _, t1, t2, _ = env.step(a)
            traj.append(s)
            if t1 or t2:
                break
        trajs.append(traj)
    return np.array(trajs)

# Grenerate expert trajectories
env = gym.make("FrozenLake-v1", is_slippery=False)
trajs = get_expert_trajs()
# Train the reward function
model = MaxEntIRL(env, expert_trajectories=trajs)
theta, rewards = model.train(num=500)
with open("theta.npy", "wb") as f:
    np.save(f, theta)
# Plot the true rewards
plt.plot(rewards)
plt.xlabel("Iterations")
plt.ylabel("Total rewards")
plt.grid()
plt.show()
