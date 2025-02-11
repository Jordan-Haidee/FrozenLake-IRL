import gymnasium as gym

env = gym.make("FrozenLake-v1", map_name="8x8", render_mode="human")

s, _ = env.reset()

while True:
    env.render()
    a = env.action_space.sample()
    s, r, t1, t2, _ = env.step(a)
    if t1 or t2:
        break
