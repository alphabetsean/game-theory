from covid19_gym import Covid19Gym
from pprint import pprint
import numpy as np

env = Covid19Gym()
env.reset()
# pprint(env.state)
obs = env.reset()
# for i in range(len(df['Date'])):
for i in range(100):
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    env.render(title="sdsd")