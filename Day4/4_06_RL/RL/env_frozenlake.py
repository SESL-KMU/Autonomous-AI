import gym
from gym.envs.registration import register

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v3')
state = env.reset()
env.render()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    'w' : UP,
    's' : DOWN,
    'd' : RIGHT,
    'a' : LEFT
}

while True:
    key = input()
    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State:", state, "Action", action, "Reward:", reward, "Info:", info)

    if done:  # 도착하면 게임을 끝낸다.
        print("Finished with reward", reward)
        break
