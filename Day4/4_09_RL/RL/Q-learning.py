import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

def qmax_action(q):
    maxq = np.amax(q)
    indices = np.nonzero(q == maxq)[0]
    return np.random.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)
env = gym.make('FrozenLake-v3')

#Inistialize Q-table with all zeros, shape = [States num, 4(left,down,right,up)]
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-Table learning algorithm
    while not done:
        action = qmax_action(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.title("Success rate: " + str(sum(rList) / num_episodes))
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
