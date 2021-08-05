import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.envs.registration import register

def one_hot(x):
    return np.identity(16)[x:x+1]

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': True}
)
env = gym.make('FrozenLake-v3')

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1
lr_stochastic = .85

x = tf.placeholder(tf.float32, [1, input_size])
w = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))

Qpred = tf.matmul(x, w)
y = tf.placeholder(tf.float32, [1, output_size])

loss = tf.reduce_mean(tf.square(y-Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

dis = .99
num_episodes = 2000

rList =[]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
global_loss= []
for i in range(num_episodes):
    state = env.reset()
    e = 1. / ((i/100) + 1)
    rAll = 0
    done = False
    local_loss = []

    while not done:
        Qs = sess.run(Qpred, feed_dict={x: one_hot(state)})
        if np.random.rand(1)<e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Qs)

        new_state , reward, done, _ = env.step(action)
        if done:
            Qs[0, action] = reward
        else:
            Qs1 = sess.run(Qpred, feed_dict={x: one_hot(new_state)})
            Qs[0, action] = reward + dis * np.max(Qs1)
        l,_ = sess.run([loss, train], feed_dict={x: one_hot(state), y: Qs})
        global_loss.append(l)
        local_loss.append(l)
        rAll +=reward
        state = new_state
    print('episode:', i, 'loss:', np.mean(local_loss))
    rList.append(rAll)

print('Success rate : '+str(sum(rList)/num_episodes))
ax1 = plt.subplot(212)
ax1.plot(global_loss)
ax = plt.subplot(211)
ax.title.set_text("Success rate: " + str(sum(rList) / num_episodes))
ax.bar(range(len(rList)), rList, color='blue')
plt.show()