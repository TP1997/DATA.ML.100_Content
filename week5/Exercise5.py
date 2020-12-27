import gym
import random
import numpy
import time

# Environment
env = gym.make("Taxi-v3")

# Training parameters for Q learning
alpha = 0.9         # Learning rate
gamma = 0.9         # Future reward discount factor
num_of_episodes = 1000

# Q table for rewards
Q_reward = numpy.zeros((500,6))

# Training w/ random sampling of actions
epsilon = 0.1
for ep in range(num_of_episodes):
    state = env.reset()
    for t in range(50):
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = numpy.argmax(Q_reward[state,:])
            
        next_state, reward, done, _ = env.step(action)
        
        Q_reward[state,action] += alpha*(reward + gamma*numpy.max(Q_reward[next_state,:]) - Q_reward[state,action])

        state = next_state         
        if done: break
        
    epsilon = numpy.max([0.01, epsilon*0.99])
    
print('training finished.')

# Compute average reward and number of actions on 10 runs.
tot_rewards = 0
nof_actions = 0
for i in range(10):
    state = env.reset()
    done = False
    while not done:
        action = numpy.argmax(Q_reward[state,:])
        state, reward, done, _ = env.step(action)
        
        tot_rewards += reward
        nof_actions += 1
        
print('Total reward on average = {}'.format(tot_rewards/10))
print('Actions taken on average = {}'.format(nof_actions/10))


# Single step-by-step evaluation
if False:
    tot_reward = 0
    state = env.reset()
    print('(Starting position)')
    env.render()
    act=0
    for t in range(50):
        action = numpy.argmax(Q_reward[state,:])
        state, reward, done, info = env.step(action)
        tot_reward += reward
        
        env.render()
        time.sleep(1)
        act+=1
        if done:
            print('Total reward = {}'.format(tot_reward))
            print('Actions = {}'.format(act))
            break
