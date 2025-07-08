# -*- coding: utf-8 -*-

import gym
import test_env
import numpy as np
import torch
import agent as agent_ppo


title = "MountainCarContinuous-v0"

test_env.print_env(title)

env = gym.make(title)

STATE_N = env.observation_space.shape[0]
ACTION_N = env.action_space.shape[0]
BATCH_SIZE = 128

epochs_n = 320
traject_n = 1024

agent = agent_ppo.PPO(BATCH_SIZE, STATE_N, ACTION_N, 3e-3, 3e-4,
                      hidden_dim=256, epochs=10, epsilon=0.2,
                      entropy_coef=0.1, entropy_coef_decay=0.98, gamma=0.99)

for epoch in range(epochs_n):
    state, _ = env.reset(seed=np.random.randint(1000))
    total_reward = 0
    for num in range(traject_n):
        action, log_prob = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        reward += abs(state[1]*action)*10
        total_reward += reward
        agent.memory_func(state, action, log_prob, reward, done, next_state)
        if num == 1000:
            print(agent.pi(torch.Tensor(state)))
        if done:
            print("win game")
            break
        state = next_state
    print()

    agent.fit()

    print(f"{epoch+1} episode end with total reward equal {total_reward}")

test_env.print_env(title, num_iter=1000, agent=agent)
agent.save_agent_params()
