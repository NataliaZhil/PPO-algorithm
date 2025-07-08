# -*- coding: utf-8 -*-

import gym
import matplotlib.pyplot as plt


def print_env(title: str, num_iter: int = 10, human: bool = False,
              agent=None) -> None:
    """
    Create image of enviroment with your or zeros actions

    Args:
        title: title of enviroment
        num_iter: steps in env
        human: human control action (True) or constant action = 0
        policy: dictionary with behaviour politic

    """
    env = gym.make(title, render_mode="rgb_array")
    state, _ = env.reset()
    for _ in range(num_iter):
        if human:
            action = [int(input())]
        elif agent is not None:
            action, _ = agent.get_action(state)
        else:
            action = [-0.5]
        tuple_env_params = env.step(action)
        state = tuple_env_params[0]
        frame = env.render()  # Returns RGB array
        plt.imshow(frame)
        plt.axis('off')
        plt.pause(0.1)
        plt.clf()
        if tuple_env_params[2]:
            break

    env.close()
