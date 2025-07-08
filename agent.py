# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class NN(nn.Module):
    """
    NN for Value functin and politic
    Args:
        state_dim: state space of enviroment
        hidden_dim: number of neurons in hidden layer
        action_dim: action space of enviroment
        pi_nn: if True set NN for policy else Value
    """

    def __init__(self, state_dim: int,
                 hidden_dim: int, action_dim: int, pi_nn: bool) -> None:
        super().__init__()
        self.pi_nn = pi_nn
        self.seq_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )
        self.v_fun = nn.Linear(hidden_dim, action_dim)

        self.seq_layer.apply(self.init_weight)
        self.mu.apply(self.init_weight)
        self.var.apply(self.init_weight)
        self.v_fun.apply(self.init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.seq_layer(x)
        if self.pi_nn:
            mu = self.mu(out)
            var = self.var(out)*0.99+0.01
            return mu, var
        v = self.v_fun(out)
        return v

    def init_weight(self, layer) -> None:
        """
        Initiaye start weight values of layer

        Args:
            layer: layer of NN

        """
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, -0.01, 0.01)
            nn.init.constant_(layer.bias, 0.1)


class PPO:
    """
    PPO algorithm

    Args:
        batch_size: size of batch for education NNs
        state_space: state space of enviroment
        action_space: action space of enviroment
        lr_v: learning rate for NN predict value function
        lr_pi: learning rate for NN predict policy
        load_params: if True load weight of NNs
        title: main part of file's name
                (exp: {title}_actor)
        epsilon: clipping range (default 0.2)
        hidden_dim: number of neurons in NN in hidden layer (default 200)
        gamma: discount factor (default 0.99)
        lambda_v: smoothing parameter (default 0.95)
        epochs: number of epoch for training NNs (default 10)
        entropy_coef: information entropy coefficient (default 1e-3)
        entropy_coef_decay: entropy decrease coefficient with evere epoch
                            (default 0.99)

    """

    def __init__(self, batch_size: int, state_space: int, action_space: int,
                 lr_v: float, lr_pi: float, load_params: bool = False,
                 title: str = "weights", epsilon: float = 0.2,
                 hidden_dim: int = 200, gamma: float = 0.99,
                 lambda_v: float = 0.95, epochs: int = 10,
                 entropy_coef: float = 1e-3,
                 entropy_coef_decay: float = 0.99) -> None:

        self.batch_size = batch_size
        self.memory = []
        self.pi = NN(state_space, hidden_dim, action_space, True)
        self.V = NN(state_space, hidden_dim, action_space, False)
        self.action_space = action_space
        self.epsilon = epsilon
        self.optim_v = torch.optim.Adam(
            self.V.parameters(), lr=lr_v)
        self.optim_pi = torch.optim.Adam(
            self.pi.parameters(), lr=lr_pi)
        self.gamma = gamma
        self.lambda_v = lambda_v
        self.crit = nn.MSELoss()
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        if load_params:
            self.load_agent_params(title)

    def get_action(self, state, action=None) -> tuple:
        """
        Function return action and log probability of that action
        if action is None else log probability and entropy

        Args:
            state: enviroment's state
                    (np.array or torch.tensor)
            action: previously selected action according to state
                    (np.array or torch.tensor)

        Return:
            (action, log prob): (selected action,
                                 log of probability of choosen action)
            (log prob, entropy): log of probability of choosen action
                                 entropy of disctribution)

        """

        state = torch.FloatTensor(state)
        mu, var = self.pi(state)
        dist = torch.distributions.Normal(mu, var)
        if action is None:
            action = torch.clamp(dist.sample(), min=-1, max=1)
            log_prob_a = dist.log_prob(action.detach()).detach().numpy()
            return (action.detach().numpy(), log_prob_a)
        entr = dist.entropy()
        return (dist.log_prob(action), entr)

    def memory_func(self, state: np.array, action: np.array,
                    log_prob_a: np.array, reward: float,
                    done: bool, next_state:  np.array) -> None:
        """
        Add parameters to internal memory

        Args:
            state: state of enviroment
            action : selected action
            log_prob_a : log of probability of action
            reward : reward for step system
            done : end cycle
            next_state : next state of systmem after step

        """
        self.memory.append(
            (state, action, log_prob_a, reward, done, next_state))

    def reset_memory(self) -> None:
        """
        Clear memoryy list

        """
        self.memory = []

    def save_agent_params(self, title: str = "weights") -> None:
        """
        Save agents NN params in file

        Args:
            title: main name of files (default weights)
                (full name would be {title}_critic/{title}_actor)

        """
        torch.save(self.pi.state_dict(), f'{title}_actor.pth')
        torch.save(self.V.state_dict(), f'{title}_critic.pth')

    def load_agent_params(self, title: str) -> None:
        """
        Load agents NN parameters

        Args:
            title: main name of files
                (full name would be {title}_critic/{title}_actor)

        """
        self.pi.load_state_dict(torch.load(f'{title}_actor.pth'))
        self.V.load_state_dict(torch.load(f'{title}_critic.pth'))

    def fit(self) -> None:
        """
        Create cycle of agents training

        """
        self.entropy_coef = max(self.entropy_coef * self.entropy_coef_decay,
                                1e-4)
        state, action, log_prob_a, reward, done, next_state = map(
            torch.FloatTensor, zip(*self.memory))
        reward = reward.reshape(-1, 1)
        done = done.reshape(-1, 1)

        # Advantage

        with torch.no_grad():
            value_fun = self.V(state)
            next_value_fun = self.V(next_state)
            delta = reward + (1-done) * self.gamma * \
                next_value_fun - value_fun

            advantages = torch.zeros_like(reward)
            last_adv = 0
            for ind in range(len(state)-1, -1, -1):
                advantages[ind] = delta[ind] + \
                    self.gamma*self.lambda_v*last_adv
                last_adv = advantages[ind]

            target = advantages + value_fun
            advantages = (advantages - advantages.mean()
                          ) / (advantages.std() + 1e-4)

        for n in range(self.epochs):
            dataset = TensorDataset(
                state, action, log_prob_a, advantages, target)
            loader_data = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True)
            for batch in loader_data:
                s = batch[0]
                a = batch[1]
                log_p_a = batch[2]
                adv = batch[3].reshape(-1, 1)
                target_batch = batch[4]

                # Policy

                log_prob_a_new, entropy = self.get_action(s, a)
                ratio = torch.exp(log_prob_a_new-log_p_a).reshape(-1, 1)
                self.optim_pi.zero_grad()
                clip_val = torch.clamp(ratio, 1-self.epsilon,
                                       1+self.epsilon)*adv
                loss_politic = torch.mean(-torch.min(
                    ratio*adv, clip_val) - self.entropy_coef * entropy)
                loss_politic.backward()
                torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
                self.optim_pi.step()

                # Value function

                v_fun = self.V(s)
                self.optim_v.zero_grad()
                loss_v_fun = torch.mean((v_fun-target_batch)**2)
                loss_v_fun.backward()
                torch.nn.utils.clip_grad_norm_(self.V.parameters(), 0.5)
                self.optim_v.step()
        self.reset_memory()
