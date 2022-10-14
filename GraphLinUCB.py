# %%
import numpy as np
from math import sqrt, log
from matplotlib import pyplot as plt
from numpy.random import default_rng
random = default_rng()
from tqdm import tqdm
from policy import GraphLinUCB
from bandit import MultiTaskContextualBandit
from experiment import bandit_multitask_experiment
import utils

# %%
n_users = 10
dim = 5
n_arms= 500

Adj = utils.generate_graph(n_users, p= 1.0)
L = utils.random_walk_laplacian(Adj)



Theta_0 = random.standard_normal(size=(n_users, dim))
gamma = 7.0
Theta = np.linalg.solve(np.eye(n_users) + gamma*0.5*(L.T+L),  Theta_0)
# Theta = np.linalg.solve(np.eye(n_users) + gamma*0.5*(L.T+L),  Theta_0)/sqrt(dim)

Theta /= np.linalg.norm(Theta, axis= 1, keepdims= True)


X =  random.standard_normal(size= (n_arms, dim))
X /= np.linalg.norm(X, axis= 1, keepdims= True)
def context_sampler(dim, n_arms, random_generator):
    # return  X#random.standard_normal(size= (n_arms, dim))
    return X[np.random.choice(range(len(X)), size= 50)]

horizon = 1000
noise_sampler = random.standard_normal


repetitions = 30

rewards = []
rewards_oracle = []
errors = []
# player = GraphLinUCB_old(a= 1.0, delta= 0.01) 
player = GraphLinUCB(a= 1.0, delta= 0.01) 

for i in tqdm(range(repetitions)):
    bandit = MultiTaskContextualBandit(Theta, n_arms= n_arms, Laplacian= L, context_sampler= context_sampler, 
                                        noise_sampler= noise_sampler, sigma= 0.01, context_generator= 2)
    expected_reward, oracle_expected_reward, Theta_error = bandit_multitask_experiment(bandit, player, horizon= horizon)
    rewards.append(expected_reward)
    rewards_oracle.append(oracle_expected_reward)
    errors.append(Theta_error)
rewards = np.array(rewards).T
rewards_oracle = np.array(rewards_oracle).T
errors = np.array(errors).T

# %%
rewards_cumul = np.cumsum(rewards, axis= 0)
rewards_orcale_cumul = np.cumsum(rewards_oracle, axis= 0)
meanRegret = rewards_orcale_cumul - rewards_cumul
plt.figure()
plt.plot(meanRegret.mean(axis= 1))
plt.show()

# %%
