#%%
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from math import log
from numpy import random

#%%

#xMean = np.zeros((2, horizon))


def epsilon_greedy(emp_means, n_repetitions, explore_proba):
    explore = random.binomial(1, explore_proba, size= n_repetitions).astype(np.bool8)
    action = np.argmax(emp_means, axis= 0)    
    action_explore = action[explore]
    # for exploration, make sure the selected action is not the greedy one. 
    # It is made sure of artificially here
    action[explore] = random.randint(0,n_arms-1, size= sum(explore))
    incorrect_explore_inds = action[explore] == action_explore
    action[explore][incorrect_explore_inds] += 1
    return action

def ucb1(emp_means, n_pulls, n_repetitions, delta):
    ucb = emp_means + np.sqrt(2*log(1/delta)/n_pulls)
    return np.argmax(ucb, axis= 0)

def lcb1(emp_means, n_pulls, n_repetitions, delta):
    lcb = emp_means - np.sqrt(2*log(1/delta)/n_pulls)
    return np.argmax(lcb, axis= 0)


n_repetitions= 1000
true_means = np.array([0.9, 0.2])
n_arms = len(true_means)
horizon = 500

plt.figure()
for delta in np.logspace(-6,-2,5):
    print(f"delta={delta}") 
    
    actions = -np.ones((horizon, n_repetitions),dtype=np.int64)
    rewards = -np.ones((horizon, n_repetitions))

    # initialization: pull each arm once
    for k in range(n_arms):
        actions[k] = k
        rewards[k] = random.binomial(1, true_means[actions[k]])

    emp_means = rewards[:n_arms,:].copy()
    n_pulls = np.ones((n_arms, n_repetitions))



#%% run n_repetitions experiments in a vectorized way

    for t in range(n_arms, horizon):
        # play action according to a certain policy

        # for the moment there is some wasteful computation: greedy and exploration are
        # computed for all repetitions
        
        actions[t] = epsilon_greedy(emp_means, n_repetitions, t**(-1/3))
        #actions[t] = ucb1(emp_means, n_pulls, n_repetitions, delta)
        #actions[t] = lcb1(emp_means, n_pulls, n_repetitions, delta)
        # observe reward
        rewards[t] = random.binomial(1,true_means[actions[t]])

        # update the empirical mean of the played arm
        mean_idx_weight = 1/(t - n_arms + 2)
        update_tuple =  (actions[t],range(n_repetitions)) 
        emp_means[update_tuple] = (1-mean_idx_weight)*emp_means[update_tuple]\
                                + mean_idx_weight*rewards[t]
        n_pulls[update_tuple] += 1

        agent_cumul_reward = np.cumsum(rewards, axis= 0)
        #oracle_cumul_reward = np.cumsum(random.binomial(1, np.max(true_means), size= (horizon,n_repetitions)), axis= 0)
        #regret = (oracle_cumul_reward - agent_cumul_reward)#/(np.arange(horizon)[:,None]+1)
        oracle_expec_cumul_reward = (np.arange(horizon)+1)*true_means.max()

        #plt.figure()
        #plt.hist(action, density= True)

        #%
    plt.plot(np.arange(horizon)+1,oracle_expec_cumul_reward - agent_cumul_reward.mean(axis=1), 
    linewidth= 2, label= f"$\delta={delta}$")
plt.legend()
# %%
