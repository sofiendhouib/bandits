# %%
import numpy as np
from numpy.random import default_rng
random =default_rng(0)
from math import log
from matplotlib import pyplot as plt
import policy
from tqdm import tqdm
import bandit
from functools import partial
import experiment


arm_means = [0.1, 0.2]
bandit_instance = bandit.StochasticBandit(arm_means, partial(random.binomial, 1))
horizon = 100000
repetitions = 1000
policy_with_args = partial(policy.ucb1, a= lambda x: 1+x*log(x)**2)
#policy_with_args = policy.kl_ucb

actions, rewards = experiment.run_experiment(bandit_instance, policy_with_args, 
horizon= horizon, repetitions= repetitions, show_progress= True)

plt.figure()
horizon_range = (np.arange(horizon)+1)
realized_regret = experiment.compute_regret(rewards, actions, bandit_instance, 'realized')#(oracle_realized_cumul_reward - agent_cumul_reward)
semi_realized_regret = experiment.compute_regret(rewards, actions, bandit_instance, 'semi')
pseudo_regret = experiment.compute_regret(rewards, actions, bandit_instance, 'pseudo')
plt.plot(horizon_range, pseudo_regret.mean(axis= 1), linewidth= 2, label= 'expected')
plt.plot(horizon_range, realized_regret.mean(axis= 1), linewidth= 2)
plt.plot(horizon_range, semi_realized_regret.mean(axis=1), linewidth= 2)
# %%
