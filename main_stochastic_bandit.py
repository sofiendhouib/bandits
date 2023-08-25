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
from experiment import StochasticBanditExperiment


arm_means = [0.1, 0.2]

experiment = StochasticBanditExperiment(horizon= 10000, repetitions= 1000)
bandit_instance = bandit.StochasticBandit(arm_means, partial(random.binomial, 1))
# agent = policy.UCB1(a= lambda x: 1+x*log(x)**2)
agent = policy.AdaUCB(horizon= experiment.horizon)

experiment.run_parallel(bandit_instance, agent, show_progress= True)

plt.figure()
plt.plot(np.arange(experiment.horizon)+1, experiment.compute_regret(bandit_instance))
plt.show()


# %%
