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
bandit_instance = bandit.StochasticBandit(arm_means, partial(random.binomial, 1))

policy_with_args = partial(policy.ucb1, a= lambda x: 1+x*log(x)**2)
#policy_with_args = policy.kl_ucb

experiment = StochasticBanditExperiment(horizon= 1000, repetitions= 1000)
experiment.run_parallel(bandit_instance, policy_with_args, show_progress= True)

plt.figure()
plt.plot(np.arange(experiment.horizon)+1, experiment.compute_regret(bandit_instance))
plt.show()


# %%
