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



experiment = StochasticBanditExperiment(horizon= 10000, repetitions= 100)


bandit_instance = bandit.StochasticBandit([0.5, 0.6], partial(random.normal, scale= 1.0))

plt.figure()
for ratio in np.linspace(0.5,1.5,11):
    agent = policy.AdaUCB(horizon= experiment.horizon, sigma= ratio)
    # agent = policy.UCB1(a= lambda x: 1+x*log(x)**2)


    experiment.run_parallel(bandit_instance, agent, show_progress= True)

    plt.plot(np.arange(experiment.horizon)+1, experiment.compute_regret(bandit_instance), label= str(ratio))
    plt.legend()
plt.show()
# %%
