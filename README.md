# Some bandit algorithms

Efficient implementation of several bandit algorithms.

## Implemented algorithms

We implement the following algorithms depending on the bandit type

### Stochastic bandits

* $\epsilon$-greedy
* UCB
* MOSS
* Optimally Confident UCB
* AdaUCB
* Thompson sampling for Bernoulli rewards

#### TODO:

* [ ] Gradient bandits
* [ ] Thompson sampling for other families of distributions

### Contextual bandits

* LinUCB
* LALasso
* GraphUCB

#### TODO:

* [X] Directly updating logdet instead of det
* [ ] Linear Thompson sampling (3 variants)
* [ ] Generalized linear model
* [ ] ILOVETOCONBANDITS
* [ ] SupLinUCB

## On repeating experiments

For a bandit algorithm, one is mainly interested in the expected regret, which is estimated via several runs of the bandit algorithm.
We implement repetitions in two ways:

* Classic way: a for loop to iterate over the different iterations.
* Vectorized way: At each time step, several agents/players play in parallel, after separately having observed iid random variables for the rewards.
  This considerably reduces the time needed for repetitions.

## General description of the code:

* `experiments.py`: contains functions that perfom an experiment. As arguments, they take a bandit instance and a policy/player/agent, along with a horizon and a number of repetitions.
* `bandit.py`: classes for several bandit instances. For the moment, instance of stochastic, contextual and multi-task contextual bandits
* `policy.py`: classes for different policies.
