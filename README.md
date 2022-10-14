# Some bandit algorithms
Efficient implementation of several bandit algorithms.
## Implemented algorithms
We implement the following algorithms depending on the bandit type
### Stochastic bandits
* $\epsilon$-greedy
* UCB
* MOSS
* Thompson sampling for Bernoulli rewards

### Contextual bandits
* LinUCB
* LALasso
* GraphUCB

## On repeating experiments
For a bandit algorithm, one is mainly interested in the expected regret, which is estimated via several runs of the bandit algorithm.
We implement repetitions in two ways:
* Classic way: a for loop to iterate over the different iterations.
* Vectorized way: At each time step, several agents/players play in parallel, after separately having observed iid random variables for the rewards.
This considerably reduces the time needed for repetitions.
