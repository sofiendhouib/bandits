#%%
from math import log
import numpy as np
from scipy.special import rel_entr, entr
from numpy.random import default_rng
random =default_rng()
from os import system
from numba import vectorize

def epsilon_greedy(cumul_rewards, n_pulls, explore_proba):
        # for the moment there is some wasteful computation: greedy and exploration are
    # computed for all repetitions
    
    explore = random.binomial(1, explore_proba, cumul_rewards.shape[1]).astype('bool8')
    action = np.argmax(cumul_rewards/n_pulls, axis= 0)    
    action[explore] = random.integers(n_pulls.shape[0], size= sum(explore))
    return action

def epsilon_greedy_2(cumul_rewards, n_pulls, n_repetitions, explore_proba):
    explore = random.binomial(1, explore_proba, size= n_repetitions).astype('bool8')
    action = np.argmax(cumul_rewards/n_pulls, axis= 0)    
    action_explore = action[explore]
    # for exploration, make sure the selected action is not the greedy one. 
    # It is made sure of artificially here
    action[explore] = random.integers(n_pulls.shape[0]-1, size= sum(explore))
    incorrect_explore_inds = action[explore] == action_explore
    action[explore][incorrect_explore_inds] += 1
    return action

def ucb1(cumul_rewards, n_pulls, a):
    u = cumul_rewards/n_pulls + np.sqrt(2*log(a)/n_pulls)
    return np.argmax(u, axis= 0)

def moss(cumul_rewards, n_pulls, horizon_arms_ratio):
    rad = 2*np.sqrt(np.log(np.maximum(horizon_arms_ratio/n_pulls,1))/n_pulls)
    u = cumul_rewards/n_pulls + rad
    return np.argmax(u, axis= 0)

def __rev_rel_entr(x, p):
    #return rel_entr(p,x) + rel_entr(1-p,1-x), np.diag((x-p)/(x*(1-x)))
    #return rel_entr(p,x) + rel_entr(1-p,1-x), (x-p)/(x*(1-x))
    return rel_entr(p,x) + rel_entr(1-p,1-x), (x-p)/(x*(1-x)), p/x**2 + (1-p)/(1-x)**2


def kl_ucb(cumul_rewards, n_pulls, a):
    """
    We implement the kl-ucb policy that uses the kl-divergence to build confidence bounds.
    For the moment, it can only be used with Bernoulli rewards.
    We distinguish the case of having empirical means in (0,1), and the case of extreme values 0 and 1
    that we treat separately. For (0,1), we compute the upper confidence bound using the Halley method 
    (second order analogue of Newton iterations).

    Parameters
    ----------
    cumul_rewards : _type_
        _description_
    n_pulls : _type_
        _description_
    a : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    rad = 2*log(a)/n_pulls.flatten()
    emp_means = (cumul_rewards/n_pulls).flatten()
    u= np.empty_like(emp_means)
    # to avoid problems at boundaries, we consider them separately
    u[emp_means >= 1] = 1
    u[emp_means <= 0] = 1-np.exp(-rad[emp_means<=0])
    inds = np.logical_and(emp_means < 1, emp_means > 0)
    rad_inds = rad[inds]
    means_inds = emp_means[inds]
    # initialization by a value for which KL is greater than the target value
    # I used the fact that maximum of two nonnegative quantities is smaller than their sum here
    ent_means = entr(means_inds)+entr(1-means_inds)
    mu = np.maximum(np.exp(-(rad_inds+ent_means)/means_inds), 
                    1-np.exp(-(rad_inds+ent_means)/(1-means_inds)))

    #x0 = np.ones_like(means_inds)-1e-8
    if np.any(inds):
        for i in range(3): # achieves error 1e-6 while newton 1e-3 with same number of iterations
            val, deriv, hess= __rev_rel_entr(mu, means_inds)
            mu -= 2*(val-rad_inds)*deriv/(2*deriv**2 - (val-rad_inds)*hess)
        # for i in range(3):
        #    val, deriv, _= __rev_rel_entr(mu, means_inds)
        #    mu -= (val-rad_inds)/deriv
        #    mu -= (np.log(val) - np.log(rad_inds))*val/deriv
        # print(np.max(np.abs(val-rad_inds)))
        u[inds] = mu.copy()

    return np.argmax(u.reshape(n_pulls.shape), axis= 0)

def halley_iter(x, target_val):
    pass

def myopic(cumul_rewards, n_pulls):
    centered = 2*cumul_rewards - n_pulls 
    return np.argmax(centered, axis= 0)

def Thompson_Bernoulli(cumul_rewards, n_pulls, a, b):
    # update prior to create posterior
    a += cumul_rewards
    b += n_pulls-cumul_rewards
    # sample from posterior
    from_posterior = random.beta(a, b, n_pulls.shape).astype("float32")
    return np.argmax(from_posterior, axis= 0)

# %%
