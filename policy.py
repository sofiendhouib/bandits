#%%
from math import log, sqrt
import numpy as np
from scipy.special import rel_entr, entr, expit
from numpy.random import default_rng
random =default_rng()
from sklearn.linear_model import Lasso as LassoSklearn
from abc import ABC, abstractmethod

class epsilon_greedy():
    """ class for a player that playes makes the greedy choice with a certain probability at each round.
    """
    def __init__(self, explore_proba= 'cubic'):
        self.explore_proba = explore_proba


def epsilon_greedy(cumul_rewards, n_pulls, t, explore_proba= 'cubic'):
        # for the moment there is some wasteful computation: greedy and exploration are
    # computed for all repetitions
    if explore_proba == "cubic": explore_proba = 1/t**1.5
    explore = random.binomial(1, explore_proba, cumul_rewards.shape[1]).astype('bool8')
    arm = np.argmax(cumul_rewards/n_pulls, axis= 0)    
    arm[explore] = random.integers(n_pulls.shape[0], size= sum(explore))
    return arm

def epsilon_greedy_2(cumul_rewards, n_pulls, t, n_repetitions, explore_proba):
    explore = random.binomial(1, explore_proba, size= n_repetitions).astype('bool8')
    arm = np.argmax(cumul_rewards/n_pulls, axis= 0)    
    arm_explore = arm[explore]
    # for exploration, make sure the selected arm is not the greedy one. 
    # It is made sure of artificially here
    arm[explore] = random.integers(n_pulls.shape[0]-1, size= sum(explore))
    incorrect_explore_inds = arm[explore] == arm_explore
    arm[explore][incorrect_explore_inds] += 1
    return arm

class IndexPolicy():
    """
    Parent class representing index policies
    """
    def play_arm(self, *args, **kwargs):
        return np.argmax(self.make_index(*args, **kwargs))

    def make_index(cumul_rewards, n_pulls):
        pass

class AdditiveConfidenceRadiusPolicy(IndexPolicy):
    """ Parent class representing ucb policies for which the 
    index is the sum of the empirical mean plus some radius
    """
    pass


def ucb1(cumul_rewards, n_pulls, t, a, sigma= 1.0):
    if callable(a): a = a(t)
    u = cumul_rewards/n_pulls + sigma*np.sqrt(2*log(a)/n_pulls)
    return np.argmax(u, axis= 0)

def moss(cumul_rewards, n_pulls, t, horizon_arms_ratio):
    rad = 2*np.sqrt(np.log(np.maximum(horizon_arms_ratio/n_pulls,1))/n_pulls)
    u = cumul_rewards/n_pulls + rad
    return np.argmax(u, axis= 0)

def __rev_rel_entr(x, p):
    return rel_entr(p,x) + rel_entr(1-p,1-x), (x-p)/(x*(1-x)), p/x**2 + (1-p)/(1-x)**2


def kl_ucb(cumul_rewards, n_pulls, t):
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
    rad = 2*log(1+t*log(t)**2)/n_pulls.flatten()
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

def myopic(cumul_rewards, n_pulls, t):
    arms = np.empty(n_pulls.shape[1], dtype= 'uint8')
    ksi = np.array([1,-1]) @ (2*cumul_rewards - n_pulls)
    arms[ksi>0] = 0
    arms[ksi<0] = 1
    arms[ksi==0] = random.integers(low= 0, high= 2, size= (ksi==0).sum())
    return arms

def myopic_softmax(cumul_rewards, n_pulls, t, temp):
    ksi = np.array([1,-1]) @ (2*cumul_rewards - n_pulls)
    probas = expit(-ksi/temp)
    return random.binomial(1, probas, size= len(ksi))


def Thompson_Bernoulli(cumul_rewards, n_pulls, t, a, b):
    # update prior to create posterior
    a += cumul_rewards
    b += n_pulls-cumul_rewards
    # sample from posterior
    from_posterior = random.beta(a, b, n_pulls.shape).astype("float32")
    return np.argmax(from_posterior, axis= 0)

def Thompson_Bernoulli_symmetric(cumul_rewards, n_pulls, t, a, b):
    # update prior to create posterior
    a += cumul_rewards
    b += n_pulls-cumul_rewards
    # sample from posterior
    from_posterior = random.beta(a, b, n_pulls.shape).astype("float32")
    return np.argmax(from_posterior, axis= 0)

class PolicyContextual():

    def __init__(self, l_reg= 1, delta= 0.1, theta_bound= 1.0):
        self.l_reg = l_reg
        self.delta = delta
        self.theta_bound = theta_bound

class LinUCB(PolicyContextual):
    # TODO make confidence radius a function of the determinant
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def play_arm(self, cxt_mat, t):
        # TODO Add sigma standard deviation parameter
        # dim = cxt_mat.shape[1]
        # sqrtbeta = sqrt(self.l_reg) *self.theta_bound+ sqrt(2*log(1/self.delta)+dim*log(1+t*self.theta_bound/(dim*self.l_reg)))
        sqrtbeta = sqrt(self.l_reg)*self.theta_bound + np.sqrt(2*log(1/self.delta) + self.log_det_diff)
        radius = sqrtbeta * np.sqrt(np.einsum("ij,jkl,ik->il", cxt_mat, self.invV, cxt_mat))
        ucb = cxt_mat @ self.theta_estim + radius
        return np.argmax(ucb, axis= 0)

    def initialize(self, bandit, repetitions):
        self.log_det_diff = np.zeros(repetitions)
        self.invV =  np.einsum("ij,k->ijk", 1/self.l_reg*np.eye(bandit.dim), np.ones(repetitions))
        self.b = np.zeros((bandit.dim,repetitions))
        self.theta_estim = np.einsum("ijk,jk->ik", self.invV, self.b)
        return self

    def update(self, x, y, t):
        # using Sherman-Morrison formula to update the inverse and determinant lemma to update the log determinant
        invV_x = np.einsum("ijk, jk -> ik", self.invV , x) # used for both formulas
        # Updating the determinant and the inverse respectively using the determinant lemma and the Sherman-Morrison formula
        x_invV_x = np.einsum("ik,ik->k", x, invV_x)
        self.log_det_diff += np.log1p(x_invV_x/self.l_reg)
        self.invV -= np.einsum("ik,jk -> ijk", invV_x, invV_x)/(1 + x_invV_x[None,None,:])
        self.b += y * x
        self.theta_estim = np.einsum("ijk,jk->ik", self.invV, self.b)
        return self

class LALasso():

    def __init__(self, theta_estim, l_reg= 1, delta= 0.1):
        self.l_reg = l_reg
        self.delta = delta
        self.theta_estim = theta_estim
        

    def play_arm(self, cxt_mat, t):
        return np.argmax(cxt_mat @ self.theta_estim, axis= 0) # greedy policy

    def initialize(self, bandit, repetitions):
        self.V =  np.zeros((bandit.dim, bandit.dim, repetitions))
        self.b = np.zeros((bandit.dim,repetitions))
        self.theta_estim = np.zeros((bandit.dim, repetitions))
        self.dim = bandit.dim
        return self

    def update(self, x, y, t):
        self.V += np.einsum("ik,jk->ijk", x, x)
        self.b += y * x
        self.lr = t/np.linalg.norm(self.V, axis= (0,1), ord= 2)
        l_t = self.l_reg*sqrt((4*log(t) + 2*log(self.dim))/t)
        # self.theta_estim = solve_lasso(V= self.V, b= self.b, l= l_t, 
        #                                theta= self.theta_estim, lr= self.lr, t= t)
        # lasso_estimator = Lasso(alpha= l_t, warm_start= True, fit_intercept= False, tol=1e-06, verbose= False)
        lasso_estimator = LassoSklearn(alpha= l_t, warm_start= True, fit_intercept= False)
        scale = sqrt(len(self.b)/t)
        for i in range(self.b.shape[1]):
            D, P  = np.linalg.eigh(self.V[:,:,i])
            D[np.abs(D)<1e-10] = 0.0
            X_lasso = P @ np.diag(np.sqrt(D)) @ P.T
            y_lasso, _, _ , _ = np.linalg.lstsq(X_lasso, self.b[:,i], rcond= None)
            if np.any(np.isnan(X_lasso)) or np.any(np.isnan(y_lasso)):
                print("NaN detected !")
                break
            lasso_estimator.fit(scale*X_lasso, scale*y_lasso)
            self.theta_estim[:,i] = lasso_estimator.coef_
        return self

class GraphLinUCB():
    """Implement the GraphLinUCB algorithm of Yang et al 2020
    """

    def __init__(self, laplacian, a= 1.0, delta= 0.01, sigma= 1.0):
        self.delta = delta
        self.a = a
        self.aL = self.a * (0.5*(laplacian + laplacian.T) + 0.01*np.eye(len(laplacian)))
        self.sigma = sigma
    
    def play_arm(self, cxt_mat, u, t):
        # invLambda_u = np.linalg.inv(self.Lambda[u])
        # print(np.linalg.norm(self.invLambda[u] - invLambda_u))
        # radius = self.beta[u] * np.sqrt(np.einsum("ki, ij, kj -> k", cxt_mat, invLambda_u, cxt_mat))
        radius = self.beta[u] * np.sqrt(np.einsum("ki, ij, kj -> k", cxt_mat, self.invLambda[u], cxt_mat))
        #radius = self.beta[u] * np.sqrt(np.einsum("ki, ij, kj -> k", cxt_mat, self.invA[u], cxt_mat))
        # theta_estim_vec, _, rank, _ = np.linalg.lstsq(self.A_aug, self.b_aug, rcond= None)
        theta_estim_u = self.theta_estim_vec[u*self.dim:(u+1)*self.dim]
        return np.argmax(cxt_mat @ theta_estim_u)# + radius)
    
    def get_theta(self):
        return self.theta_estim_vec.reshape((self.n_users, self.dim))

    def initialize(self, bandit):
        self.dim = bandit.dim
        self.n_users = bandit.n_users
        # Graph Laplacian
        # the symmetric part is used in the solution. Also, in the auhtors impl, they add 0.01
        
        # matrix A and its inverse
        self.A = 0.1 * np.einsum("k, ij -> kij", np.ones(bandit.n_users), np.eye(bandit.dim))
        self.invA = 10.0 * np.einsum("k, ij -> kij", np.ones(bandit.n_users), np.eye(bandit.dim)) # authors use a 0.1 constant
        # Matrix Lambda
        # XXX The second line is not indicated in the paper, unless when looking at the proof in the appendix
        # XXX it does not result in a linear regret though
        self.aL_diag_stack = np.einsum("k, ij -> kij", np.diag(self.aL), np.eye(bandit.dim))
        self.Lambda = self.A + 2*self.aL_diag_stack + np.einsum("lk, kij -> lij", self.aL**2, self.invA)
        self.Lambda -= (np.diag(self.aL)**2)[:,None,None] * self.invA
        self.invLambda = np.linalg.inv(self.Lambda)

        # Augmented invertible matrix of Equation (4) that estimates Theta
        self.invA_aug =   np.kron(np.linalg.inv(self.aL), np.eye(self.dim))
        self.b_aug = np.zeros(bandit.dim * bandit.n_users)
        self.theta_estim_vec = self.invA_aug @ self.b_aug
        self.eye_u = np.eye(bandit.n_users)
        self.beta = np.zeros(bandit.n_users)
        
        return self
    
    def update(self, x, y, u, t):
        # update the inverse from Equation (10)
        f = np.kron(self.eye_u[u], x) # augmented feature vector
        self.invA_aug -= rank1_update(self.invA_aug, f)
        self.b_aug[u*self.dim:(u+1)*self.dim] += y * x
        self.theta_estim_vec = self.invA_aug @ self.b_aug

        # update Lambda: recomputed at each time step. A and invA are incrementally updated
        # self.b[u] += y*x
        
        # updating inv Lambda for current user via Sherman-Morrison
        self.invLambda[u] -= rank1_update(self.invLambda[u], x)

        # Updating inv Lambda for the rest of users: more delicate
        users_but_u = np.array([v for v in range(self.n_users) if v != u], dtype= "int16")
        # prepare the vector of rank 1 update per user
        vec = rank1_update_decomposed(self.invA[u], x) # -outer(vec,vec) is the update due to the inverse parts
        aL_vec = np.outer(self.aL[u, users_but_u], vec) # the vector of update per user different from u
        # apply Sherman-Morrison
        invLambda_aL_vec = np.einsum("kij, kj -> ki", self.invLambda[users_but_u], aL_vec)
        self.invLambda[users_but_u] += np.einsum("ki, kj -> kij", invLambda_aL_vec, invLambda_aL_vec)\
                                    /(1 - np.einsum("ki, ki -> k", aL_vec, invLambda_aL_vec))[:,None,None]

        self.A[u] += np.outer(x,x)
        self.invA[u] -= rank1_update(self.invA[u], x)
        # self.Lambda = self.A + 2*self.aL_diag_stack + np.einsum("lk, kij -> lij", self.aL**2, self.invA)
        # self.Lambda -= (np.diag(self.aL)**2)[:,None,None] * self.invA

        # update beta
        detV_u = np.linalg.det(self.A[u] + self.aL[u,u]*np.eye(self.dim))
        beta_term_1 = self.sigma*sqrt(log(detV_u/(self.a**self.dim*self.delta**2)))
        # In the authors implementation, the theta in  the paper is replaced by a classic LSQ estimation, i.e. one
        # that does not consider the Laplacian
        theta_per_user = np.einsum("ki, kij -> kj" , self.b_aug.reshape(self.n_users, self.dim), self.invA)
        beta_term_2 =  np.linalg.norm(self.aL[u] @ theta_per_user)/sqrt(self.a)
        self.beta[u] = beta_term_1 + beta_term_2
        # XXX using a norm bound on the dot product aL * theta_user results in linear regret
        return self


def value_lasso(V, b, theta, l, t):
    return 0.5/t*(theta @ V - b) @ theta + l*np.linalg.norm(theta, ord= 1)

def rank1_update(invA, u):
    invA_u = invA @ u
    return np.outer(invA_u, invA_u)/(1 + u @ invA_u)

def rank1_update_decomposed(invA, u):
    """The Sherman Morrison formula updates the inverse with a rank one matrix
    that can be written as the outer product of a vector U by itself. This function outputs that
    U vector.
    """
    invA_u = invA @ u
    return invA_u/sqrt(1 + u @ invA_u)

def rank1_update_vectorized_last(invA, u):
    "Computes the increment to subtract"
    invA_u = np.einsum("ijk, jk -> ik", invA , u)
    return np.einsum("ik,jk -> ijk", invA_u, invA_u)/(1 + np.einsum("ik,ik->k", u, invA_u)[None,None,:])

def rank1_update_vectorized_first(invA, u):
    "Computes the increment to subtract"
    invA_u = np.einsum("ijk, jk -> ik", invA , u)
    return np.einsum("ik,jk -> ijk", invA_u, invA_u)/(1 + np.einsum("ik,ik->k", u, invA_u)[None,None,:])
