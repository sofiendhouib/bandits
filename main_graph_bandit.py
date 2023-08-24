# %%
import numpy as np
from math import sqrt, log
from matplotlib import pyplot as plt
from numpy.random import default_rng
random = default_rng()
from tqdm import tqdm
import policy
from bandit import MultiTaskContextualBandit
from experiment import bandit_multitask_experiment
import utils
import networkx as nx
plt.close('all')

# %%
n_users = 10
dim = 20
n_arms= 500
horizon = 150
noise_sampler = random.standard_normal
sigma = 0.0
repetitions = 20

#%% Generate signal
Theta_0 = random.standard_normal((n_users, dim))
# Adj = utils.graph_Erdos_Renyi(n_users, p= 0.1)
Adj, Theta_0 = utils.graph_rbf(node_num= n_users, dimension= dim, sparsity= 0.1, clusters= [[1,-0.3], [1,0.3]])
# Adj /= Adj.sum(axis=1, keepdims= True)

print(Adj)
print(np.where(Adj>0,1,0))
L = utils.random_walk_laplacian(Adj)

def laplacian_smooth_theta(Theta_0, gamma, laplacian):
    return np.linalg.solve(np.eye(n_users) + gamma*0.5*(laplacian.T+laplacian),  Theta_0)



# For Laplacian regularization
Theta = laplacian_smooth_theta(Theta_0= random.random((n_users, dim)), gamma= 7.0, laplacian= L)
Theta /= np.linalg.norm(Theta, axis= 1, keepdims= True) #normalizing

# For Network Lasso regularization

if dim == 2:
    graph = nx.from_numpy_array(Adj)
    plt.figure()
    utils.quiver_graph(graph= graph, pos= Theta_0, signal= Theta_0, alpha= 0.1)
    plt.title("vectors used to build the true signal")

    plt.figure()
    utils.quiver_graph(graph= graph, pos= Theta_0, signal= Theta, alpha= 0.1)
    plt.title("True signal")

# def context_sampler(dim, n_arms, random_generator):
#     X =  random_generator.random(size= (n_arms, dim))/sqrt(dim)
#     return X# / np.linalg.norm(X, axis= 1, keepdims= True)
X =  random.standard_normal(size= (n_arms, dim))
#X /= np.linalg.norm(X, axis= 1, keepdims= True)
def context_sampler(dim, n_arms, random_generator):
    return X[random_generator.choice(range(len(X)), size= 50)]




# player = GraphLinUCB_old(a= 1.0, delta= 0.01)
# TODO special step size selection for quadratic case
player_dict = {
                "GraphUCB": policy.GraphLinUCB(a= 1.0, delta= 0.01, laplacian= L), 
            }
#rewards_dict = dict(zip(player_dict.keys(), ({"player": None, "oracle": None} for _ in range(len(player_dict.keys())))))
regret_dict = dict(zip(player_dict.keys(), (None for _ in range(len(player_dict.keys())))))

#%%
for name, player in player_dict.items():
    rewards = []
    rewards_oracle = []
    errors = []
    for i in tqdm(range(repetitions)):

        bandit = MultiTaskContextualBandit(Theta, n_arms= n_arms, context_sampler= context_sampler, 
                                            noise_sampler= noise_sampler, sigma= sigma, context_generator= 2)
        expected_reward, oracle_expected_reward, Theta_error = bandit_multitask_experiment(bandit, player, horizon= horizon)
        rewards.append(expected_reward)
        rewards_oracle.append(oracle_expected_reward)
        errors.append(Theta_error)
    # rewards_dict[name]["player"] = np.array(rewards).T
    # rewards_dict[name]["oracle"] = np.array(rewards_oracle).T
    regret_dict[name] = np.array(rewards_oracle) - np.array(rewards)
    errors = np.array(errors).T

# %%
plt.figure()
for name, regret in regret_dict.items():
    # rewards_cumul = np.cumsum(reward["player"], axis= 0)
    # rewards_orcale_cumul = np.cumsum(reward["oracle"], axis= 0)
    # meanRegret = rewards_orcale_cumul - rewards_cumul
    regret_cumul = np.cumsum(regret, axis= 1)
    plt.plot(regret_cumul.mean(axis= 0), label= name)
    plt.plot(regret_cumul.T, 'k', alpha= 0.1)
    plt.savefig(f"{name}.pdf", format= 'pdf')
plt.legend()
plt.show()
# 
#%%
graph = nx.from_numpy_array(Adj)
if dim == 2:
    for name, player in player_dict.items():
        plt.figure()
        utils.quiver_graph(graph= graph, pos= Theta_0, signal= player.get_theta(), alpha= 0.1)
        plt.title(f"{name} solution")
       # plt.show()
        #plt.savefig(f"signal_{name}.png")
        print(np.linalg.norm(player.get_theta()-Theta)/np.linalg.norm(Theta))
# %%
