from os import times_result
import numpy as np
from tqdm import tqdm

def run_experiment(bandit_instance, policy_with_args, horizon, repetitions, show_progress= False):
    bandit_instance.create_reward_table(horizon, repetitions)
    #reward_tensor = random.binomial(1, true_means, (horizon, n_repetitions, n_arms)).astype('int8')

    arms = []
    rewards = []
    # initialization: pull each arm once
    ones_rep = np.ones(repetitions, dtype= 'uint8')
    n_arms = len(bandit_instance.arm_means)
    assert n_arms <= horizon, "The horizon must be at least equal to the number of arms !"

    for t in range(n_arms):
        arms.append(t*ones_rep)
        rewards.append(bandit_instance.reward_table[t, np.arange(repetitions), arms[-1]])

    cumul_rewards = np.array(rewards, dtype= 'float64')
    n_pulls = np.ones((n_arms, repetitions), dtype= "int")


    # run n_repetitions experiments in a vectorized way
    timesteps = range(n_arms, horizon)
    if show_progress: times_result = tqdm(timesteps)

    for t in timesteps:
        
        # play arm / pull arm
        arms.append(policy_with_args(cumul_rewards, n_pulls, t))
    
        # collect reward from pulled arm
        rewards.append(bandit_instance.reward_table[t,np.arange(repetitions),arms[-1]])

        # update the empirical mean of the played arm
        act_rep_tuple =  (arms[-1], np.arange(repetitions)) 
        cumul_rewards[act_rep_tuple] += rewards[-1]
        n_pulls[act_rep_tuple] += 1
    return np.array(arms, dtype= 'uint8'), np.array(rewards)

def compute_regret(rewards, arms, bandit_instance, which = "pseudo"):
    which_lower = which.lower()
    assert which_lower in ["realized", "pseudo", "semi"], "the 'which' argument can be either 'realized', 'pseudo' or 'semi'"
    
    if which_lower == "realized":
        oracle_realized_rewards = bandit_instance.reward_sampler(bandit_instance.arm_means.max(), rewards.shape)
        regret = np.cumsum(oracle_realized_rewards - rewards, axis= 0)
    
    if which_lower == "pseudo":
        expected_rewards = bandit_instance.arm_means[arms]
        expected_oracle_rewards = np.ones(rewards.shape[0])*bandit_instance.arm_means.max()
        regret =  np.cumsum(expected_oracle_rewards[:,None] - expected_rewards, axis= 0)
    
    if which_lower == "semi":
        expected_oracle_rewards = np.ones(rewards.shape[0])*bandit_instance.arm_means.max()
        regret = np.cumsum(expected_oracle_rewards[:,None] - rewards, axis= 0)

    return regret

def compute_mistakes(arms, bandit_instance):
    best_ind = np.argmax(bandit_instance.arm_means)
    return np.cumsum(arms != best_ind, axis= 0)


def contextual_bandit_experiment(bandit, player, horizon, repetitions):
    if bandit.context_generator is None:
        bandit.create_context_generator(horizon)
    bandit.create_noise_table(horizon, repetitions)
    arm = [] #arm,
    expected_reward = [] # expectation of realized reward given the selected arm
    oracle_expected_reward = []

    player.initialize(bandit, repetitions)
    
    for t in tqdm(range(1, horizon+1)):
        # generate a set of contexts
        #cxt_mat = (2*random.random((n_arms,dim))-1)/dim
        cxt_mat = next(bandit.context_generator)
        
        oracle_reward_pool = cxt_mat @ bandit.theta
        
        # select arm
        arm_played = player.play_arm(cxt_mat, t)
        
        x_played = cxt_mat[arm_played].T
        y_played = oracle_reward_pool[arm_played] + bandit.noise_table[t-1]
        
        # Update estimated theta
        player.update(x_played, y_played, t)

        # Store history for plotting
        arm.append(arm_played)
        expected_reward.append(bandit.theta @ x_played)
        oracle_expected_reward.append(np.max(oracle_reward_pool))

    return np.array(expected_reward), np.array(oracle_expected_reward)

def bandit_multitask_experiment(bandit, player, horizon):
    # TODO: also separate regret by user
    if bandit.context_generator is None:
        bandit.create_context_generator(horizon)
    bandit.create_noise_table(horizon)
    arm = [] #arm,
    expected_reward = [] # expectation of realized reward given the selected arm
    oracle_expected_reward = []
    Theta_error = []

    player.initialize(bandit)

    for t in range(1, horizon+1):

        u = t % bandit.n_users#random.integers(0, bandit.n_users) # select a user
        cxt_mat = next(bandit.context_generator) # generated a context
        
        oracle_reward_pool = cxt_mat @ bandit.Theta[u]
        
        # select arm
        arm_pull = player.play_arm(cxt_mat, u, t)
        #arm_pull = random.integers(0, bandit.n_arms)
        #arm_pull = np.argmax(oracle_reward_pool)
        
        x_pull = cxt_mat[arm_pull].T
        y_pull = oracle_reward_pool[arm_pull] + bandit.noise_table[t-1]
        
        # Update estimated theta
        player.update(x_pull, y_pull, u, t)

        # Store history for plotting
        arm.append(arm_pull)
        expected_reward.append(bandit.Theta[u] @ x_pull)
        oracle_expected_reward.append(np.max(oracle_reward_pool))
        # Theta_error.append(np.linalg.norm(np.linalg.lstsq(player.A_aug, player.b_aug)[0] - (bandit.Theta.T).flatten()))

    return np.array(expected_reward), np.array(oracle_expected_reward), Theta_error