import numpy as np
from tqdm import tqdm

random = np.random.default_rng()

# TODO create an experiment class with attributes such as horizon and repetitions, and functions for plotting etc

class StochasticBanditExperiment():
    """ Class of a stochastic bandit experiment
    """
    def __init__(self, horizon: int, repetitions: int):
        """

        Parameters
        ----------
        horizon : int
            number of time steps
        repetitions : int
            number of independent experiment repetitions
        """
        self.horizon = horizon
        self.repetitions = repetitions
        self.arm_history = None
        self.reward_history = None
    
    def run_sequential(self, bandit, policy, show_progress= False):
        pass
    
    def run_parallel(self, bandit, policy, show_progress= False):
        """Runs a stochastic bandit experiment and stores the history of arm selection and
        received rewards for all repetitions. The repetitions are ran in parallel in a vectorize way

        Parameters
        ----------
        bandit : bandit instance
            instance of the class StochasticBandit
        policy : object of class StochasticBanditPolicy
            a policy that only takes the number of pulls and the cumulated reward as an input. All other hyperparameters are fixed a priori
        show_progress : bool, optional
            Whether to show a progress bar. Set to False by default.

        """

        bandit.create_reward_table(self.horizon, self.repetitions)

        arms = []
        rewards = []
        # initialization: pull each arm once
        ones_rep = np.ones(self.repetitions, dtype= 'uint8')
        n_arms = len(bandit.arm_means)
        assert n_arms <= self.horizon, "The horizon must be at least equal to the number of arms !"

        for t in range(n_arms):
            arms.append(t*ones_rep)
            rewards.append(bandit.reward_table[t, np.arange(self.repetitions), arms[-1]])

        cumul_rewards = np.array(rewards, dtype= 'float64')
        n_pulls = np.ones((n_arms, self.repetitions), dtype= "int")


        # run n_repetitions experiments in a vectorized way
        timesteps = range(n_arms, self.horizon)
        if show_progress: timesteps = tqdm(timesteps)

        for t in timesteps:
            
            # play arm / pull arm
            arms.append(policy.play_arm(cumul_rewards, n_pulls, t))
        
            # collect reward from pulled arm
            rewards.append(bandit.reward_table[t,np.arange(self.repetitions),arms[-1]])

            # update the empirical mean of the played arm
            act_rep_tuple =  (arms[-1], np.arange(self.repetitions)) 
            cumul_rewards[act_rep_tuple] += rewards[-1]
            n_pulls[act_rep_tuple] += 1
        
        self.arm_history = np.array(arms, dtype= 'uint8')
        self.reward_history= np.array(rewards)
        return self
    
    def compute_regret(self, bandit, take_mean= True):
        """ Computes the cumulative regret.

        Parameters
        ----------
        bandit : instance of class StochasticBandit
        
        take_mean : bool
            whether to take the mean over repetitions. Set to True by default

        Returns
        -------
        cumulative regret either per repetition or averaged over repetitions
            _description_
        """
        assert self.arm_history is not None and self.reward_history is not None, "Please run an experiment before plotting the regret !"
        
        expected_oracle_rewards = np.ones(self.horizon)*bandit.arm_means.max()
        if take_mean:
            expected_rewards = np.mean(bandit.arm_means[self.arm_history], axis= 1)
            return np.cumsum(expected_oracle_rewards - expected_rewards)
        else:
            expected_rewards = bandit.arm_means[self.arm_history]
            return  np.cumsum(expected_oracle_rewards[:,None] - expected_rewards, axis= 0)
        
    def _compute_regret_old(self, bandit, which = "pseudo"):
        assert self.arm_history is not None and self.reward_history is not None, "Please run an experiment before plotting the regret !"
        which_lower = which.lower()
        assert which_lower in ["realized", "pseudo", "semi"], "the 'which' argument can be either 'realized', 'pseudo' or 'semi'"
        
        if which_lower == "realized":
            oracle_realized_rewards = bandit.reward_sampler(bandit.arm_means.max(), self.reward_history.shape)
            regret = np.cumsum(oracle_realized_rewards - self.reward_history, axis= 0)
        
        if which_lower == "pseudo":
            expected_rewards = bandit.arm_means[self.arm_history]
            expected_oracle_rewards = np.ones(self.reward_history.shape[0])*bandit.arm_means.max()
            regret =  np.cumsum(expected_oracle_rewards[:,None] - expected_rewards, axis= 0)
        
        if which_lower == "semi":
            expected_oracle_rewards = np.ones(self.reward_history.shape[0])*bandit.arm_means.max()
            regret = np.cumsum(expected_oracle_rewards[:,None] - self.reward_history, axis= 0)

        return regret

def compute_regret_old(rewards, actions, bandit, which = "pseudo"):
        assert actions is not None and rewards is not None, "Please run an experiment before plotting the regret !"
        which_lower = which.lower()
        assert which_lower in ["realized", "pseudo", "semi"], "the 'which' argument can be either 'realized', 'pseudo' or 'semi'"
        
        if which_lower == "realized":
            oracle_realized_rewards = bandit.reward_sampler(bandit.arm_means.max(), rewards.shape)
            regret = np.cumsum(oracle_realized_rewards - rewards, axis= 0)
        
        if which_lower == "pseudo":
            expected_rewards = bandit.arm_means[actions]
            expected_oracle_rewards = np.ones(rewards.shape[0])*bandit.arm_means.max()
            regret =  np.cumsum(expected_oracle_rewards[:,None] - expected_rewards, axis= 0)
        
        if which_lower == "semi":
            expected_oracle_rewards = np.ones(rewards.shape[0])*bandit.arm_means.max()
            regret = np.cumsum(expected_oracle_rewards[:,None] - rewards, axis= 0)

        return regret
# def compute_mistakes(arms, bandit):
#     best_ind = np.argmax(bandit.arm_means)
#     return np.cumsum(arms != best_ind, axis= 0)

class ContextualBanditExperiment():
    """ Class for a contextual bandit experiment
    """
    def __init__(self, horizon, repetitions):
        self.horizon = horizon
        self.repetitions = repetitions
        self.reward_history = None
        self.reward_history_oracle = None

    def run_parallel(self, bandit, agent):
        # initialize bandit and agent with information from the experiment
        bandit.initialize(self.horizon, self.repetitions)
        agent.initialize(bandit, self.repetitions)
        
        arm = [] #arm,
        expected_reward = [] # expectation of realized reward given the selected arm
        oracle_expected_reward = []

        for t in tqdm(range(1, self.horizon+1)):

            cxt_mat = next(bandit.context_generator) # context set
            oracle_reward_pool = cxt_mat @ bandit.theta # oracle rewards
            arm_played = agent.play_arm(cxt_mat, t) # select arm
        
            x_played = cxt_mat[arm_played].T
            y_played = oracle_reward_pool[arm_played] + bandit.noise_table[t-1]
            
            agent.update(x_played, y_played, t) # Update parameters upon observed context and collected reward

            # Store history for plotting
            arm.append(arm_played)
            expected_reward.append(bandit.theta @ x_played)
            oracle_expected_reward.append(np.max(oracle_reward_pool))

        self.reward_history = np.array(expected_reward)
        self.reward_history_oracle = np.array(oracle_expected_reward)

        return self
    
    def compute_regret(self, take_mean= True):
        assert self.reward_history is not None and self.reward_history_oracle is not None, "Please run an experiment before plotting the regret !"
        if take_mean:
            return np.cumsum(np.mean(self.reward_history_oracle[:,None] - self.reward_history, axis= 1))
        else:
            return np.cumsum(self.reward_history_oracle[:,None] - self.reward_history, axis= 0)
        



class MultiTaskContextualBanditExperiment():
    
    def __init__(self, horizon):
        self.horizon = horizon
        self.reward_history = None
        self.reward_history_oracle = None
        
    def run(self, bandit, agent):
        # TODO: also separate regret by user
        #if bandit.context_generator is None:
        bandit.create_context_generator(self.horizon)
        bandit.create_noise_table(self.horizon)
        arm = [] #arm,
        expected_reward = [] # expectation of realized reward given the selected arm
        oracle_expected_reward = []
        Theta_error = []

        agent.initialize(bandit)

        for t in range(1, self.horizon+1):

            u = random.integers(0, bandit.n_users) # select a user
            cxt_mat = next(bandit.context_generator) # generated a context
            
            oracle_reward_pool = cxt_mat @ bandit.Theta[u]
            
            # select arm
            arm_pull = agent.play_arm(cxt_mat, u, t)
            #arm_pull = random.integers(0, bandit.n_arms)
            #arm_pull = np.argmax(oracle_reward_pool)
            
            x_pull = cxt_mat[arm_pull].T
            y_pull = oracle_reward_pool[arm_pull] + bandit.noise_table[t-1]
            
            # Update estimated theta
            agent.update(x_pull, y_pull, u, t)

            # Store history for plotting
            arm.append(arm_pull)
            expected_reward.append(bandit.Theta[u] @ x_pull)
            oracle_expected_reward.append(np.max(oracle_reward_pool))
            # Theta_error.append(np.linalg.norm(np.linalg.lstsq(agent.A_aug, agent.b_aug)[0] - (bandit.Theta.T).flatten()))

        self.reward_history = np.array(expected_reward) 
        self.reward_history_oracle = np.array(oracle_expected_reward) 

        return self
    
    def compute_regret(self, take_mean= True):
        assert self.reward_history is not None and self.reward_history_oracle is not None, "Please run an experiment before plotting the regret !"
        if take_mean:
            return np.cumsum(np.mean(self.reward_history_oracle[:,None] - self.reward_history, axis= 1))
        else:
            return np.cumsum(self.reward_history_oracle[:,None] - self.reward_history, axis= 0)
        