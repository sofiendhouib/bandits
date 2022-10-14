import numpy as np

class StochasticBandit():
    def __init__(self, arm_means, reward_sampler):
        self.arm_means = np.array(arm_means)
        self.reward_sampler = reward_sampler
    
    def sample(self, arm_index, repetitions):
        return self.reward_sampler(self.arm_means[arm_index], size= (repetitions, len(self.arm_means)))

    def create_reward_table(self, horizon, repetitions):
        # TODO : in case of additive noise that is identical across arms, horizon x repetitions is sufficient
        self.reward_table = self.reward_sampler(self.arm_means, size= (horizon, repetitions, len(self.arm_means)))
        u = np.unique(self.reward_table)
        if len(u) ==2 and np.all(np.isin(np.array([0,1]), u)):
            self.reward_table = self.reward_table.astype('uint8')
        return self


class ContextualBandit():
    
    def __init__(self, theta, n_arms, noise_sampler, context_sampler, context_generator= None):
        """ Class for contextual bandits
        For the moment, it is adapted to noise that is identically distributed across arms

        Parameters
        ----------
        theta : The preference vector
            _description_
        n_arms : number of arms
            _description_
        context_generator : either a stochastic sampler or a given generator
            _description_
        noise_sampler: a function to sample additive noise
        """
        self.theta = theta
        self.n_arms = n_arms
        self.noise_sampler = noise_sampler
        self.dim = len(theta)
        self.context_generator = context_generator
        self.context_sampler = context_sampler
    
    def create_context_generator(self, horizon):
        def generator(horizon):
            for _ in range(horizon): 
                yield self.context_sampler(self.dim, self.n_arms)
        self.context_generator = generator(horizon)
    
    def create_noise_table(self, horizon, repetitions):
        self.noise_table = self.noise_sampler(size= (horizon, repetitions))
        return self

class MultiTaskContextualBandit():
    
    def __init__(self, Theta, n_arms, Laplacian, noise_sampler, sigma, context_sampler, context_generator= 0):
        self.Theta = Theta # matrix 
        self.n_arms = n_arms
        self.n_users = len(Theta)
        self.L = Laplacian
        self.noise_sampler = noise_sampler
        self.dim = Theta.shape[1]

        self.context_generator = context_generator
        if isinstance(context_generator, int):  
            self.context_seed = context_generator
            self.context_generator = None
        
        
        self.context_sampler = context_sampler
        self.sigma = sigma
    
    def create_context_generator(self, horizon):
        def generator(horizon):
            random_generator = np.random.default_rng(self.context_seed)
            for _ in range(horizon): 
                yield self.context_sampler(self.dim, self.n_arms, random_generator)
        self.context_generator = generator(horizon)
    
    def create_noise_table(self, horizon):
        self.noise_table = self.sigma*self.noise_sampler(size= (horizon))
        return self