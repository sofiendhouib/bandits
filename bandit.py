import numpy as np

class StochasticBandit():
    """
        Class that represents a stochastic bandit.
    """
    def __init__(self, arm_means: list, reward_sampler):
        """ 
        Class constructor

        Parameters
        ----------
        arm_means : list or uni-dimensional array of float
            True expectations of the arms
        reward_sampler : function that creates a random sample
           The functions should take a mean and a size and return the rewards
        """
        self.arm_means = np.array(arm_means)
        self.reward_sampler = reward_sampler
    
    def sample(self, arm_index: int, repetitions: int):
        """ samples the received reward from arm having index arm_index
        Parameters
        ----------
        arm_index : integer
           index of the pulled/played arm
        repetitions : integer
            number of independent repetitions of the experiment

        Returns
        -------
        A table corresponding to realized rewards for each arm and for each repetition of the experiment
        """
        return self.reward_sampler(self.arm_means[arm_index], size= (repetitions, len(self.arm_means)))

    def create_reward_table(self, horizon: int, repetitions: int):
        """ Creates a reward table with entries for each time step and each repetition.

        Parameters
        ----------
        horizon : integer
           number of time steps
        repetitions : integer
            number of independent repetitions of the experiment

        Returns
        -------
        array of float
            a reward table with entries for each time step and each repetition.
        """
        # TODO : in case of additive noise that is identical across arms, horizon x repetitions is sufficient
        self.reward_table = self.reward_sampler(self.arm_means, size= (horizon, repetitions, len(self.arm_means)))
        u = np.unique(self.reward_table)
        if len(u) ==2 and np.all(np.isin(np.array([0,1]), u)):
            self.reward_table = self.reward_table.astype('uint8')
        return self


class ContextualBandit():
    """Class for contextual bandits
       For the moment, it is adapted to noise that is identically distributed across arms
    """
    
    def __init__(self, theta, n_arms, noise_sampler, context_generator= None, context_sampler= None):
        """ 
        Class constructor

        Parameters
        ----------
        theta : array of floats
            The preference vector
        n_arms : integer
            number of arms
        context_sampler: function
            describes how to generate a context at a given time step
        context_generator : generator
            generator of the sets of contexts observed at each time step along an experiment.
        noise_sampler: a function to sample additive noise
        """
        self.theta = theta
        self.n_arms = n_arms
        self.noise_sampler = noise_sampler
        self.dim = len(theta)
        self.context_generator = context_generator
        if context_generator is None:
            if context_sampler is None: raise ValueError("if the context generator is None then the context sampler must not be None")    
            else:    self.context_sampler = context_sampler

    def initialize(self, horizon: int, repetitions: int):
        if self.context_generator is None:
            self.create_context_generator(horizon)
        self.create_noise_table(horizon, repetitions)
        return self
    
    def create_context_generator(self, horizon):
        def generator(horizon):
            for _ in range(horizon): 
                yield self.context_sampler(self.dim, self.n_arms)
        self.context_generator = generator(horizon)
    
    def create_noise_table(self, horizon, repetitions):
        self.noise_table = self.noise_sampler(size= (horizon, repetitions))
        return self

class MultiTaskContextualBandit():
    
    def __init__(self, Theta, n_arms, noise_sampler, context_sampler, context_generator= 0):
        self.Theta = Theta # matrix 
        self.n_arms = n_arms
        self.n_users = len(Theta)
        self.noise_sampler = noise_sampler
        self.dim = Theta.shape[1]

        self.context_generator = context_generator
        if isinstance(context_generator, int):  
            self.context_seed = context_generator
            self.context_generator = None
        
        
        self.context_sampler = context_sampler
    
    def create_context_generator(self, horizon):
        """_summary_

        Parameters
        ----------
        horizon : _type_
            _description_
        """
        def generator(horizon):
            random_generator = np.random.default_rng(self.context_seed)
            for _ in range(horizon): 
                yield self.context_sampler(self.dim, self.n_arms, random_generator)
        self.context_generator = generator(horizon)
    
    def create_noise_table(self, horizon):
        self.noise_table = self.noise_sampler(size= (horizon))
        return self