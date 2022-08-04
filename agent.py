import numpy as np
class BanditAgent():
    def __init__(self, horizon, n_arms, repetitions):
        self.rewards = np.empty((horizon, repetitions))
        self.actions = np.empty_like(self.rewards)


    def update(self):
        """
        TODO: implement knowledge update strategy
        """
        pass
    
    def pull(self, arm):
        """_summary_
        TODO implement arm pulling strategy
        Parameters
        ----------
        arm : _type_
            _description_
        """
        pass

class UcbAgent(BanditAgent):
    def __init__(self, *args, delta):
        super.__init__(self,UcbAgent)
        self.delta = delta
        self.pull_count = np
    
    def update(self):
        # TODO update means
        pass
    
    def pull(self):
        pass

class EpsGreedyAgent(BanditAgent):
    