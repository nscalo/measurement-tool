import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StatisticalSignificance:

    def __init__(self, model, prob=0.5):
        self.model = model
        self.prob = prob

    def compute(self):
        mean = self.model.mean()
        std = self.model.stddev()
        
        return self.model.measure(mean=mean, std=std)

    def iterate(self, num_iterations=5):
        value = 0
        n = 0
        for idx in range(1,num_iterations+1):
            value += (self.compute() / idx) / idx
            if value <= self.prob:
                n = idx
        return n
        

class StickModelMetric:

    def __init__(self):
        pass

    def sticks(self): np.ndarray
        return np.ones((6,2))

    def stddev(self):
        return np.zeros((6,1))

    def mean(self):
        return np.ones((6,1))

    def measure(self, mean=0, std=0):
        return ((mean - 3*std) / mean.shape[0]).sum()

    