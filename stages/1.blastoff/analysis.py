import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StatisticalSignificance:

    def __init__(self, model):
        pass

    def compute(self):
        model.measure()
        pass

    def iterate(self, num_iterations=5):
        for idx in range(num_iterations):
            value = self.compute()

class StickModelMetric:

    def __init__(self):
        pass

    def sticks(self): np.ndarray
        pass

    def stddev(self):
        pass

    def mean(self):
        pass

    def measure(self, mean=0, std=0):
        pass

    