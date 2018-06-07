import numpy as np


class RandomSearch:
    def __init__(self, distance, flow, pop_size):
        self.distance = distance
        self.flow = flow
        self.pop_size = pop_size
        self.pop = None
        self.pop_value = None

    def initialise(self):
        self.pop = np.ones((self.pop_size, 1), dtype=int) * np.array(range(self.flow.shape[1]))
        self.pop_value = np.zeros(self.pop.shape[0])
        for i in range(self.pop_size):
            np.random.shuffle(self.pop[i, :])
        pass

    def evaluate(self):
        for i in range(self.pop.shape[0]):
            self.pop_value[i] = (self.distance * ((self.flow[self.pop[i], :])[:, self.pop[i]])).sum()
        pass

    def run_random(self):
        self.initialise()
        self.evaluate()
        print('Best: ' + str(self.pop[np.argmin(self.pop_value)]) + '\t==>\t' + str(np.min(self.pop_value)))
        pass
