import numpy as np
import time
'''
    distance - macierz sąsiedztwa pomiędzy lokalizacjami
    flow - macierz przepływu
    p_m - prawdopodobieństwo mutacji
    p_x - prawdopodobieństwo krzyżowania
    p_s - prawdopodobieństwo selekcji
    pop_size - rozmiar populacji
    gen - liczba pokoleń
    '''


class Qap:
    def __init__(self, distance, flow, p_m, p_x, Tour, pop_size, gen):
        self.distance = distance
        self.flow = flow
        self.p_m = p_m
        self.p_x = p_x
        self.Tour = Tour
        self.pop_size = pop_size
        self.gen = gen
        self.pop = None
        self.pop_value = None
        self.best_unit = None
        self.initialise()

    def initialise(self):
        self.pop = np.ones((self.pop_size, 1), dtype=int) * np.array(range(self.flow.shape[1]))
        self.pop_value = np.zeros(self.pop.shape[0])
        for i in range(self.pop_size):
            np.random.shuffle(self.pop[i, :])
        return

    def evaluate_mihau(self):

        for o in range(self.pop.shape[0]):
            cost = 0
            for i in range(0, self.flow.shape[0]):
                for j in range(0, self.flow.shape[0]):
                    cost += self.flow[i][j] * self.distance[int(self.pop[o][i])][int(self.pop[o][j])]
            self.pop_value[o] = cost
        return

    def evaluate(self):
        for i in range(self.pop.shape[0]):
            self.pop_value[i] = (self.distance * ((self.flow[self.pop[i], :])[:, self.pop[i]])).sum()
        return

    def isDone(self, t):
        return t > self.Tour

    def mutation(self):
        for pos in range(self.pop.shape[0]):
            if(np.random.random() > self.p_m):
                a, b = np.random.randint(0, self.pop.shape[1], 2)
                self.pop[pos][[a, b]] = self.pop[pos][[b, a]]
        return None

    def main(self):
        t = 0
        self.initialise()
        self.evaluate()
        self.best_unit = self.pop[0], self.pop_value[0]

        while not self.isDone(t):
            self.selection()
            self.crossover()
            self.mutation()
            self.evaluate()
            if np.min(self.pop_value) < self.best_unit[1]:
                self.best_unit = self.pop[np.argmin(self.pop_value)], np.min(self.pop_value)
            t += 1
        return

    def selection(self):
        acc = np.zeros(self.pop.shape, dtype=int)
        indices = np.arange(self.pop.shape[0])
        for i in indices:
            units = np.random.choice(indices, self.Tour, False)
            acc[i] = self.pop[np.argmin(self.pop_value[units])]
        self.pop = acc
        return

    def crossover(self):
        pattern = np.arange(self.pop.shape[1])

        def repair_unit(unit):
            indices = np.setdiff1d(pattern, np.unique(unit, return_index=True)[1])
            unit[indices] = np.setdiff1d(pattern, unit)
            return
        parents_indices = np.where((np.random.rand(self.pop.shape[0], 1) > self.p_x))[0]
        pairs = np.random.choice(parents_indices, (int(parents_indices.shape[0] / 2), 2), False)
        children = np.random.randint(0, 2, pairs.shape[0])
        for i in range(pairs.shape[0]):
            m, f = pairs[i]
            for child in range(children[i]):
                split = np.random.randint(self.pop[pairs[i][0]].shape[0])
                c = np.concatenate((self.pop[m][0:split], self.pop[f][split : self.pop[m].shape[0]]))
                repair_unit(c)
                self.pop[pairs[i][child]] = c
        return

    def result(self):
        print('Best: ' + str(test.best_unit[0]+1) + '\t==>\t' + str(test.best_unit[1]))
        return

test_flow_matrix = np.array([[0, 3, 0, 2],
                        [3, 0, 0, 1],
                        [0, 0, 0, 4],
                        [2, 1, 4, 0]])

test_distance_matrix = np.array([[0, 22, 53, 53],
                            [22, 0, 40, 62],
                            [53, 40, 0, 55],
                            [53, 62, 55, 0]])

x = np.array(np.loadtxt('data/flow_12.txt')).astype(int)
y = np.array(np.loadtxt('data/distance_12.txt')).astype(int)


start = time.time()

test = Qap(flow=x, distance=y, pop_size=100, gen=100, p_x=0.7, p_m=0.01, Tour=4)
for i in range(25):
    test.main()
    test.result()

stop = time.time()
duration = int(round((stop - start)*1000))
print('Duration: ' + str(duration) + ' ms')