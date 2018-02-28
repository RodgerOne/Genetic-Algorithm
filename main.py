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
            '''
            tmp = np.array(range(self.flow.shape[1]))
            np.random.shuffle(tmp)
            self.pop.append(tmp)
            '''
        return

    def evaluate(self):
        for i in range(self.pop.shape[0]):
            self.pop_value[i] = (self.distance * ((self.flow[self.pop[i], :])[:, self.pop[i]])).sum() / 2
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
        def repairUnit(unit):
            indices = np.setdiff1d(pattern, np.unique(unit, return_index=True)[1])
            unit[indices] = np.setdiff1d(pattern, unit)
            return
        parents_indices = np.where((np.random.rand(self.pop.shape[0], 1) > self.p_x))[0]
        #indexy rodziców - teraz ich sparuj (nieparzystego po prostu odrzuć) i zrób potomstwo : 1 lub 2 - jak jeden to zastąp losowego z rodziców (albo 1 albo 2), jak 2 dzieci to zastąp oboje rodziców
        #zrób sprawne dobieranie i naprawianie genów dzieci
        pairs = np.random.choice(parents_indices, (int(parents_indices.shape[0] / 2), 2), False) #indexy par - ostatniego nieparzystego odrzuca
        children = np.random.randint(1, 3, pairs.shape[0])
        for i in range(pairs.shape[0]):
            m, f = pairs[i]
            split = np.random.randint(self.pop[m].shape[0])
            c1 = np.concatenate((self.pop[m][np.arange(split)], self.pop[f][np.arange(start=split, stop=self.pop[m].shape[0])]))
            repairUnit(c1)
            self.pop[m] = c1
            if children[i] == 2:
                split = np.random.randint(self.pop[m].shape[0])
                c2 = np.concatenate(
                    (self.pop[m][0:split], self.pop[f][split : self.pop[m].shape[0]]))
                repairUnit(c2)
                self.pop[f] = c2
        return


flow_matrix = np.array([[0, 3, 0, 2],
                        [3, 0, 0, 1],
                        [0, 0, 0, 4],
                        [2, 1, 4, 0]])

distance_matrix = np.array([[0, 22, 53, 53],
                            [22, 0, 40, 62],
                            [53, 40, 0, 55],
                            [53, 62, 55, 0]])


start = time.time()

test = Qap(flow_matrix, distance_matrix, pop_size=100, gen=100, p_x=0.7, p_m=0.01, Tour=5)
for i in range(25):
    test.main()
    print('Best: ' + str(test.best_unit[0]) + '\t==>\t' + str(test.best_unit[1]))
stop = time.time()
duration = int(round((stop - start)*1000))
print('Duration: ' + str(duration) + ' ms')