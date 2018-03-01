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

    def evaluate(self):
        for i in range(self.pop.shape[0]):
            self.pop_value[i] = (self.distance * ((self.flow[self.pop[i], :])[:, self.pop[i]])).sum()

    def isDone(self, t):
        return t > self.gen


    def mutation(self):
        mutants = np.nonzero(np.random.rand(self.pop.shape[0], self.pop.shape[1]) < self.p_m)
        for ind in range(mutants[0].shape[0]):
            r = np.random.randint(0, self.pop.shape[1])
            self.pop[ind][[mutants[1][ind], r]] = self.pop[ind][[r, mutants[1][ind]]]

    def main(self):
        t = 0
        self.initialise()
        self.evaluate()
        self.best_unit = self.pop[0], self.pop_value[0]

        while not self.isDone(t):
            #print(np.min(self.pop_value))
            self.selection_tournament()
            self.crossover()
            self.mutation()
            self.evaluate()
            if np.min(self.pop_value) < self.best_unit[1]:
                i = np.argmin(self.pop_value)
                self.best_unit = self.pop[i], self.pop_value[i]
            t += 1

    def selection_tournament(self):
        acc = np.zeros(self.pop.shape, dtype=int)
        indices = np.arange(self.pop.shape[0])
        for i in indices:
            units = np.random.choice(indices, self.Tour, False)
            acc[i] = self.pop[units[np.argmin(self.pop_value[units])]]
        self.pop = acc

    def crossover(self):
        pattern = np.arange(self.pop.shape[1])
        def repair_unit(unit):
            indices = np.setdiff1d(pattern, np.unique(unit, return_index=True)[1])
            new = np.setdiff1d(pattern, unit)
            np.random.shuffle(new)
            unit[indices] = new
        parents_indices = np.where((np.random.rand(self.pop.shape[0], 1) > self.p_x))[0]
        pairs = np.random.choice(parents_indices, (int(parents_indices.shape[0] / 2), 2), False)
        children = np.random.randint(0, 3, pairs.shape[0])
        for i in range(pairs.shape[0]):
            m, f = pairs[i]
            for child in range(children[i]):
                split = np.random.randint(self.pop[pairs[i][0]].shape[0])
                c = np.concatenate((self.pop[m][0:split], self.pop[f][split : self.pop[m].shape[0]]))
                repair_unit(c)
                self.pop[pairs[i, child]] = c

    def result(self):
        print('Best: ' + str(self.best_unit[0]+1) + '\t==>\t' + str(self.best_unit[1]))


def single_run():
    x = np.loadtxt('data/flow_20.txt').astype(int)
    y = np.loadtxt('data/distance_20.txt').astype(int)
    start = time.time()
    test = Qap(flow=x, distance=y, pop_size=100, gen=100, p_x=0.7, p_m=0.01, Tour=2)
    buff = []
    for i in range(20):
        test.main()
        test.result()
        buff.append(test.best_unit[1])
    stop = time.time()
    print('AVG:  ' + str(np.average(buff)))
    duration = int(round((stop - start)*1000))
    print('Duration: ' + str(duration) + ' ms')

def params_selection():       #LAST BEST TOUR 2 AND 5
                              #LAST BEST P_X  ==> 0.4
    x = np.loadtxt('data/flow_12.txt').astype(int)
    y = np.loadtxt('data/distance_12.txt').astype(int)
    start = time.time()
    buff1 = []
    for j in (np.arange(1, 25, 2) / 100):
        print('\n\n##########\t\tParam: ' + str(j) + '\t\t##############')
        test = Qap(flow=x, distance=y, pop_size=100, gen=100, p_x=0.4, p_m=j, Tour=2)
        buff2 = []
        for i in range(50):
            test.main()
            test.result()
            buff2.append(test.best_unit[1])
        buff1.append((np.average(buff2), j))
    stop = time.time()
    for a in buff1:
        print(a)
    duration = int(round((stop - start)*1000))
    print('Duration: ' + str(duration) + ' ms')

def multi_run():
    data = []
    data.append((np.loadtxt('data/flow_12.txt').astype(int), np.loadtxt('data/distance_12.txt').astype(int)))
    data.append((np.loadtxt('data/flow_14.txt').astype(int), np.loadtxt('data/distance_14.txt').astype(int)))
    data.append((np.loadtxt('data/flow_16.txt').astype(int), np.loadtxt('data/distance_16.txt').astype(int)))
    data.append((np.loadtxt('data/flow_18.txt').astype(int), np.loadtxt('data/distance_18.txt').astype(int)))
    data.append((np.loadtxt('data/flow_20.txt').astype(int), np.loadtxt('data/distance_20.txt').astype(int)))
    for d in data:
        print('\n\n##################### \t N: \t' + str(d[0].shape[0]) + '\t\t########################')
        start = time.time()
        test = Qap(flow=d[0], distance=d[1], pop_size=100, gen=100, p_x=0.7, p_m=0.01, Tour=5)
        buff = 0
        for i in range(10):
            test.main()
            test.result()
            buff += test.best_unit[1]
        print('AVG:  ' + str(buff/10))
        stop = time.time()
        duration = int(round((stop - start) * 1000))
        print('Duration: ' + str(duration) + ' ms')

single_run()