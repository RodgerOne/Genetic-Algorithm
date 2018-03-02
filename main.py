import numpy as np
import time
import matplotlib.pyplot as plt
'''
    distance - macierz sąsiedztwa pomiędzy lokalizacjami
    flow - macierz przepływu
    p_m - prawdopodobieństwo mutacji
    p_x - prawdopodobieństwo krzyżowania
    p_s - prawdopodobieństwo selekcji
    pop_size - rozmiar populacji
    gen - liczba pokoleń
    tour - wielkosć turnieju - jesli jest None to ruletka
    '''

class Qap:
    def __init__(self, distance, flow, p_m, p_x, pop_size, gen, tour = None):
        self.distance = distance
        self.flow = flow
        self.p_m = p_m
        self.p_x = p_x
        self.tour = tour
        self.pop_size = pop_size
        self.gen = gen
        self.pop = None
        self.pop_value = None
        self.best_unit = None
        self.stat = np.zeros((pop_size, 3))
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
        return t >= self.gen


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
            self.selection_roulette() if self.tour is None else self.selection_tournament()
            self.crossover()
            self.mutation()
            self.evaluate()
            if np.min(self.pop_value) < self.best_unit[1]:
                i = np.argmin(self.pop_value)
                self.best_unit = self.pop[i], self.pop_value[i]
            self.stat[t] = np.array([np.min(self.pop_value), np.average(self.pop_value), np.max(self.pop_value)])
            t += 1

    def selection_tournament(self):
        acc = np.zeros(self.pop.shape, dtype=int)
        indices = np.arange(self.pop.shape[0])
        for i in indices:
            units = np.random.choice(indices, self.tour, False)
            acc[i] = self.pop[units[np.argmin(self.pop_value[units])]]
        self.pop = acc

    def selection_roulette(self):
        probs = (np.full(self.pop_value.shape[0], np.max(self.pop_value)) - self.pop_value + 1)
        probs /= probs.sum()
        for i in range(probs.shape[0]-1):
            probs[i+1] += probs[i]
        acc = np.zeros(self.pop.shape, dtype=int)
        for p in range(self.pop.shape[0]):
            acc[p] = self.pop[np.where(np.random.rand() < probs)[0][0]]
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


def single_run(tour = None):
    x = np.loadtxt('data/flow_20.txt').astype(int)
    y = np.loadtxt('data/distance_20.txt').astype(int)
    start = time.time()
    test = Qap(flow=x, distance=y, pop_size=100, gen=100, p_x=0.7, p_m=0.01, tour=tour)
    buff = []
    for i in range(1):
        test.main()
        test.result()
        buff.append(test.best_unit[1])
    stop = time.time()
    print('AVG:  ' + str(np.average(buff)))
    duration = int(round((stop - start)*1000))
    print('Duration: ' + str(duration) + ' ms')


def single_run_charts(number, tour=None):
    x = np.loadtxt('data/flow_'+str(number)+'.txt').astype(int)
    y = np.loadtxt('data/distance_'+str(number)+'.txt').astype(int)
    start = time.time()
    test = Qap(flow=x, distance=y, pop_size=100, gen=100, p_x=0.7, p_m=0.01, tour=tour)
    buff = []
    graph = np.zeros(test.stat.shape)
    for i in range(10):
        test.main()
        graph += test.stat
        test.result()
        buff.append(test.best_unit[1])
    graph /= 10
    stop = time.time()
    print('AVG of Bests:  ' + str(np.average(buff)))
    duration = int(round((stop - start)*1000))
    print('Duration: ' + str(duration) + ' ms')

    x_axis = np.arange(graph.shape[0])
    min = graph[:, 0]
    avg = graph[:, 1]
    max = graph[:, 2]

    fig, ax = plt.subplots()
    ax.plot(x_axis, min, 'k:', label='Min')
    ax.plot(x_axis, avg, 'k', label='Avg')
    ax.plot(x_axis, max, 'k--', label='Max')

    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('#EEEEEE')
    plt.show()


def params_selection():
    x = np.loadtxt('data/flow_12.txt').astype(int)
    y = np.loadtxt('data/distance_12.txt').astype(int)
    start = time.time()
    buff1 = []
    x_axis = np.arange(1, 20, 1)
    for j in x_axis:                # zalezy od parametru
        print('\n\n##########\t\tParam: ' + str(j) + '\t\t##############')
        test = Qap(flow=x, distance=y, pop_size=100, gen=100, p_x=0.7, p_m=0.01, tour=j)
        buff2 = []
        for i in range(50):
            test.main()
            #test.result()
            buff2.append(test.best_unit[1])
        buff1.append(np.average(buff2))
    stop = time.time()
    duration = int(round((stop - start)*1000))
    print('Duration: ' + str(duration) + ' ms')

    y_axis = buff1

    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis, 'k', label='Avg')

    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('#EEEEEE')
    plt.show()

def multi_run(tour = None):
    data = []
    data.append((np.loadtxt('data/flow_12.txt').astype(int), np.loadtxt('data/distance_12.txt').astype(int)))
    data.append((np.loadtxt('data/flow_14.txt').astype(int), np.loadtxt('data/distance_14.txt').astype(int)))
    data.append((np.loadtxt('data/flow_16.txt').astype(int), np.loadtxt('data/distance_16.txt').astype(int)))
    data.append((np.loadtxt('data/flow_18.txt').astype(int), np.loadtxt('data/distance_18.txt').astype(int)))
    data.append((np.loadtxt('data/flow_20.txt').astype(int), np.loadtxt('data/distance_20.txt').astype(int)))
    for d in data:
        print('\n\n##################### \t N: \t' + str(d[0].shape[0]) + '\t\t########################')
        start = time.time()
        test = Qap(flow=d[0], distance=d[1], pop_size=100, gen=100, p_x=0.7, p_m=0.01, tour=tour)
        buff = 0
        for i in range(10):
            test.main()
            test.result()
            buff += test.best_unit[1]
        print('AVG:  ' + str(buff/10))
        stop = time.time()
        duration = int(round((stop - start) * 1000))
        print('Duration: ' + str(duration) + ' ms')

params_selection()