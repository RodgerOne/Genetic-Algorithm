import numpy as np
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
    def __init__(self, distance, flow, p_m, p_x, pop_size, gen, tour=None):
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
        self.stat = np.zeros((gen, 3))
        self.initialise()

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

    def is_done(self, t):
        return t >= self.gen

    def mutation(self):
        mutants = np.nonzero(np.random.rand(self.pop.shape[0], self.pop.shape[1]) < self.p_m)
        for ind in range(mutants[0].shape[0]):
            r = np.random.randint(0, self.pop.shape[1])
            self.pop[mutants[0][ind]][[mutants[1][ind], r]] = self.pop[mutants[0][ind]][[r, mutants[1][ind]]]
        pass

    def main(self):
        t = 0
        self.initialise()
        self.evaluate()
        self.best_unit = self.pop[0], self.pop_value[0]
        while not self.is_done(t):
            self.selection_roulette() if self.tour is None else self.selection_tournament()
            self.crossover()
            self.mutation()
            self.evaluate()
            if np.min(self.pop_value) < self.best_unit[1]:
                i = np.argmin(self.pop_value)
                self.best_unit = self.pop[i], self.pop_value[i]
            self.stat[t] = np.array([np.min(self.pop_value), np.average(self.pop_value), np.max(self.pop_value)])
            t += 1
        pass

    def selection_tournament(self):
        acc = np.zeros(self.pop.shape, dtype=int)
        indices = np.arange(self.pop.shape[0])
        for i in indices:
            units = np.random.choice(indices, self.tour, False)
            acc[i] = self.pop[units[np.argmin(self.pop_value[units])]]
        self.pop = acc
        pass

    def selection_roulette(self):
        probs = (np.full(self.pop_value.shape[0], np.max(self.pop_value)) - self.pop_value + 1)
        probs /= probs.sum()
        for i in range(probs.shape[0]-1):
            probs[i+1] += probs[i]
        acc = np.zeros(self.pop.shape, dtype=int)
        for p in range(self.pop.shape[0]):
            acc[p] = self.pop[np.where(np.random.rand() < probs)[0][0]]
        self.pop = acc
        pass

    def crossover(self):
        pattern = np.arange(self.pop.shape[1])

        def repair_unit(unit):
            indices = np.setdiff1d(pattern, np.unique(unit, return_index=True)[1])
            new = np.setdiff1d(pattern, unit)
            np.random.shuffle(new)
            unit[indices] = new
            pass

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
        pass

    def result(self):
        print('Best: ' + str(self.best_unit[0]+1) + '\t==>\t' + str(self.best_unit[1]))
        pass
