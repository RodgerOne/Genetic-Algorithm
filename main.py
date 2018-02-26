import numpy as np
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
        self.initialise()

    def initialise(self):
        self.pop = np.ones((self.pop_size, 1), dtype=int) * np.array(range(self.flow.shape[1]))
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
                self.pop[pos][a, b] = self.pop[pos][b, a]
        return None

    def main(self):
        t = 0
        self.initialise()
        self.evaluate()

        while(not self.isDone(t)):
            self.selection()
            self.crossover()
            self.mutation()
            self.evaluate()
            t += 1

        return

    def selection(self):
        acc = np.zeros(self.pop.shape)
        indices = np.arange(self.pop.shape[0])
        for i in indices:
            units = np.random.choice(indices, self.Tour, False)
            acc[i] = self.pop[np.argmin(self.pop_value[units])]
        self.pop = acc
        return

    def crossover(self):
        parents_indices = np.where((np.random.rand(self.pop.shape[0], 1) > self.p_x))[0]
        #indexy rodziców - teraz ich sparuj (nieparzystego po prostu odrzuć) i zrób potomstwo : 1 lub 2 - jak jeden to zastąp losowego z rodziców (albo 1 albo 2), jak 2 dzieci to zastąp oboje rodziców
        #zrób sprawne dobieranie i naprawianie genów dzieci
        pairs = np.random.choice(parents_indices, (int(parents_indices.shape[0] / 2), 2), False) #indexy par - ostatniego nieparzystego odrzuca
        children = np.random.randint(1, 3, pairs.shape[0])
        for pair in pairs:


        return None




