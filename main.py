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
    def __init__(self, distance, flow, p_m, p_x, p_s, pop_size, gen):
        self.distance = distance
        self.flow = flow
        self.p_m = p_m
        self.p_x = p_x
        self.p_s = p_s
        self.pop_size = pop_size
        self.gen = gen
        self.pop = []
        self.initialise()


    def initialise(self):
        for i in range(self.pop_size):
            tmp = np.array(range(self.flow.shape[1]))
            np.random.shuffle(tmp)
            self.pop.append(tmp)
        return


    def main(self):
        t = 0
        pop = np.list()
        initialise(pop[t])
        evaluate(pop[t])

        while (True):
            pop[t + 1] = selection(pop[t])
            pop[t + 1] = crossover(pop[t + 1])
            pop[t + 1] = mutation(pop[t + 1])
            evaluate(pop[t + 1])
            t += 1

        return


    def evaluate(self, unit):
        return None

    def selection(self, unit):
        return None

    def crossover(self, unit):
        return None

    def mutation(self, unit):
        return None


