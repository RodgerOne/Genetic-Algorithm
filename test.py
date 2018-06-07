import numpy as np

######### OLD FUNCTIONS - NOT IN USE ANYMORE ##################


def mutation_old(self):
    for pos in range(self.pop.shape[0]):
        if (np.random.random() < self.p_m):
            a, b = np.random.randint(0, self.pop.shape[1], 2)
            self.pop[pos][[a, b]] = self.pop[pos][[b, a]]
    return None


def repair_unitq(unit):
    pattern = np.arange(unit.shape[0])
    result = np.array(unit)
    x, w = np.unique(unit, return_index=True)
    indices = np.setdiff1d(pattern, w)
    c = np.setdiff1d(pattern, unit)
    np.random.shuffle(c)
    result[indices] = c
    return result


def repair_unit(unit):
    pattern = np.arange(unit.shape[0])
    indices = np.setdiff1d(pattern, np.unique(unit, return_index=True)[1])
    c = np.setdiff1d(pattern, unit)
    np.random.shuffle(c)
    unit[indices] = c
    return


def evaluate(permutation, distance, flow):
    return (distance * ((flow[permutation, :])[:, permutation])).sum()


def calculate_cost(permutation, distance, flow):
    cost = 0
    for i in range(0, distance.shape[0]):
        for j in range(0, distance.shape[0]):
            cost += flow[i][j] * distance[int(permutation[i])][int(permutation[j])]
    return cost


def crossover(a,b):
    pattern = np.arange(a.shape[0])

    def repair_unit(unit):
        indices = np.setdiff1d(pattern, np.unique(unit, return_index=True)[1])
        unit[indices] = np.setdiff1d(pattern, unit)
        pass

    repair_unit(a)
    repair_unit(b)
    print('Parent_1: ' + str(a))
    print('Parent_2: ' + str(b))

    split = np.random.randint(a.shape[0])
    print('Split: ' + str(split))
    c = np.concatenate((a[0:split], b[split : a.shape[0]]))
    print(c)
    repair_unit(c)
    print(c)
    return
##########################################################
