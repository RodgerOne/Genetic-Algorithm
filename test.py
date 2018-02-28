import numpy as np
pattern = np.arange(6)
he = np.array([5,4,3,2,1,0])
a = np.array([1,2,2,4,6,6]) - 1
r = np.array([6,5,4,4,2,1]) - 1
x, w, y, z = np.unique(a, return_index=True, return_inverse=True, return_counts=True)

def repairUnitq(unit):
    pattern = np.arange(unit.shape[0])
    result = np.array(unit)
    x, w = np.unique(unit, return_index=True)
    indices = np.setdiff1d(pattern, w)
    c = np.setdiff1d(pattern, unit)
    np.random.shuffle(c)
    result[indices] = c
    return result

def repairUnit(unit):
    pattern = np.arange(unit.shape[0])
    indices = np.setdiff1d(pattern, np.unique(unit, return_index=True)[1])
    c = np.setdiff1d(pattern, unit)
    np.random.shuffle(c)
    unit[indices] = c
    return


def evaluate(permutation, distance, flow):
    return (distance * ((flow[permutation, :])[:, permutation])).sum()

def calculateCost(permutation, distance, flow):
    cost = 0
    for i in range(0, distance.shape[0]):
        for j in range(0, distance.shape[0]):
            cost += flow[i][j] * distance[int(permutation[i])][int(permutation[j])]
    return cost


x = np.loadtxt('data/flow_12.txt').astype(int)
y = np.loadtxt('data/distance_12.txt').astype(int)
z = np.array([3,10,11,2,12,5,6,7,8,1,4,9])

print(calculateCost(z, y, x))