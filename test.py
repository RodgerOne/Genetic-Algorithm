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

print(np.min([2,0]))