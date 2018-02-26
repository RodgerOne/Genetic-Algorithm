import numpy as np
a = np.array([1,2,3,4,5,6,7,8,9,0])
#c = np.array([[1,2,3,4,5],[6,7,8,9,0],[1,2,3,4,5],[6,7,8,9,0]])
b = np.random.choice(a, 5)
c = (a*np.ones((a.shape[0], 1)))


f = np.random.randint(1,3,5)
print(f)
