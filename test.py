from random import seed
from random import randint
from collections import deque
import numpy as np


seed(1234)
magic = [1, 3, 5, 3, 4, 6, 1, 0, 1, 3]
distance = [4, 5, 6, 1, 2, 1, 1, 1, 2, 1]

optimal_start = 0
optimal_max = 0
for i in range(len(magic)):
    print("Checking " + str(i))
    magic.append(magic.pop(0))
    distance.append(distance.pop(0))
    tmp = np.subtract(magic, distance)
    func = (lambda: [(t - s) for s, t in zip(tmp, tmp[1:])])
    distances = func()
    print(distances)
    if any(n < 0 for n in distances):
        pass
    else:
        sum = np.sum(distances)
        if optimal_max > sum:
            optimal_max = sum
            optimal_start = i
print(optimal_start)
