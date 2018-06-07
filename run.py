import numpy as np
import Qap
import RandomSearch
import time
import matplotlib.pyplot as plt


def single_run(tour=None):
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
    pass


def single_run_charts(number, tour=None):
    x = np.loadtxt('data/flow_'+str(number)+'.txt').astype(int)
    y = np.loadtxt('data/distance_'+str(number)+'.txt').astype(int)
    start = time.time()
    test = Qap(flow=x, distance=y, pop_size=100, gen=100, p_x=0.7, p_m=0.01, tour=tour)
    buff = []
    graph = np.zeros(test.stat.shape)
    for i in range(50):
        test.main()
        graph += test.stat
        test.result()
        buff.append(test.best_unit[1])
    graph /= 50
    stop = time.time()
    print('AVG of Bests:  ' + str(np.average(buff)))
    duration = int(round((stop - start)*1000))
    print('Duration: ' + str(duration) + ' ms')
    print('AVG Duration: ' + str(duration /50) + ' ms')
    x_axis = np.arange(graph.shape[0])
    min = graph[:, 0]
    avg = graph[:, 1]
    max = graph[:, 2]

    fig, ax = plt.subplots()
    ax.plot(x_axis, min, 'g', label='Min')
    ax.plot(x_axis, avg, 'y', label='Avg')
    ax.plot(x_axis, max, 'r', label='Max')

    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('#EEEEEE')
    plt.show()
    pass


def params_selection():
    x = np.loadtxt('data/flow_12.txt').astype(int)
    y = np.loadtxt('data/distance_12.txt').astype(int)
    start = time.time()
    buff1 = []
    x_axis = np.arange(50, 400, 50)
    for j in x_axis:                # zalezy od parametru
        print('\n\n##########\t\tParam: ' + str(j) + '\t\t##############')
        test = Qap(flow=x, distance=y, pop_size=100, gen=j, p_x=0.7, p_m=0.01, tour=5)
        buff2 = []
        for i in range(75):
            test.main()
            buff2.append(test.best_unit[1])
        buff1.append(np.average(buff2))
    stop = time.time()
    duration = int(round((stop - start)*1000))
    print('Duration: ' + str(duration) + ' ms')

    y_axis = buff1

    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis, 'g', label='Avg')

    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('#EEEEEE')
    plt.show()
    pass


def multi_run(tour=None):
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
    pass

f, d = (np.loadtxt('data/flow_12.txt').astype(int), np.loadtxt('data/distance_12.txt').astype(int))
start = time.time()
test = RandomSearch(f, d, 1000)
test.run_random()
stop = time.time()
duration = int(round((stop - start) * 1000))
print('Duration: ' + str(duration) + ' ms')
