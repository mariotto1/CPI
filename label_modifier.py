import sys
import os
import utils
import matplotlib.pyplot as plt
import numpy as np
import shutil
import scipy.io as sio


def plot_order(order, label, label2, i):
    z = np.polyfit([x for x in range(len(order))], order, 20)
    p = np.poly1d(z)
    pol = p([x for x in range(len(order))])
    plt.plot(order, 'r', pol, 'b', label, 'g', label2, 'y', )
    plt.title("Ordine " + str(i))
    plt.show()


def interp(sim):
    pols = []
    vars = []
    for i in range(1, 60, 1):
        z = np.polyfit([x for x in range(len(sim[:, i]))], sim[:, i], 20)
        p = np.poly1d(z)
        pol = p([x for x in range(len(sim[:, i]))])
        pols.append(pol)
        vars.append(np.var(pol))
    print vars.index(max(vars))
    return pols[vars.index(max(vars))]


def new_max_label(order):
    media = np.mean(order[-200:])
    massimo = max(order[-200:])
    minimo = min(order[-200:])
    return [0 if x < media - (massimo - minimo) * 2 else 1 for x in order]


folder = sys.argv[1]

path = "./"
path_data = path + folder

orders = []
data = []  # lista di coppie (sims,nomi), composta da danno + no_danno + eventuale test
data.append(utils.import_mats(path_data, orders))

shutil.rmtree(path_data)
os.makedirs(path_data)

for i in range(len(data[0][0])):
    data[0][0][i] = data[0][0][i][:, :-1]  # toglie colonna con indicazione tipo di danno
    ordine = interp(data[0][0][i])
    data[0][0][i][:, -1] = new_max_label(ordine)
    # for col in range(dati[0][0][i].shape[1]-1):
    # plot_order(dati[0][0][i][:,col],dati[0][0][i][:,-1],label,col)
    sio.savemat(path_data + data[0][1][i], {'export_down': data[0][0][i]})
