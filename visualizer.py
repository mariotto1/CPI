import sys
from utils import *
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


def plot_ordine(ordine_origine, ordine_target, label, i, sfasamento):
    '''
    origine, = plt.plot(ordine_origine, 'b', label='origine')
    target, = plt.plot(ordine_target, 'r', label='target')
    plt.plot(label, 'g')
    plt.title("Ordine " + str(i))
    plt.legend(handles=[origine, target])
    plt.show()
    '''
    plt.figure(1)
    plt.subplot(211)
    plt.plot(ordine_origine, 'b', label='origine')
    plt.plot(label, 'g')
    plt.title("Ordine: {}, sfasamento: {}".format(i, sfasamento))

    plt.subplot(212)
    plt.plot(ordine_target, 'r', label='target')
    plt.plot(label, 'g')

    plt.show()


sfasa = True
folder_origine = sys.argv[1] + "individuazione componente/Condizioni miste/Danno/"
folder_target = sys.argv[2] + "individuazione componente/Condizioni miste/Danno/"

dict_origine = read_data(sys.argv[1])
dict_target = read_data(sys.argv[2])
if len(sys.argv) > 3:
    sfasa = False

path = "./"
path_data_origine = path + folder_origine
path_data_target = path + folder_target

dati_origine = []  # lista di triple (nome,danno,sim)
dati_target = []
dati_origine.extend(import_mat(path_data_origine))
dati_target.extend(import_mat(path_data_target))
sorted_dati_origine = sorted(dati_origine, key=lambda tup: tup[0])
sorted_dati_target = sorted(dati_target, key=lambda tup: tup[0])
for tripla_origine, tripla_target in zip(sorted_dati_origine[0:30], sorted_dati_target[0:30]):
    print tripla_origine[0], tripla_target[0]
    if tripla_origine[0].split('_')[:-1] != tripla_target[0].split('_')[:-1]: raise Exception('I nomi delle sim non coincidono')

    ordini = []
    i = 1
    ordine_origine = calculate_order(tripla_origine[0].split('_')[4], **dict_origine)
    while ordine_origine * i <= 7.5:
        ordini.append(ordine_a_colonna(round_to_05(ordine_origine * i)))
        i += 1
    print calculate_order(tripla_target[0].split('_')[4], **dict_target) - ordine_origine
    modifica = int(round_to_05(calculate_order(tripla_target[0].split('_')[4], **dict_target) - ordine_origine) * 2)
    print modifica
    ordine_origine = ordine_a_colonna(round_to_05(ordine_origine))
    # print ordini, ordine_origine, modifica

    label = tripla_origine[2][:, -1]
    sim_origine = tripla_origine[2][:, 1:16]
    sim_target = tripla_target[2][:, 1:16]
    sfasamento = 0
    for col in range(15):
        if sfasa and len(ordini) > 0 and col + 1 == ordini[0] - 1:
            sfasamento += modifica
            del ordini[0]
        if sfasamento > 0:
            plot_ordine(sim_origine[:, col], sim_target[:, col + sfasamento], label, (col + 1) * 0.5 + 0.5, sfasamento)
        else:
            plot_ordine(sim_origine[:, col + abs(sfasamento)], sim_target[:, col], label, (col + 1) * 0.5 + 0.5, sfasamento)
