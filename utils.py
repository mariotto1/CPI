import scipy.io as sio
import numpy
import os
import sys
import scipy.misc as smp
import math
import conf


def import_mats(path, orders=[]):
    sims = []
    for file_name in next(os.walk(path))[2]:
        sim = numpy.float32(sio.loadmat(path + file_name)["export_down"])
        damage = True if sum(sim[:, -1]) > 0 else False
        orders.append(int(file_name.split('_')[-1][0:2]))  # estrazione ordine max da nome file
        sims.append((file_name, damage, sim))
    return sims


def stack_sims(sims):
    return numpy.concatenate(sims, axis=0), [len(sim) for sim in sims]


def calculate_score(preds, labels):
    score = {'FP': 0, 'FN': 0, 'Corretti': 0, 'Danno sbagliato': 0}
    for damage in conf.damage_types:
        score['Errori ' + damage] = 0
    for pred, label in zip(preds, labels):
        if pred == 0 and label > 0:
            score['FN'] += 1
        elif pred > 0 and label == 0:
            score['FP'] += 1
        elif pred == label:
            score['Corretti'] += 1
        else:
            score['Danno sbagliato'] += 1
            score['Errori ' + conf.damage_types[label - 1]] += 1
    score['Negativi'] = labels.count(0)
    score['Positivi'] = len(labels) - score['Negativi']
    for damage in conf.damage_types:
        score['Danno su ' + damage] = labels.count(conf.damage_types.index(damage) + 1)
    score['Accuratezza'] = float(score['Corretti']) / (score['Negativi'] + score['Positivi']) * 100
    score['FPR'] = float(score['FP']) / score['Negativi'] * 100
    score['FNR'] = float(score['FN']) / score['Positivi'] * 100
    return score


def save_fold_results(fold, confidences, labels, score):
    print "Saving fold results..."

    prediction_time = ''
    if conf.window_size > 0:
        prediction_time += '_' + str(conf.window_size) + "min"

    with open('/'.join([conf.results_path, "confidenze", "confidenze"]) + prediction_time, 'a') as f:
        f.write("Fold" + str(fold) + "\n")
        f.write(';'.join(str(x) for x in confidences) + "\n")
        f.write(';'.join(str(x) for x in labels) + "\n")

    with open(conf.results_path + '/score_per_fold' + prediction_time, 'a') as f:
        f.write("Fold" + str(fold) + "\n")
        f.write(';'.join([i + "=" + str(score[i]) for i in sorted(score)]) + "\n")


def save_global_score(scores):
    file_name = conf.component + '_risultati_' + conf.classifier
    if conf.window_size > 0:
        file_name += '_' + str(conf.window_size) + "min"

    global_score = {'FP': 0, 'FN': 0, 'Corretti': 0, 'Danno sbagliato': 0, 'Positivi': 0, 'Negativi': 0}
    for score in scores:
        for key in global_score:
            global_score[key] += score[key]
    global_score['Accuratezza'] = float(global_score['Corretti']) / (global_score['Negativi'] + global_score['Positivi']) * 100
    global_score['FPR'] = float(global_score['FP']) / global_score['Negativi'] * 100
    global_score['FNR'] = float(global_score['FN']) / global_score['Positivi'] * 100
    global_score['Rateo Danno sbagliato'] = float(global_score['Danno sbagliato']) / global_score['Positivi'] * 100

    with open(conf.results_path + file_name + ".txt", 'a') as f:
        f.write("Accuratezza: {:.1f}% ({} predizioni corrette su {})\n".format(global_score['Accuratezza'], global_score['Corretti'],
                                                                               global_score['Negativi'] + global_score['Positivi']))
        f.write("FPR: {:.1f}% ({} falsi positivi su {} negativi)\n".format(global_score['FPR'], global_score['FP'], global_score['Negativi']))
        f.write("FNR: {:.1f}% ({} falsi negativi su {} positivi)\n".format(global_score['FNR'], global_score['FN'], global_score['Positivi']))
        f.write("Tipologia di danno sbagliata: {:.1f}% ({} errori su {} positivi)\n".format(global_score['Rateo Danno sbagliato'],
                                                                                            global_score['Danno sbagliato'], global_score['Positivi']))


def samples_per_epoch(window, lengths):
    tot = 0
    for l in lengths:
        tot += l - window + 1
    return tot


'''
Funzioni per conversione/visualizzazione
'''


def calculate_order(damage, Nb, Db, Dp, S=1, phi=0.0):
    Nb, Db, Dp = float(Nb), float(Db), float(Dp)
    if damage == "Outer":
        return Nb / 2 * S * (1 - Db / Dp * math.cos(phi))
    elif damage == "Inner":
        return Nb / 2 * S * (1 + Db / Dp * math.cos(phi))
    elif damage == "Ball":
        return Dp / (2 * Db) * S * (1 - (Db / Dp * math.cos(phi)) ** 2)


def read_data(path):
    dict = {}
    with open(path + "dati.txt", 'r') as f:
        for l in f.read().splitlines():
            dict[l.split('=')[0]] = float(l.split('=')[1])
    return dict


def round_to_05(n):
    return round(n * 2.0) / 2.0


def floor_to_05(n):
    return math.floor(n * 2.0) / 2.0


def column_index_to_order(n):
    b = n / 10 + 1
    b += (n % 10) / 10.0
    return b


def column_index_to_order05(n):
    return 1 + n * 0.5


def order_to_column_index(n):
    n = round(n, 1)
    return int(round(n % 1 * 10) + (math.floor(n) - 1) * 10) + 1


def plot(sims, lenghts, pred):
    width = 5
    offset = 0
    for l in lenghts:
        sim = sims[offset:offset + l, :]
        label = pred[offset:offset + l]
        data = numpy.empty((sim.shape[0], (sim.shape[1] + 1) * width, 3), dtype=numpy.uint8)
        for x in range(sim.shape[0]):
            for y in range(sim.shape[1]):
                for i in range(width):
                    data[x][y * width + i] = [sim[x][y] * 255, sim[x][y] * 255, (1 - sim[x][y]) * 255]
            for i in range(width):
                data[x][-1 - i] = [label[x] * 255, (1 - label[x]) * 255, 0]
        img = smp.toimage(data.T)  # Create a PIL image
        img.resize((900, 600)).show()
        offset += l


def plot2(sim, ordine):
    width = 5
    label = sim[:, -1]
    indexes_to_delete = [0] + [n for n in range(ordine * 2, sim.shape[1], 1)]
    sim = numpy.delete(sim, indexes_to_delete, axis=1)  # rimuove colonne inutili
    data = numpy.empty((sim.shape[0], (sim.shape[1] + 1) * width, 3), dtype=numpy.uint8)
    for x in range(sim.shape[0]):
        for y in range(sim.shape[1]):
            for i in range(width):
                data[x][y * width + i] = [sim[x][y] * 255, sim[x][y] * 255, (1 - sim[x][y]) * 255]
        for i in range(width):
            data[x][-1 - i] = [label[x] * 255, (1 - label[x]) * 255, 0]
    return smp.toimage(data.T)  # Create a PIL image
    # img.resize((900, 600)).show()


if __name__ == "__main__":
    print "Outer: {}".format(calculate_order("Outer", *sys.argv[1:]))
    print "Inner: {}".format(calculate_order("Inner", *sys.argv[1:]))
    print "Ball: {}".format(calculate_order("Ball", *sys.argv[1:]))
