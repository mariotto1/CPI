import numpy
import configuration as conf
from sklearn import preprocessing
import keras

# Remove of unnecessary orders and modify labels for multidamage and prediction
def clean(data, min_order):
    sims, labels = [], []
    for name, damage, sim in data:  # tupla = tripla (nome,danno,sim)
        delete_indexes = [0] + [n for n in range(min_order * 2, sim.shape[1] - 1)]
        sim = numpy.delete(sim, delete_indexes, axis=1)  # rimuove colonne inutili
        if damage:
            if conf.window_size > 0:
                len_transitory = numpy.count_nonzero(sim[:, -1] == 2) if 0 < numpy.count_nonzero(sim[:, -1] == 2) < conf.window_size * 60 else conf.window_size * 60
                sim = sim[sim[:, -1] != 1]  # rimuove spettri con danno
                sim[:, -1] = [0] * (sim.shape[0] - len_transitory) + [1] * len_transitory
            elif 2 in sim[:, -1]:
                sim[:, -1] = [0] * (sim.shape[0] - numpy.count_nonzero(sim[:, -1] == 1)) + [1] * numpy.count_nonzero(sim[:, -1] == 1)
            for damage in conf.damage_types:
                if damage in name:
                    sim[:, -1] *= conf.damage_types.index(damage) + 1
                    labels.append(conf.damage_types.index(damage) + 1)
                    break
        else:
            labels.append(0)
        sims.append(sim)
    return sims, labels


def normalization(train, test):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(train), scaler.transform(test)


def batch_generator(examples, lengths, look_back, batch_size):
    num_classes = len(set(examples[:, -1]))
    sim_offset = 0
    global_offset = 0
    lengths_index = 0
    while True:
        batch_labels = []
        batch_data = []
        for i in range(batch_size):
            offset = global_offset + sim_offset
            batch_data.append(examples[offset:offset + look_back, :-1])
            batch_labels.append(keras.utils.np_utils.to_categorical(
                max(examples[offset:offset + look_back, -1]),
                num_classes=num_classes
            ))
            sim_offset += 1
            if sim_offset + look_back >= lengths[lengths_index]:
                global_offset += lengths[lengths_index]
                lengths_index += 1
                sim_offset = 0
                if lengths_index >= len(lengths):
                    lengths_index = 0
                    global_offset = 0
        yield numpy.array(batch_data), numpy.array(batch_labels)


def reduce_to_sim_05(sim, order):
    indexes_to_delete = [n + 1 for n in range(0, order * 10 - 1) if n % 5 != 0] + [n for n in range(order * 10, sim.shape[1] - 2)]
    sim = numpy.delete(sim, indexes_to_delete, axis=1)
    dim = sim.shape[1] - 3
    sim = numpy.insert(sim, sim.shape[1] - 2, [numpy.nan] * sim.shape[0], axis=1)
    sim = numpy.insert(sim, sim.shape[1] - 2, [[0] * sim.shape[0] for n in range(dim)], axis=1)
    sim = numpy.insert(sim, sim.shape[1] - 2, [numpy.nan] * sim.shape[0], axis=1)
    sim = numpy.insert(sim, sim.shape[1] - 2, [[0] * sim.shape[0] for n in range(dim)], axis=1)
    sim = numpy.insert(sim, sim.shape[1] - 2, [numpy.nan] * sim.shape[0], axis=1)
    return sim
