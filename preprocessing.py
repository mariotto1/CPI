import numpy
from sklearn import preprocessing
import keras
from configuration import *

# Remove of unnecessary orders and modify labels for multidamage and prediction
def clean(data, min_order):
    sims, labels = [], []
    for sim_name, has_damage, sim in data:
        nan_indexes = [0] + numpy.where(numpy.isnan(sim[0, :]))[0].tolist()
        delete_indexes = []
        for i, active_sensor in enumerate(active_sensors):
            if active_sensor:
                delete_indexes += [nan_indexes[i]] + [n for n in range(nan_indexes[i] + min_order * 2, nan_indexes[i + 1])]
            else:
                delete_indexes += [n for n in range(nan_indexes[i], nan_indexes[i + 1])]
        delete_indexes += [nan_indexes[-1], sim.shape[1] - 2]
        sim = numpy.delete(sim, delete_indexes, axis=1)  # remove columns of inactive sensors
        if has_damage:
            if prediction:
                if 0 < sim[:, -1].tolist().count(transitory_label) < predict_window_size:
                    transitory_length = sim[:, -1].tolist().count(transitory_label)
                else:
                    transitory_length = predict_window_size
                sim = sim[sim[:, -1] != damage_label]  # remove row labelled as damage
                sim[:, -1] = [0] * (sim.shape[0] - transitory_length) + [1] * transitory_length
            elif not prediction and transitory_label in sim[:, -1]:
                sim[:, -1] = [0] * (sim.shape[0] - sim[:, -1].tolist().count(damage_label)) + [1] * sim[:, -1].tolist().count(damage_label)
            for i, damage in enumerate(damage_types):
                if damage in sim_name:
                    sim[:, -1] *= i + 1
                    labels.append(i + 1)
                    break
        else:
            labels.append(0)
        sims.append(sim)
    return sims, labels


def normalization(train, test):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(train), scaler.transform(test)


def batch_generator(examples, lengths):
    num_classes = len(damage_types) + 1
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
                num_classes=num_classes))
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
