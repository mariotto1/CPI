import sys
import os
import utils
import classifiers as clf
import datetime as dt
import numpy
import preprocessing
import conf
from sklearn.model_selection import StratifiedKFold
import cProfile


def run():
    time = dt.datetime.now()
    print "Fold {} start  {:%H:%M:%S %d-%m-%Y}".format(n_fold, time)
    score = clf.classification(train, test, train_lengths, test_lengths, n_fold)
    print "Fold {} end  {:%H:%M:%S %d-%m-%Y}".format(n_fold, dt.datetime.now())
    return score


(conf.classifier, conf.component, conf.classification) = sys.argv[1:4]

if len(sys.argv) < 4:
    print "Usage: main.py classifier component target [prediction window size]"
elif conf.classification == 'prediction':
    if len(sys.argv) < 5:
        print "Prediction window size needed for predicting"
        exit(0)
    else:
        conf.window_size = int(sys.argv[4])
conf.results_path = '/'.join(["risultati", conf.classifier, conf.classification, conf.component])

orders, data, test_data = [], [], []  # dati = lista di triple (nome, danno si/no, sim)
for folder in os.listdir(conf.path_data):
    component_folder = conf.path_data + folder
    if conf.component == 'tutti cuscinetti' or folder == conf.component:
        if os.path.isdir(component_folder + "/Test/"):
            test_data.extend(utils.import_mats(component_folder + "/Test/", orders))
            conf.separated_test = True
        data.extend(utils.import_mats(component_folder + '/No_Danno/', orders))
    data.extend(utils.import_mats(component_folder + '/Danno/', orders))

# eliminazione ordini non necessari e modifica label per multidanno e prediction
sims, sims_labels = preprocessing.clean(data, min(orders))
if conf.separated_test:
    test_sims, _ = preprocessing.clean(test_data, min(orders))
del orders, data, test_data

n_fold = 1
scores = []
if conf.separated_test:
    train, train_lengths = utils.stack_sims(sims)
    test, test_lengths = utils.stack_sims(test_sims)
    train[:, :-1], test[:, :-1] = preprocessing.normalization(train[:, :-1], test[:, :-1])
    scores.append(run())
else:
    skf = StratifiedKFold(n_splits=5)  # numero di fold
    for train_indexes, test_indexes in skf.split(sims, sims_labels):
        train, test, train_lengths, test_lengths = [], [], [], []
        for i in train_indexes:
            train.extend(spectrum for spectrum in sims[i])
            train_lengths.append(len(sims[i]))
        for i in test_indexes:
            test.extend(spectrum for spectrum in sims[i])
            test_lengths.append(len(sims[i]))
        train = numpy.array(train)
        test = numpy.array(test)
        preprocessing.normalization(train, test)
        scores.append(run())
        n_fold += 1
utils.save_global_score(scores)
