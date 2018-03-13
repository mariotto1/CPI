import sys
import os
import utils
import classifiers as clf
import datetime as dt
import numpy
import preprocessing
import configuration as conf
from sklearn.model_selection import StratifiedKFold


def read_input():
    (conf.classifier, conf.component, conf.classification) = sys.argv[1:4]
    if len(sys.argv) < 4:
        print "Usage: main.py classifier component target [prediction window size]"
    elif conf.classification == 'prediction':
        if len(sys.argv) < 5:
            print "Prediction window size needed if predicting"
            exit(0)
        else:
            conf.predict_window_size = int(sys.argv[4]) * 60
            conf.prediction = True
    conf.set_results_path()


def load_dataset():
    for folder in os.listdir(conf.data_path):
        component_folder = conf.data_path + folder
        if conf.component == 'tutti' or folder == conf.component:
            if os.path.isdir(component_folder + "/Test/"):
                test_data.extend(utils.import_mats(component_folder + "/Test/", orders))
                conf.separated_test = True
            data.extend(utils.import_mats(component_folder + '/No_Danno/', orders))
        data.extend(utils.import_mats(component_folder + '/Danno/', orders))


def run():
    time = dt.datetime.now()
    print "Fold {} start  {:%H:%M:%S %d-%m-%Y}".format(n_fold, time)
    score = clf.classification(train, test, train_lengths, test_lengths, n_fold)
    print "Fold {} end  {:%H:%M:%S %d-%m-%Y}".format(n_fold, dt.datetime.now())
    return score


read_input()
orders, data, test_data = [], [], []  # data = list of tuples (sim_name, has_damage, sim)
load_dataset()

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
    skf = StratifiedKFold(n_splits=5)  # number of folds
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
