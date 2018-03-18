from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import keras as kr
import numpy
import utils
import preprocessing
import configuration as conf

nn_layers = {
    'lstm': kr.layers.LSTM,
    'gru': kr.layers.GRU,
    'dense': kr.layers.Dense
}
optimizers = {
    'sgd': kr.optimizers.SGD,
    'adam': kr.optimizers.Adam
}

def mlp(train, test):
    model = kr.models.Sequential()
    # model.add(kr.layers.Conv1D(2,5,strides=1, padding='valid',input_shape=(train.shape[1]-1)))
    model.add(kr.layers.InputLayer(input_shape=(train.shape[1] - 1,)))
    for layer in conf.mlp_layers:
        model.add(nn_layers[layer['type']](**layer['params']))
    model.add(kr.layers.Dense(units=len(conf.damage_types) + 1, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizers[conf.optimizer['type']](**conf.optimizer['params']))

    model.fit(train[:, :-1], kr.utils.np_utils.to_categorical(train[:, -1]), epochs=conf.epochs,
              batch_size=conf.batch_size, shuffle=True)

    print "Evaluating testing set..."
    confidences = model.predict(test[:, :-1])
    predictions = [numpy.argmax(x) for x in confidences]
    return confidences, predictions, test[:, -1].astype(int).tolist()


def lstm(train, train_lengths, test, test_lengths):
    model = kr.models.Sequential()
    model.add(kr.layers.InputLayer(input_shape=(conf.look_back, train.shape[1] - 1)))
    for layer in conf.rnn_layers:
        if 'bidir' in layer and layer['bidir'] == True:
            model.add(kr.layers.Bidirectional(nn_layers[layer['type']](**layer['params']), **layer['bidir_param']))
        else:
            model.add(nn_layers[layer['type']](**layer['params']))
    model.add(kr.layers.Dense(units=len(conf.damage_types) + 1, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizers[conf.optimizer['type']](**conf.optimizer['params']))

    model.fit_generator(preprocessing.batch_generator(train, train_lengths), epochs=conf.epochs,
                        steps_per_epoch=utils.samples_per_epoch(train_lengths) / conf.batch_size)

    generator = preprocessing.batch_generator(test, test_lengths)
    test_labels = []
    confidences = []
    print "Evaluating testing set..."
    for x in range(utils.samples_per_epoch(test_lengths) / conf.batch_size):
        batch = next(generator)
        confidences.extend(model.predict_on_batch(batch[0]))
        test_labels.extend(numpy.argmax(x) for x in batch[1])
    predictions = [numpy.argmax(x) for x in confidences]
    return confidences, predictions, test_labels


def SVM(train, test, cost=1.0, ker='rbf', verb=False):
    clf = svm.SVC(C=cost, kernel=ker, decision_function_shape='ovr', verbose=verb)
    clf.fit(train[:, :-1], train[:, -1])
    conf = numpy.array(clf.decision_function(test[:, :-1]))
    pred = numpy.array(clf.predict(test[:, :-1]))
    return conf, pred, test[:, :-1].astype(int).tolist()


def random_forest(train, test):
    clf = RandomForestClassifier()
    clf.fit(train[:, :-1], train[:, -1])
    pred = clf.predict(test[:, :-1])
    return pred, pred, test[:, :-1].astype(int).tolist()


def classification(train, test, train_lengths, test_lengths):
    if conf.classifier == 'mlp':
        return mlp(train, test)
    elif conf.classifier == 'lstm':
        return lstm(train, train_lengths, test, test_lengths)
    elif conf.classifier == 'svm':
        return SVM(train, test)
    elif conf.classifier == 'rf':
        return random_forest(train, test)
