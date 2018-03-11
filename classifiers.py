from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import keras as kr
import numpy
import utils
import preprocessing
import configuration as conf


def mlp(train, test):
    model = kr.models.Sequential()
    # model.add(kr.layers.Conv1D(2,5,strides=1, padding='valid',input_shape=(train.shape[1]-1)))
    model.add(kr.layers.Dense(units=128, activation='relu', input_shape=(train.shape[1] - 1,)))
    model.add(kr.layers.Dense(units=64, activation='relu'))
    model.add(kr.layers.Dense(units=32, activation='relu'))
    model.add(kr.layers.Dense(units=16, activation='relu'))
    model.add(kr.layers.Dense(units=len(set(train[:, -1]))))
    model.add(kr.layers.Activation('softmax'))

    sgd = kr.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(train[:, :-1], kr.utils.np_utils.to_categorical(train[:, -1]), epochs=conf.epoch, batch_size=conf.batch_size,
              shuffle=True)

    confidences = model.predict(test[:, :-1], batch_size=128)
    predictions = [numpy.argmax(x) for x in confidences]
    return confidences, predictions, test[:, -1]


def lstm(train, train_lengths, test, test_lengths):
    model = kr.models.Sequential()
    model.add(kr.layers.Bidirectional(kr.layers.LSTM(5, return_sequences=True), input_shape=(conf.look_back, train.shape[1] - 1), merge_mode='concat'))
    model.add(kr.layers.LSTM(5, return_sequences=False))
    model.add(kr.layers.Dense(32, activation='relu'))
    model.add(kr.layers.Dense(16, activation='relu'))
    model.add(kr.layers.Dense(units=len(set(train[:, -1]))))
    model.add(kr.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit_generator(preprocessing.batch_generator(train, train_lengths, conf.look_back, conf.batch_size),
                        steps_per_epoch=utils.samples_per_epoch(conf.look_back, train_lengths) / conf.batch_size,
                        epochs=conf.epoch)

    generator = preprocessing.batch_generator(test, test_lengths, conf.look_back, conf.batch_size)
    test_labels = []
    confidences = []
    for x in range(utils.samples_per_epoch(conf.look_back, test_lengths) / conf.batch_size):
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
    return conf, pred


def RF(train, test):
    clf = RandomForestClassifier()
    clf.fit(train[:, :-1], train[:, -1])
    pred = clf.predict(test[:, :-1])
    return pred, pred


def classification(train, test, train_lengths, test_lengths, n_fold):
    if conf.classifier == 'mlp':
        confidences, predictions, test_labels = mlp(train, test)
    elif conf.classifier == 'lstm':
        confidences, predictions, test_labels = lstm(train, train_lengths, test, test_lengths)
    score = utils.calculate_score(predictions, test_labels)
    utils.save_fold_results(n_fold, confidences, test_labels, score)
    return score
