import keras

'''General config'''
data_path = "dataset/simulato/"
results_path = ''
separated_test = False  # if testing set is given (i.e., no cross-validation needed)
classifier = ''
component = ''  # target component for classification (6001, 6010, etc)
classification = ''  # classification type: detection/prediction
predict_window_size = 0  # prediction windows, in seconds (0 == detection)
prediction = False

'''Simulations config'''
damage_types = ["Outer", "Inner", "Ball"]
active_sensors = [True, False, False]
damage_label = 1
transitory_label = 2


'''Neural networks config'''
batch_size = 1024
epochs = 1
look_back = 30  # sequence length for recurrent networks
optimizers = {
    'sgd': keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9),
    'adam': 'adam'
}
optimizer = optimizers['adam']


def set_results_path():
    global results_path
    results_path = '/'.join(["risultati", classifier, classification, component, ''])
