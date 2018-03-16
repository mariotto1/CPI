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
epochs = 3
look_back = 30  # sequence length for recurrent networks
mlp_layers = [
    {'type': 'dense', 'params': {'units': 128, 'activation': 'relu'}},
    {'type': 'dense', 'params': {'units': 64, 'activation': 'relu'}},
    {'type': 'dense', 'params': {'units': 32, 'activation': 'relu'}},
    {'type': 'dense', 'params': {'units': 16, 'activation': 'relu'}}
]
rnn_layers = [
    {'type': 'lstm', 'params': {'units': 5, 'return_sequences': True}, 'bidir': True, 'bidir_param': {'merge_mode': 'concat'}},
    {'type': 'lstm', 'params': {'units': 5, 'return_sequences': False}, 'bidir': False},
    {'type': 'dense', 'params': {'units': 32, 'activation': 'relu'}},
    {'type': 'dense', 'params': {'units': 16, 'activation': 'relu'}}
]
optimizer = {'type': 'adam', 'params': {}}


def set_results_path():
    global results_path
    results_path = '/'.join(["risultati", classifier, classification, component, ''])