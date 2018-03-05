path_data = "dataset/simulato/"
damage_types = ["Outer", "Inner", "Ball"]
results_path = ''
separated_test = False  # if testing set is given (i.e., no cross-validation needed)
classifier = ''
component = ''  # target component for classification (nativo, 6001, 6010, etc)
classification = ''  # classification type: detection/prediction
window_size = 0  # prediction windows, in minutes (0 == detection)
batch_size = 1024
look_back = 30  # sequence length for lstm
