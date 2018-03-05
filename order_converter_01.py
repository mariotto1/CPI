from utils import *
from preprocessing import *

origin_path = sys.argv[1]
target_path = sys.argv[2]
bearing_model = sys.argv[3]

dict_origin = read_data(origin_path)
dict_target = read_data(target_path)

data_path = origin_path + "0.1/"
destination_path = target_path + "0.5/"
data = import_mats(data_path)
buffer_column = numpy.float32(numpy.loadtxt("buffer_column"))

for name, damage, sim in data:
    origin_order = calculate_order(name.split('_')[4], **dict_origin)
    target_order = calculate_order(name.split('_')[4], **dict_target)
    print "Differenza ordini per {} = {}".format(name, round(target_order - origin_order, 1))

    temp = name.split('_')
    temp[-1] = str(int(temp[-1][0:2]) + int(round(target_order - origin_order, 1))) + ' ' + bearing_model + '.mat'
    name = '_'.join(temp)

    if round(target_order - origin_order, 1) == 0.0:
        sio.savemat(destination_path + name, {'export_down': reduce_to_sim_05(sim, int(temp[-1][0:2]) + int(round(target_order - origin_order, 1)))})
        continue

    origin_columns_index = []
    target_columns_index = []
    step = 1
    while step * origin_order < int(name.split('_')[-1][0:2]):
        origin_columns_index.append(order_to_column_index(round(origin_order * step, 1) if round(origin_order * step, 1) >= 1.2 else 1.2))
        target_columns_index.append(order_to_column_index(round(target_order * step, 1) if round(target_order * step, 1) >= 1.2 else 1.2))
        step += 1
    del step

    if len(buffer_column) > sim.shape[0]:
        buffer_column = buffer_column[:sim.shape[0]]
    if len(buffer_column) < sim.shape[0]:
        buffer_column = numpy.concatenate((buffer_column, buffer_column[-(sim.shape[0] - len(buffer_column)):]))

    offset = 0
    for origin_index, target_index in zip(origin_columns_index, target_columns_index):
        origin_index += offset
        n_columns = target_index - (origin_index)
        if n_columns > 0:
            sim = numpy.insert(sim, origin_index - 5, [buffer_column for j in range(n_columns)], axis=1)
        elif n_columns < 0:
            min_column = order_to_column_index(1.2)
            if origin_index <= min_column:
                sim = numpy.delete(sim, min_column, axis=1)
            else:
                sim = numpy.delete(sim, [step for step in range(origin_index - 5, origin_index + n_columns - 5, -1)], axis=1)
        offset += n_columns
    sio.savemat(destination_path + name, {'export_down': reduce_to_sim_05(sim, int(temp[-1][0:2]) + int(round(target_order - origin_order, 1)))})
