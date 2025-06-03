import itertools
import pickle

import pandas
import scipy

import numpy as np
from tqdm import tqdm




def run_scan(fun, var_axes, filename=None, trials=1, beep_on_done=True):
    to_run = list(itertools.product(*var_axes)) * trials
    results = list()
    for x in tqdm(to_run):
        result = fun(*x)
        results.append(tuple(list(x) + [result]))
    if filename is not None:
        with open(filename, "wb+") as file:
            pickle.dump(results, file)
    if beep_on_done:
        beep()
    return results


def data_to_map(data):
    result = dict()
    for row in data:
        x = tuple(row[:-1])
        r = row[-1]
        if x not in result:
            result[x] = list()
        result[x].append(r)
    return result


def calc_quantity(fun, data):
    mapped = data_to_map(data)
    result = dict()
    for k in mapped:
        values = [fun(r) for r in mapped[k]]
        result[k] = np.mean(values), np.std(values)
    return result


def qmap_to_meshgrid(mesh, quantity_map):
    meshx, meshy = mesh
    result = np.empty_like(meshx)
    for i in range(len(meshx)):
        for j in range(len(meshx[0])):
            x = meshx[i][j], meshy[i][j]
            result[i][j] = quantity_map[x][0]
    return result


def qmap_to_arrays(array, quantity_map):
    means, stds = zip(*[quantity_map[(x,)] for x in array])
    means = np.array(list(means))
    stds = np.array(list(stds))
    return array, means, stds


def qmap_to_dataframe(quantity_map, input_names, quantity_name):
    rows = list()
    for k, v in quantity_map.items():
        rows.append(list(k) + list(v))
    return pandas.DataFrame(rows, columns=input_names + [quantity_name, quantity_name + "_std"])


def qmaps_to_dataframe(quantity_maps, input_names, quantity_names):
    rows = list()
    for k in quantity_maps[0]:
        rows.append(list(k) + [q[k][0] for q in quantity_maps] + [q[k][1] for q in quantity_maps])
    return pandas.DataFrame(rows, columns=input_names + quantity_names + [q + "_std" for q in quantity_names])


num_iter = 0


# Pretty minimize function
def minimize(func, start, **kwargs):
    global num_iter
    num_iter = 0

    def callback(intermediate_result):
        global num_iter
        num_iter += 1
        print(f"{num_iter:4} {intermediate_result.fun:.5e}", end="\r")

    print("iter value")
    result = scipy.optimize.minimize(func, start, callback=callback, **kwargs)
    print()
    return result