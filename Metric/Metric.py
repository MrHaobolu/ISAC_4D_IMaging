from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
import cv2
import imageio
import math
from shapely.geometry import Polygon
from scipy.spatial.distance import directed_hausdorff


def dataloader():
    # load original pos & imaging pos
    true_pos = loadmat('./data/pos_all_true.mat')
    true_pos = true_pos['pos_all_true']
    imaging_pos = loadmat('./data/pos_all.mat')
    imaging_pos = imaging_pos['pos_all']

    return true_pos, imaging_pos


def density_mertic(true_pos, imaging_pos, env_size):
    volum = np.prod(env_size)
    true_density = true_pos.shape[0] / volum
    imaging_density = imaging_pos.shape[0] / volum
    metric_value = abs((imaging_density - true_density) / true_density)

    return metric_value


def velociy_metric(true_pos, imaging_pos):
    # True Velocity
    true_array = np.unique(true_pos[:, 3])
    # Estimate Velocity
    img_array, img_count = np.unique(imaging_pos[:, 3], return_counts=True)
    img_count_indices = np.argsort(-img_count)
    if len(img_array) >= len(true_array):
        img_array_select_mid = []
        for i in range(len(true_array)):
            img_array_select_mid.append(img_array[img_count_indices[i]])
        img_array_select = np.array(img_array_select_mid)
    else:
        img_array_select = img_array
    # Calculate nmse
    true_array = np.sort(-true_array)
    img_array_select = np.sort(-img_array_select)
    sum_d = 0.0
    for index, item in enumerate(img_array_select):
        sum_d = sum_d + abs(np.std((true_array[index], item)))

    sum_d = sum_d / len(img_array_select)
    return sum_d


def pos_metric(true_pos, imaging_pos, env_size):
    true_pos_zero = np.zeros(true_pos.shape, dtype=float)
    true_pos_zero[:, 0] = true_pos[:, 0] - env_size[0]
    true_pos_zero[:, 1] = true_pos[:, 1] - env_size[1]
    true_pos_zero[:, 2] = true_pos[:, 2] - env_size[2]

    imaging_pos_zero = np.zeros(imaging_pos.shape, dtype=float)
    imaging_pos_zero[:, 0] = imaging_pos[:, 0] - env_size[0]
    imaging_pos_zero[:, 1] = imaging_pos[:, 1] - env_size[1]
    imaging_pos_zero[:, 2] = imaging_pos[:, 2] - env_size[2]

    Hausdorff_distance_1 = max(directed_hausdorff(true_pos_zero[:, 0:1], imaging_pos_zero[:, 0:1])[0],
                               directed_hausdorff(imaging_pos_zero[:, 0:1], true_pos_zero[:, 0:1])[0])
    Hausdorff_distance_2 = max(directed_hausdorff(true_pos_zero[:, 1:2], imaging_pos_zero[:, 1:2])[0],
                               directed_hausdorff(imaging_pos_zero[:, 1:2], true_pos_zero[:, 1:2])[0])
    Hausdorff_distance_3 = max(directed_hausdorff(true_pos_zero[:, [0, 2]], imaging_pos_zero[:, [0, 2]])[0],
                               directed_hausdorff(imaging_pos_zero[:, [0, 2]], true_pos_zero[:, [0, 2]])[0])

    Hausdorff_distance = Hausdorff_distance_1 + Hausdorff_distance_2 + Hausdorff_distance_3
    return Hausdorff_distance


if __name__ == '__main__':
    true_pos, imaging_pos = dataloader()
    env_size = [30, 20, 20]
    pos_metric_value = pos_metric(true_pos, imaging_pos, env_size)
    velociy_metric_value = velociy_metric(true_pos, imaging_pos)
    density_mertic_value = density_mertic(true_pos, imaging_pos, env_size)
    metric_value = pos_metric_value + velociy_metric_value + density_mertic_value
    print('pos_metric_value:', pos_metric_value, 'velociy_metric_value:', velociy_metric_value, 'density_mertic_value:', density_mertic_value, 'metric_value:', metric_value)

