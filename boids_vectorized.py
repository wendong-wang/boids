import numpy as np
import matplotlib.pyplot as plt
import progressbar
import datetime
import os
import cv2 as cv
import glob
import h5py
from random import uniform
from scipy.spatial import Voronoi as ScipyVoronoi
import math

# functional definitions
def compute_alignment(vel, const_speed, neighbor_mask, neighbour_count):
    """
    :param vel: velocity array: (num_of_boids, 2)
    :param const_speed: constant speed, scalar
    :param neighbor_mask: adjacency matrix (num_of_boids, num_of_boids)
    :param neighbour_count: the number of neighbours for each boid, (num_of_boids, )
    :return: alig: the alignment, (num_of_boids, 2)
    """
    num_of_boids = vel.shape[0]
    # Compute the average velocities of neighbours
    target = neighbor_mask @ vel / neighbour_count.reshape(num_of_boids, 1)

    # normalize the target velocity
    norm_of_target = np.sqrt((target * target).sum(axis=1)).reshape(num_of_boids, 1)
    np.divide(target, norm_of_target, out=target, where=norm_of_target != 0)

    # scale the target velocity to the constant speed
    # target *= const_speed

    # target = current velocity + the alignment
    # alig = target - vel
    alig = target
    return alig


# cohesion
def compute_cohesion(pos, vel, const_speed, neighbour_mask, neighbour_count):
    """
    :param pos: positions of boids, (num of boids, 2)
    :param vel: velocities of boids, (num of boids, 2)
    :param const_speed: constant speed
    :param neighbour_mask: adjacency matrix, (num of boids, num of boids)
    :param neighbour_count: the number of neighbours for each boid (num of boids, )
    :return: coh: cohesion
    """
    num_of_boids = vel.shape[0]
    # Compute the center of mass of  neighbours
    center = neighbour_mask @ pos / neighbour_count.reshape(num_of_boids, 1)

    # Compute the target vector pointing from the current position to the center of neighbours
    # target = center - pos
    target = np.zeros(pos.shape)
    count = neighbour_mask.sum(axis=1)
    nonzero = np.hstack((count.reshape(num_of_boids, 1), count.reshape(num_of_boids, 1)))
    np.subtract(center, pos, out=target, where=nonzero != 0)

    # Normalize the target vector (after the normalization, the dimension of the target is 1.)
    norm_of_target = np.sqrt((target * target).sum(axis=1)).reshape(num_of_boids, 1)
    np.divide(target, norm_of_target, out=target, where=norm_of_target != 0)

    # Scale the target vector by the constant speed
    # target *= const_speed

    # target = current velocity + cohesion
    # coh = target - vel
    coh = target
    return coh


# separation
def compute_separation(diff_x, diff_y, dist, vel, const_speed, neighbour_mask, neighbour_count):
    """
    :param diff_x: difference in x, (num_of_boids, num_of_boids)
    :param diff_y: difference in y, (num_of_boids, num_of_boids)
    :param dist: distance array, (num_of_boids, num_of_boids)
    :param vel: velocities of boids, (num_of_boids, 2)
    :param const_speed: constant speed
    :param neighbour_mask: adjacency matrix of boids, (num_of_boids, num_of_boids)
    :param neighbour_count: the count of the number of neighbours, (num_of_boids, )
    :return: sep: separation
    """
    num_of_boids = vel.shape[0]
    # compute the repulsion tensor; shape of repulsion: (num_of_boids, num_of_boids, 2)
    repulsion = np.dstack((diff_x, diff_y))

    # the repulsion is inversely proportional to the distance (scaling law ~1/r)
    np.divide(repulsion, dist.reshape(num_of_boids, num_of_boids, 1) ** 3, out=repulsion,
              where=dist.reshape(num_of_boids, num_of_boids, 1) != 0)

    # the target vector considers the repulsions from only neighbours; shape: (num_of_boids, 2)
    target = (repulsion * neighbour_mask.reshape(num_of_boids, num_of_boids, 1)).sum(axis=1) / \
             neighbour_count.reshape(num_of_boids, 1)

    # Normalize the target vector
    norm_of_target = np.sqrt((target * target).sum(axis=1)).reshape(num_of_boids, 1)
    np.divide(target, norm_of_target, out=target, where=norm_of_target != 0)

    # scale the target vector by constant speed
    # target *= const_speed

    # Compute the resulting steering
    # sep = target - vel
    sep = target
    return sep


def neighbour_pair_list_to_matrix(num_of_pts, nlist):
    """
    :param num_of_pts: number of points (boids, rafts...)
    :param nlist: neighbour list; shape (num_of_pts, 2); integer are row# (or col#) of position array
    :return: ngb_mask: neighbour mask array
    """
    ngb_mask = np.zeros((num_of_pts, num_of_pts))
    ngb_mask[nlist[:, 0], nlist[:, 1]] = 1
    ngb_mask[nlist[:, 1], nlist[:, 0]] = 1
    # use np.allclose() to check if the array is symmetric.
    # np.allclose(ngb_mask, ngb_mask.T)
    return ngb_mask


def enlarged_pos_vel(position, velocity, percent_to_involve, real_boids):  # update the enlarged position matrix
    """
    :This function is used for generate an enlarged matrix to achieve periodic boundary condition
    """
    # right
    p = np.percentile(position[0:real_boids, 0], percent_to_involve)
    logic = position[0:real_boids, 0] <= p
    x = logic*position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[real_boids:int(real_boids * (1 + percent_to_involve / 100)), 0] = new_x + 1000
    y = logic*position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[real_boids:int(real_boids * (1 + percent_to_involve / 100)), 1] = new_y

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[real_boids:int(real_boids * (1 + percent_to_involve / 100)), 0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[real_boids:int(real_boids * (1 + percent_to_involve / 100)), 1] = vy

    # down
    p = np.percentile(1000 - position[0:real_boids, 1], percent_to_involve)
    logic = 1000 - position[0:real_boids, 1] <= p
    x = logic * position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[int(real_boids * (1 + percent_to_involve / 100)):int(real_boids * (1 + 2 * percent_to_involve / 100)),
    0] = new_x
    y = logic * position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[int(real_boids * (1 + percent_to_involve / 100)):int(real_boids * (1 + 2 * percent_to_involve / 100)),
    1] = new_y - 1000

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[int(real_boids * (1 + percent_to_involve / 100)):int(real_boids * (1 + 2 * percent_to_involve / 100)),
    0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[int(real_boids * (1 + percent_to_involve / 100)):int(real_boids * (1 + 2 * percent_to_involve / 100)),
    1] = vy

    # left
    p = np.percentile(1000 - position[0:real_boids, 0], percent_to_involve)
    logic = 1000 - position[0:real_boids, 0] <= p
    x = logic * position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[int(real_boids * (1 + 2 * percent_to_involve / 100)):int(real_boids * (1 + 3 * percent_to_involve / 100)),
    0] = new_x - 1000
    y = logic * position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[int(real_boids * (1 + 2 * percent_to_involve / 100)):int(real_boids * (1 + 3 * percent_to_involve / 100)),
    1] = new_y

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[int(real_boids * (1 + 2 * percent_to_involve / 100)):int(real_boids * (1 + 3 * percent_to_involve / 100)),
    0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[int(real_boids * (1 + 2 * percent_to_involve / 100)):int(real_boids * (1 + 3 * percent_to_involve / 100)),
    1] = vy

    # up
    p = np.percentile(position[0:real_boids, 1], percent_to_involve)
    logic = position[0:real_boids, 1] <= p
    x = logic * position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[int(real_boids * (1 + 3 * percent_to_involve / 100)):int(real_boids * (1 + 4 * percent_to_involve / 100)),
    0] = new_x
    y = logic * position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[int(real_boids * (1 + 3 * percent_to_involve / 100)):int(real_boids * (1 + 4 * percent_to_involve / 100)),
    1] = new_y + 1000

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[int(real_boids * (1 + 3 * percent_to_involve / 100)):int(real_boids * (1 + 4 * percent_to_involve / 100)),
    0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[int(real_boids * (1 + 3 * percent_to_involve / 100)):int(real_boids * (1 + 4 * percent_to_involve / 100)),
    1] = vy

    # right-low
    p = np.percentile(np.hypot(position[0:real_boids, 0], 1000 - position[0:real_boids, 1]), percent_to_involve)
    logic = (np.hypot(position[0:real_boids, 0], 1000 - position[0:real_boids, 1])) <= p
    x = logic * position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[int(real_boids * (1 + 4 * percent_to_involve / 100)):int(real_boids * (1 + 5 * percent_to_involve / 100)),
    0] = new_x + 1000
    y = logic * position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[int(real_boids * (1 + 4 * percent_to_involve / 100)):int(real_boids * (1 + 5 * percent_to_involve / 100)),
    1] = new_y - 1000

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[int(real_boids * (1 + 4 * percent_to_involve / 100)):int(real_boids * (1 + 5 * percent_to_involve / 100)),
    0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[int(real_boids * (1 + 4 * percent_to_involve / 100)):int(real_boids * (1 + 5 * percent_to_involve / 100)),
    1] = vy

    # left-low
    p = np.percentile(np.hypot(1000 - position[0:real_boids, 0], 1000 - position[0:real_boids, 1]), percent_to_involve)
    logic = (np.hypot(1000 - position[0:real_boids, 0], 1000 - position[0:real_boids, 1])) <= p
    x = logic * position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[int(real_boids * (1 + 5 * percent_to_involve / 100)):int(real_boids * (1 + 6 * percent_to_involve / 100)),
    0] = new_x - 1000
    y = logic * position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[int(real_boids * (1 + 5 * percent_to_involve / 100)):int(real_boids * (1 + 6 * percent_to_involve / 100)),
    1] = new_y - 1000

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[int(real_boids * (1 + 5 * percent_to_involve / 100)):int(real_boids * (1 + 6 * percent_to_involve / 100)),
    0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[int(real_boids * (1 + 5 * percent_to_involve / 100)):int(real_boids * (1 + 6 * percent_to_involve / 100)),
    1] = vy

    # left-up
    p = np.percentile(np.hypot(1000 - position[0:real_boids, 0], position[0:real_boids, 1]), percent_to_involve)
    logic = (np.hypot(1000 - position[0:real_boids, 0], position[0:real_boids, 1])) <= p
    x = logic * position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[int(real_boids * (1 + 6 * percent_to_involve / 100)):int(real_boids * (1 + 7 * percent_to_involve / 100)),
    0] = new_x - 1000
    y = logic * position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[int(real_boids * (1 + 6 * percent_to_involve / 100)):int(real_boids * (1 + 7 * percent_to_involve / 100)),
    1] = new_y + 1000

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[int(real_boids * (1 + 6 * percent_to_involve / 100)):int(real_boids * (1 + 7 * percent_to_involve / 100)),
    0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[int(real_boids * (1 + 6 * percent_to_involve / 100)):int(real_boids * (1 + 7 * percent_to_involve / 100)),
    1] = vy

    # right-up
    p = np.percentile(np.hypot(position[0:real_boids, 0], position[0:real_boids, 1]), percent_to_involve)
    logic = (np.hypot(position[0:real_boids, 0], position[0:real_boids, 1])) <= p
    x = logic * position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[int(real_boids * (1 + 7 * percent_to_involve / 100)):int(real_boids * (1 + 8 * percent_to_involve / 100)),
    0] = new_x + 1000
    y = logic * position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[int(real_boids * (1 + 7 * percent_to_involve / 100)):int(real_boids * (1 + 8 * percent_to_involve / 100)),
    1] = new_y + 1000

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[int(real_boids * (1 + 7 * percent_to_involve / 100)):int(real_boids * (1 + 8 * percent_to_involve / 100)),
    0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[int(real_boids * (1 + 7 * percent_to_involve / 100)):int(real_boids * (1 + 8 * percent_to_involve / 100)),
    1] = vy

    return position, velocity







def enlarged_pos_vel_detail(position, velocity, percent_to_involve, real_boids, denominator):  # update the enlarged position matrix
    # right and left
    for i in range(denominator):
        logic1 = (1000 * i/denominator < position[0:real_boids, 1]) * (position[0:real_boids, 1] < 1000 * (i+1)/denominator)
        num_of_b_in_certain_range = np.sum(logic1)
        b_in_certain_range = logic1 * position[0:real_boids, 0]
        p_right = np.percentile(b_in_certain_range, (1 - num_of_b_in_certain_range/real_boids)*100 + percent_to_involve/denominator)
        p_left = np.percentile(1000 - b_in_certain_range, percent_to_involve/denominator)
        logic2_right = b_in_certain_range <= p_right
        logic2_left = (1000 - b_in_certain_range) <= p_left
        logic_right = logic1*logic2_right
        logic_left = logic1 * logic2_left

        x_right = logic_right*position[0:real_boids, 0]
        new_x_right = x_right.ravel()[np.flatnonzero(x_right)]

        x_left = logic_left * position[0:real_boids, 0]
        new_x_left = x_left.ravel()[np.flatnonzero(x_left)]

        position[int(real_boids*(1+percent_to_involve/100*i/denominator)):int(
            real_boids*(1+percent_to_involve/100*(i+1)/denominator)), 0] = new_x_right + 1000
        position[int(real_boids * (1 + 2*percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + 2*percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 0] = new_x_left - 1000

        y_right = logic_right*position[0:real_boids, 1]
        new_y_right = y_right.ravel()[np.flatnonzero(y_right)]

        y_left = logic_left * position[0:real_boids, 1]
        new_y_left = y_left.ravel()[np.flatnonzero(y_left)]

        position[int(real_boids*(1 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + percent_to_involve / 100 * (i + 1) / denominator)), 1] = new_y_right
        position[int(real_boids * (1 + 2 * percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + 2 * percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 1] = new_y_left

        v_x_right = logic_right * velocity[0:real_boids, 0]
        vx_right = v_x_right.ravel()[np.flatnonzero(v_x_right)]

        v_x_left = logic_left * velocity[0:real_boids, 0]
        vx_left = v_x_left.ravel()[np.flatnonzero(v_x_left)]

        velocity[int(real_boids * (1 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + percent_to_involve / 100 * (i + 1) / denominator)), 0] = vx_right
        velocity[int(real_boids * (1 + 2 * percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + 2 * percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 0] = vx_left

        v_y_right = logic_right * velocity[0:real_boids, 1]
        v_y_left = logic_left * velocity[0:real_boids, 1]

        vy_right = v_y_right.ravel()[np.flatnonzero(v_y_right)]
        vy_left = v_y_left.ravel()[np.flatnonzero(v_y_left)]

        velocity[int(real_boids * (1 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + percent_to_involve / 100 * (i + 1) / denominator)), 1] = vy_right
        velocity[int(real_boids * (1 + 2 * percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + 2 * percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 1] = vy_left

    # down
    for i in range(denominator):
        logic1 = (1000 * i / denominator < position[0:real_boids, 0]) * (position[0:real_boids, 0] < 1000 * (i + 1) / denominator)
        num_of_b_in_certain_range = np.sum(logic1)
        b_in_certain_range = logic1 * position[0:real_boids, 1]
        p_down = np.percentile(1000 - b_in_certain_range,
                               percent_to_involve / denominator)
        p_up = np.percentile(b_in_certain_range,
                             (1 - num_of_b_in_certain_range / real_boids)*100 + percent_to_involve / denominator)
        logic2_down = (1000 - b_in_certain_range) <= p_down
        logic2_up = b_in_certain_range <= p_up
        logic_down = logic1 * logic2_down
        logic_up = logic1 * logic2_up

        x_down = logic_down * position[0:real_boids, 0]
        new_x_down = x_down.ravel()[np.flatnonzero(x_down)]

        x_up = logic_up * position[0:real_boids, 0]
        new_x_up = x_up.ravel()[np.flatnonzero(x_up)]

        position[int(real_boids * (1 + percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 0] = new_x_down - 1000
        position[int(real_boids * (1 + 3 * percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + 3 * percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 0] = new_x_up + 1000

        y_down = logic_down * position[0:real_boids, 1]
        new_y_down = y_down.ravel()[np.flatnonzero(y_down)]

        y_up = logic_up * position[0:real_boids, 1]
        new_y_up = y_up.ravel()[np.flatnonzero(y_up)]

        position[int(real_boids * (1 + percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 1] = new_y_down
        position[int(real_boids * (1 + 3 * percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + 3 * percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 1] = new_y_up

        v_x_down = logic_down * velocity[0:real_boids, 0]
        vx_down = v_x_down.ravel()[np.flatnonzero(v_x_down)]

        v_x_up = logic_up * velocity[0:real_boids, 0]
        vx_up = v_x_up.ravel()[np.flatnonzero(v_x_up)]

        velocity[int(real_boids * (1 + percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 0] = vx_down
        velocity[int(real_boids * (1 + 3 * percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + 3 * percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 0] = vx_up

        v_y_down = logic_down * velocity[0:real_boids, 1]
        v_y_up = logic_up * velocity[0:real_boids, 1]

        vy_down = v_y_down.ravel()[np.flatnonzero(v_y_down)]
        vy_up = v_y_up.ravel()[np.flatnonzero(v_y_up)]

        velocity[int(real_boids * (1 + percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 1] = vy_down
        velocity[int(real_boids * (1 + 3 * percent_to_involve/100 + percent_to_involve / 100 * i / denominator)):int(
            real_boids * (1 + 3 * percent_to_involve/100 + percent_to_involve / 100 * (i + 1) / denominator)), 1] = vy_up

    # right-low
    p = np.percentile(np.hypot(position[0:real_boids, 0], 1000 - position[0:real_boids, 1]), percent_to_involve)
    logic = (np.hypot(position[0:real_boids, 0], 1000 - position[0:real_boids, 1])) <= p
    x = logic * position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[int(real_boids * (1 + 4 * percent_to_involve / 100)):int(real_boids * (1 + 5 * percent_to_involve / 100)),
    0] = new_x + 1000
    y = logic * position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[int(real_boids * (1 + 4 * percent_to_involve / 100)):int(real_boids * (1 + 5 * percent_to_involve / 100)),
    1] = new_y - 1000

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[int(real_boids * (1 + 4 * percent_to_involve / 100)):int(real_boids * (1 + 5 * percent_to_involve / 100)),
    0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[int(real_boids * (1 + 4 * percent_to_involve / 100)):int(real_boids * (1 + 5 * percent_to_involve / 100)),
    1] = vy

    # left-low
    p = np.percentile(np.hypot(1000 - position[0:real_boids, 0], 1000 - position[0:real_boids, 1]), percent_to_involve)
    logic = (np.hypot(1000 - position[0:real_boids, 0], 1000 - position[0:real_boids, 1])) <= p
    x = logic * position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[int(real_boids * (1 + 5 * percent_to_involve / 100)):int(real_boids * (1 + 6 * percent_to_involve / 100)),
    0] = new_x - 1000
    y = logic * position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[int(real_boids * (1 + 5 * percent_to_involve / 100)):int(real_boids * (1 + 6 * percent_to_involve / 100)),
    1] = new_y - 1000

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[int(real_boids * (1 + 5 * percent_to_involve / 100)):int(real_boids * (1 + 6 * percent_to_involve / 100)),
    0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[int(real_boids * (1 + 5 * percent_to_involve / 100)):int(real_boids * (1 + 6 * percent_to_involve / 100)),
    1] = vy

    # left-up
    p = np.percentile(np.hypot(1000 - position[0:real_boids, 0], position[0:real_boids, 1]), percent_to_involve)
    logic = (np.hypot(1000 - position[0:real_boids, 0], position[0:real_boids, 1])) <= p
    x = logic * position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[int(real_boids * (1 + 6 * percent_to_involve / 100)):int(real_boids * (1 + 7 * percent_to_involve / 100)),
    0] = new_x - 1000
    y = logic * position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[int(real_boids * (1 + 6 * percent_to_involve / 100)):int(real_boids * (1 + 7 * percent_to_involve / 100)),
    1] = new_y + 1000

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[int(real_boids * (1 + 6 * percent_to_involve / 100)):int(real_boids * (1 + 7 * percent_to_involve / 100)),
    0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[int(real_boids * (1 + 6 * percent_to_involve / 100)):int(real_boids * (1 + 7 * percent_to_involve / 100)),
    1] = vy

    # right-up
    p = np.percentile(np.hypot(position[0:real_boids, 0], position[0:real_boids, 1]), percent_to_involve)
    logic = (np.hypot(position[0:real_boids, 0], position[0:real_boids, 1])) <= p
    x = logic * position[0:real_boids, 0]
    new_x = x.ravel()[np.flatnonzero(x)]
    position[int(real_boids * (1 + 7 * percent_to_involve / 100)):int(real_boids * (1 + 8 * percent_to_involve / 100)),
    0] = new_x + 1000
    y = logic * position[0:real_boids, 1]
    new_y = y.ravel()[np.flatnonzero(y)]
    position[int(real_boids * (1 + 7 * percent_to_involve / 100)):int(real_boids * (1 + 8 * percent_to_involve / 100)),
    1] = new_y + 1000

    v_x = logic * velocity[0:real_boids, 0]
    vx = v_x.ravel()[np.flatnonzero(v_x)]
    velocity[int(real_boids * (1 + 7 * percent_to_involve / 100)):int(real_boids * (1 + 8 * percent_to_involve / 100)),
    0] = vx
    v_y = logic * velocity[0:real_boids, 1]
    vy = v_y.ravel()[np.flatnonzero(v_y)]
    velocity[int(real_boids * (1 + 7 * percent_to_involve / 100)):int(real_boids * (1 + 8 * percent_to_involve / 100)),
    1] = vy

    return position, velocity


def main():
    # %% initialize the flock
    # generating the flock, position and velocity array
    real_boids = 100
    percent_to_involve = 10
    n = int(real_boids*(1+8*percent_to_involve/100))  # num of boids
    arena_size = 1000.0   # size of the output images
    speed = 1.0
    neighbor_distance_threshold = 50
    neighbours_by_vor = 0  # 1: define neighbours by voronoi; 0: define neighbours by distance threshold
    # compare 0 and 1
    rng = np.random.default_rng()
    velocity = (rng.random((n, 2))-0.5) * speed  # uniform distribution
    position = rng.random((n, 2)) * arena_size  # uniform distribution
    # periodic boundary
    # x for boid101 should be x for boid1 + 1000
    position, velocity = enlarged_pos_vel(position, velocity, percent_to_involve, real_boids)

    # coefficients for updating position and velocity
    a_sep = 1
    a_ali = 1
    a_coh = 1
    # steering factor controls the rate of velocity update
    # smaller values means slower response to velocity change
    # bigger values (~1) leads to an interesting static phase
    steering_factor = 0.0001
    # noise added to the dir
    noise_factor = 0

    dt = 1  # time interval  # change to 1
    num_step = 100000  # total number of steps
    output_images = 1  # flag for image output
    figsave_interval = 10  # the interval for saving images

    now = datetime.datetime.now()
    output_folder = now.strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(n) + 'boids_' + \
                    str(num_step) + 'timeStep_' + \
                    str(steering_factor) + 'Steering factor_' + str(noise_factor) + 'Noise_' + \
                    str(a_sep) + 'a_sep_' + str(a_ali) + 'a_ali_' + str(a_coh) + 'a_coh_' + '9boids' + '3'

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    os.chdir(output_folder)

    # arrays to be stored
    # last_num_steps = 1000 if num_step > 1000 else num_step
    last_num_steps = num_step
    position_saved = np.zeros((real_boids, 2, last_num_steps))
    velocity_saved = np.zeros((real_boids, 2, last_num_steps))
    mask_saved = np.zeros((real_boids, real_boids, last_num_steps))
    aligment_saved = np.zeros((real_boids, 2, last_num_steps))
    separation_saved = np.zeros((real_boids, 2, last_num_steps))
    cohesion_saved = np.zeros((real_boids, 2, last_num_steps))


    # %% update the position and velocity
    for i in progressbar.progressbar(np.arange(num_step)):
        dx = np.subtract.outer(position[:, 0], position[:, 0])
        dy = np.subtract.outer(position[:, 1], position[:, 1])
        distance = np.hypot(dx, dy)
        # compute the mask
        if neighbours_by_vor == 1:
            vor = ScipyVoronoi(position)
            neighbour_pairs = vor.ridge_points
            mask = neighbour_pair_list_to_matrix(n, neighbour_pairs)
        else:
            mask = (distance > 0) * (distance < neighbor_distance_threshold)
        # exclude neighbor only contain points added
        mask[real_boids:, real_boids:] = 0
        # compute the number of neighbours; ensure at least 1 neighbour to avoid division by zero
        count = np.maximum(mask.sum(axis=1), 1)

        alignment = compute_alignment(velocity, speed, mask, count)
        separation = compute_separation(dx, dy, distance, velocity, speed, mask, count)
        cohesion = compute_cohesion(position, velocity, speed, mask, count)
        steering = a_sep * separation + a_ali * alignment + a_coh * cohesion

        velocity += steering * steering_factor  # steering has the same unit as velocity
        # mag = np.sqrt((velocity * velocity).sum(axis=1)).reshape(n, 1)
        dir = np.arctan2(velocity[:, 1], velocity[:, 0]) * 180 / np.pi
        dir += (rng.random((n,))-0.5) * noise_factor  # add noise to angle
        # velocity[:, 0] = np.cos(dir * np.pi / 180.).reshape(n,) * mag.reshape(n,)
        # velocity[:, 1] = np.sin(dir * np.pi / 180.).reshape(n,) * mag.reshape(n,)
        velocity[:, 0] = np.cos(dir * np.pi / 180.).reshape(n,) * speed
        velocity[:, 1] = np.sin(dir * np.pi / 180.).reshape(n,) * speed

        position += velocity * dt
        np.mod(position, arena_size, out=position)  # only mod 0-real_boid  # boid1 to boid100 and copy it to the other 4 area # periodic boundary condition
        # position[real_boids:2 * real_boids, 0] += 1000 # select 10% of boids (numpy.percentile), 8 area
        # position[2 * real_boids:3 * real_boids, 1] -= 1000  # enlarged_pos, enlarged_vel = function(pos, vel)
        # position[3 * real_boids:4 * real_boids, 0] -= 1000
        # position[4 * real_boids:5 * real_boids, 1] += 1000
        position, velocity = enlarged_pos_vel(position, velocity, percent_to_involve, real_boids)

        # angles = np.arctan2(velocity[:, 0], velocity[:, 1])
        # normalize the velocity for drawing velocity arrows in the images
        vel_norm = velocity
        norm = np.sqrt((velocity * velocity).sum(axis=1)).reshape(n, 1)
        np.divide(velocity, norm, out=vel_norm, where=norm != 0)

        if i >= num_step - last_num_steps:
            j = i - (num_step - last_num_steps)
            position_saved[:, :, j] = position[0:real_boids, :]
            velocity_saved[:, :, j] = velocity[0:real_boids, :]
            mask_saved[:, :, j] = mask[0:real_boids, 0:real_boids]  # don't save this
            aligment_saved[:, :, j] = alignment[0:real_boids, :]
            separation_saved[:, :, j] = separation[0:real_boids, :]
            cohesion_saved[:, :, j] = cohesion[0:real_boids, :]

        if output_images and i % figsave_interval == 0:
            figname = 'boids' + str(i).zfill(len(str(num_step))) + '.jpg'
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            # ax.set_xlim([0, arena_size])
            # ax.set_ylim([0, arena_size])
            ax.set_xlim([0, 1000])
            ax.set_ylim([0, 1000])
            ax.quiver(position[0:real_boids, 0], position[0:real_boids, 1],
                       vel_norm[0:real_boids, 0], vel_norm[0:real_boids, 1])
            #ax.quiver(position[0:real_boids, 0], position[0:real_boids, 1],
            #          cohesion[0:real_boids, 0], cohesion[0:real_boids, 1], color='red')
            #ax.quiver(position[0:real_boids, 0], position[0:real_boids, 1],
            #          alignment[0:real_boids, 0], alignment[0:real_boids, 1], color='yellow')
            #ax.quiver(position[0:real_boids, 0], position[0:real_boids, 1],
            #          separation[0:real_boids, 0], separation[0:real_boids, 1], color='green')
            # ax.quiver(position[100:180, 0], position[100:180, 1],
            #            vel_norm[100:180, 0], vel_norm[100:180, 1])
            fig.savefig(figname)
            plt.close(fig)

        # update enlarged area


    # %% writing video file
    image_files = glob.glob('*.jpg')
    image_files.sort()

    output_video_name = output_folder + '.mp4'
    output_framerate = 30
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    img = cv.imread(image_files[0])
    frameW, frameH, _ = img.shape
    video_out = cv.VideoWriter(output_video_name, fourcc,
                               output_framerate, (frameW, frameH), isColor=1)
    for i in progressbar.progressbar(np.arange(len(image_files))):
        img = cv.imread(image_files[i])
        video_out.write(img)

    video_out.release()


    # %% writing the data file
    output_file_name = output_folder + '.hdf5'

    f = h5py.File(output_file_name, 'w')
    f.create_dataset('position_saved', data=position_saved, compression='gzip', compression_opts=9)
    f.create_dataset('velocity_saved', data=velocity_saved, compression='gzip', compression_opts=9)
    f.create_dataset('mask_saved', data=mask_saved, compression='gzip', compression_opts=9)
    f.create_dataset('aligment_saved', data=aligment_saved, compression='gzip', compression_opts=9)
    f.create_dataset('separation_saved', data=separation_saved, compression='gzip', compression_opts=9)
    f.create_dataset('cohesion_saved', data=cohesion_saved, compression='gzip', compression_opts=9)
    f.create_dataset('steering_factor', data=steering_factor)  # scalar cannot be compressed
    f.create_dataset('a_sep', data=a_sep)
    f.create_dataset('a_ali', data=a_ali)
    f.create_dataset('a_coh', data=a_coh)
    print('all saved')
    f.close()

    os.chdir('..')

# periodic boundary condition achievement (5*boids or other method)

if __name__ == '__main__':
    main()


    # check the hist for
