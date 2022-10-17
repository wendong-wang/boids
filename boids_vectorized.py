import numpy as np
import matplotlib.pyplot as plt
import progressbar
import datetime
import os
import cv2 as cv
import glob
import h5py
from scipy.spatial import Voronoi as ScipyVoronoi

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
    target *= const_speed

    # target = current velocity + the alignment
    alig = target - vel
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
    target = center - pos

    # Normalize the target vector (after the normalization, the dimension of the target is 1.)
    norm_of_target = np.sqrt((target * target).sum(axis=1)).reshape(num_of_boids, 1)
    np.divide(target, norm_of_target, out=target, where=norm_of_target != 0)

    # Scale the target vector by the constant speed
    target *= const_speed

    # target = current velocity + cohesion
    coh = target - vel
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
    np.divide(repulsion, dist.reshape(num_of_boids, num_of_boids, 1) ** 2, out=repulsion,
              where=dist.reshape(num_of_boids, num_of_boids, 1) != 0)

    # the target vector considers the repulsions from only neighbours; shape: (num_of_boids, 2)
    target = (repulsion * neighbour_mask.reshape(num_of_boids, num_of_boids, 1)).sum(axis=1) / \
             neighbour_count.reshape(num_of_boids, 1)

    # Normalize the target vector
    norm_of_target = np.sqrt((target * target).sum(axis=1)).reshape(num_of_boids, 1)
    np.divide(target, norm_of_target, out=target, where=norm_of_target != 0)

    # scale the target vector by constant speed
    target *= const_speed

    # Compute the resulting steering
    sep = target - vel
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


# %% initialize the flock
# generating the flock, position and velocity array
n = 500  # num of boids
arena_size = 1000.0   # size of the output images
speed = 1.0
neighbor_distance_threshold = 50
neighbours_by_vor = 1  # 1: define neighbours by voronoi; 0: define neighbours by distance threshold
rng = np.random.default_rng()
velocity = rng.random((n, 2)) * speed  # uniform distribution
position = rng.random((n, 2)) * arena_size  # uniform distribution

# coefficients for updating position and velocity
a_sep = 1.0
a_ali = 1.0
a_coh = 1.0
# steering factor controls the rate of velocity update
# smaller values means slower response to velocity change
# bigger values (~1) leads to an interesting static phase
steering_factor = 0.1

dt = 0.1  # time interval
num_step = 10000  # total number of steps
output_images = 1  # flag for image output
figsave_interval = 50  # the interval for saving images

now = datetime.datetime.now()
output_folder = now.strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(n) + 'boids_' + \
                str(num_step) + 'timeStep_' + str(figsave_interval) + 'step_interval'

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
os.chdir(output_folder)

# arrays to be stored
last_num_steps = 1000 if num_step > 1000 else num_step
position_saved = np.zeros((n, 2, last_num_steps))
velocity_saved = np.zeros((n, 2, last_num_steps))
mask_saved = np.zeros((n, n, last_num_steps))
aligment_saved = np.zeros((n, 2, last_num_steps))
separation_saved = np.zeros((n, 2, last_num_steps))
cohesion_saved = np.zeros((n, 2, last_num_steps))


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
    # compute the number of neighbours; ensure at least 1 neighbour to avoid division by zero
    count = np.maximum(mask.sum(axis=1), 1)

    alignment = compute_alignment(velocity, speed, mask, count)
    separation = compute_separation(dx, dy, distance, velocity, speed, mask, count)
    cohesion = compute_cohesion(position, velocity, speed, mask, count)
    steering = a_sep * separation + a_ali * alignment + a_coh * cohesion
    velocity += steering * steering_factor  # steering has the same unit as velocity
    position += velocity * dt
    np.mod(position, arena_size, out=position)  # periodic boundary condition

    # angles = np.arctan2(velocity[:, 0], velocity[:, 1])
    # normalize the velocity for drawing velocity arrows in the images
    vel_norm = velocity
    norm = np.sqrt((velocity * velocity).sum(axis=1)).reshape(n, 1)
    np.divide(velocity, norm, out=vel_norm, where=norm != 0)

    if i >= num_step - last_num_steps:
        j = i - (num_step - last_num_steps)
        position_saved[:, :, j] = position
        velocity_saved[:, :, j] = velocity
        mask_saved[:, :, j] = mask
        aligment_saved[:, :, j] = alignment
        separation_saved[:, :, j] = separation
        cohesion_saved[:, :, j] = cohesion

    if output_images and i % figsave_interval == 0:
        figname = 'boids' + str(i).zfill(len(str(num_step))) + '.jpg'
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.set_xlim([0, arena_size])
        ax.set_ylim([0, arena_size])
        ax.quiver(position[:, 0], position[:, 1],
                  vel_norm[:, 0], vel_norm[:, 1])
        fig.savefig(figname)
        plt.close(fig)


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

