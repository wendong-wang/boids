import os
import glob
import h5py
import numpy as np
import progressbar
from scipy.spatial import distance as scipy_distance
from scipy.fftpack import fft, ifft
from scipy.spatial import Voronoi as ScipyVoronoi
import matplotlib.pyplot as plt
from boids_vectorized_9boids import enlarged_pos_vel



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


def compute_h_ndist(ndistances, bin_edges):
    """
    :param ndistances: neighbour distances
    :param bin_edges: bin edges
    :return: (h_ndist, h_ndist_adjusted, h_ndist_normalized)
    """
    c, edge_arr = np.histogram(ndistances, bin_edges)
    w = edge_arr[1:] - edge_arr[:-1]  # width of bins
    c = np.where(c == 0, 1e-10, c)  # add a small value to c where c = 0 # check
    p = c / np.float64(np.sum(c))
    p_nonzero = p[np.nonzero(p)]
    w_nonzero = w[np.nonzero(p)]
    h_ndist = - np.sum(p_nonzero * np.log2(p_nonzero))
    h_ndist_adjusted = -np.sum(p_nonzero * np.log2(p_nonzero/w_nonzero))
    h_ndist_normalized = h_ndist_adjusted / np.log2(w_nonzero.sum())
    return h_ndist, h_ndist_adjusted, h_ndist_normalized


# %% load the data file
project_folder = r'C:\learning\VK493\hw5\vectorized_boids\reporting data_vor_square'
os.chdir(project_folder)
project_folder_tree_gen = os.walk(project_folder)
_, results_folders, _ = next(project_folder_tree_gen)
results_folders.sort()

results_folder_id = 50# check the last result
os.chdir(results_folders[results_folder_id])

data_file_list = glob.glob('*.hdf5')

f = h5py.File(data_file_list[0], 'r+')
position_saved = f['position_saved']
velocity_saved = f['velocity_saved']
mask_saved = f['mask_saved']
aligment_saved = f['aligment_saved']
separation_saved = f['separation_saved']
cohesion_saved = f['cohesion_saved']
steering_factor = f['steering_factor']
a_sep = f['a_sep']
a_ali = f['a_ali']
a_coh = f['a_coh']
n, _, last_num_steps = position_saved.shape  # n is the number of boids
print('all read')
# f.close()

# %% set up the variables and parameters

bin_edges_ndist = np.arange(0, 700, 5).tolist() + [2000]
H_NDist_arr = np.zeros(last_num_steps)
H_NDist_arr_10 = np.zeros(int(last_num_steps/10))
H_NDist_arr_100 = np.zeros(int(last_num_steps/100))
H_NDist_adj_arr = np.zeros(last_num_steps)
H_NDist_adj_arr_10 = np.zeros(int(last_num_steps/10))
H_NDist_adj_arr_100 = np.zeros(int(last_num_steps/100))
H_NDist_norm_arr = np.zeros(last_num_steps)
H_NDist_norm_arr_10 = np.zeros(int(last_num_steps/10))
H_NDist_norm_arr_100 = np.zeros(int(last_num_steps/100))
vel_avg_norm_arr = np.zeros(last_num_steps)
vel_avg_norm_arr_10 = np.zeros(int(last_num_steps/10))
vel_avg_norm_arr_100 = np.zeros(int(last_num_steps/100))
#vel_dev_norm_arr = np.zeros(2000)  # s

for i in progressbar.progressbar(np.arange(last_num_steps)):
    vel_avg = np.mean(velocity_saved[:, :, i], axis=0)
    vel_avg_norm_arr[i] = np.sqrt(vel_avg[0]**2 + vel_avg[1]**2)
    percent_to_involve = 10
    position = np.zeros((int((n*(1+8*percent_to_involve/100))), 2))
    velocity = np.zeros((int((n * (1 + 8 * percent_to_involve/100))), 2))
    position[0:n] = position_saved[:, :, i]
    velocity[0:n] = velocity_saved[:, :, i]
    position, _ = enlarged_pos_vel(position, velocity, percent_to_involve, n)
    # vor = ScipyVoronoi(position_saved[:, :, i])
    vor = ScipyVoronoi(position)  # add periodic boundary condition
    all_vertices = vor.vertices
    neighbour_pairs = vor.ridge_points  # use enlarged, delete pairs if two vertices are not in the center area

    # ridges are perpendicular between lines drawn between the following input points:
    # row# is the index of a ridge, columns are the two point# that correspond to the ridge
    ridge_vertex_pairs = np.asarray(vor.ridge_vertices)  # used for calculating local areas
    # row# is the index of a ridge, columns are two vertex# of the ridge
    # pairwise_distance_matrix = scipy_distance.cdist(position_saved[:, :, i],
    #                                                position_saved[:, :, i], 'euclidean')

    pairwise_distance_matrix = scipy_distance.cdist(position, position, 'euclidean')

    # neighbour_pairs_mask = neighbour_pair_list_to_matrix(n, neighbour_pairs)
    neighbour_pairs_mask = neighbour_pair_list_to_matrix(int(n*(1+8*percent_to_involve/100)), neighbour_pairs)
    # exclude neighbor only contain points added
    neighbour_pairs_mask[n:, n:] = 0

    neighbour_distances = pairwise_distance_matrix[np.nonzero(neighbour_pairs_mask)]

    H_NDist_arr[i], H_NDist_adj_arr[i], H_NDist_norm_arr[i] = \
        compute_h_ndist(neighbour_distances, bin_edges_ndist)

    if i % 10 == 0:
        neighbour_distances_10 = []
        vel_10 = np.zeros([10, 2])

    if i % 100 == 0:
        neighbour_distances_100 = []
        vel_100 = np.zeros([100, 2])

    neighbour_distances_10 = np.hstack((neighbour_distances_10, neighbour_distances))
    vel_10[i%10, :] = vel_avg
    neighbour_distances_100 = np.hstack((neighbour_distances_100, neighbour_distances))
    vel_100[i%100, :] = vel_avg

    if i % 10 == 9:
        H_NDist_arr_10[int((i+1)/10-1)], H_NDist_adj_arr_10[int((i+1)/10-1)], H_NDist_norm_arr_10[int((i+1)/10-1)] = \
            compute_h_ndist(neighbour_distances_10, bin_edges_ndist)
        vel_avg_10 = np.mean(vel_10[:, :], axis=0)
        vel_avg_norm_arr_10[int((i + 1) / 10 - 1)] = np.sqrt(vel_avg_10[0] ** 2 + vel_avg_10[1] ** 2)

    if i % 100 == 99:
        H_NDist_arr_100[int((i + 1) / 100 - 1)], H_NDist_adj_arr_100[int((i + 1) / 100 - 1)], H_NDist_norm_arr_100[int((i + 1) / 100 - 1)] = \
            compute_h_ndist(neighbour_distances_100, bin_edges_ndist)
        vel_avg_100 = np.mean(vel_100[:, :], axis=0)
        vel_avg_norm_arr_100[int((i+1)/100-1)] = np.sqrt(vel_avg_100[0]**2 + vel_avg_100[1]**2)

    if i > last_num_steps - 1999:
        neighbour_distances_last = np.hstack((neighbour_distances_last, neighbour_distances))
        #vel_dev = np.std(velocity_saved[:, :, i], axis=0)  # s
        #vel_dev_norm_arr[i-last_num_steps+2000] = np.sqrt(vel_dev[0] ** 2 + vel_dev[1] ** 2)  # s
        if i > last_num_steps - 199:
            neighbour_distances_last_2 = np.hstack((neighbour_distances_last_2, neighbour_distances))
            if i > last_num_steps - 19:
                neighbour_distances_last_3 = np.hstack((neighbour_distances_last_3, neighbour_distances))
            elif i == last_num_steps - 19:
                neighbour_distances_last_3 = neighbour_distances
        elif i == last_num_steps - 199:
            neighbour_distances_last_2 = neighbour_distances
    elif i == last_num_steps - 1999:
        neighbour_distances_last = neighbour_distances
        #vel_dev = np.std(velocity_saved[:, :, i], axis=0)  # s
        #vel_dev_norm_arr[0] = np.sqrt(vel_dev[0] ** 2 + vel_dev[1] ** 2)  # s


# print average parameters
H_NDist_per_100 = np.zeros(20,)
H_NDist_adj_per_100 = np.zeros(20,)
H_NDist_norm_per_100 = np.zeros(20,)
for i in range(20):
    H_NDist_per_100[i], H_NDist_adj_per_100[i], H_NDist_norm_per_100[i] = compute_h_ndist(neighbour_distances_last[int(i/20*2000):int((i+1)/20*2000)], bin_edges_ndist)
std_for_H_NDist = np.std(H_NDist_per_100)
std_for_H_NDist_adj = np.std(H_NDist_adj_per_100)
std_for_H_NDist_norm = np.std(H_NDist_norm_per_100)

H_NDist_avg, H_NDist_adj_avg, H_NDist_norm_avg = compute_h_ndist(neighbour_distances_last, bin_edges_ndist)
vel_last_avg = np.mean(np.mean(velocity_saved[:, :, -2000:], axis=0), axis=1)
vel_last_norm_avg = np.sqrt(vel_last_avg[0]**2 + vel_last_avg[1]**2)
std_for_vel_norm = np.std(vel_avg_norm_arr[-5000:])

H_NDist_avg_2, H_NDist_adj_avg_2, H_NDist_norm_avg_2 = compute_h_ndist(neighbour_distances_last_2, bin_edges_ndist)
H_NDist_avg_3, H_NDist_adj_avg_3, H_NDist_norm_avg_3 = compute_h_ndist(neighbour_distances_last_3, bin_edges_ndist)

print("H_nDist_avg = ")
print(H_NDist_avg)
print("H_nDist_norm_avg = ")
print(H_NDist_norm_avg)

print("H_nDist_avg(/10) = ")
print(H_NDist_avg_2)
print("H_nDist_norm_avg(/10) = ")
print(H_NDist_norm_avg_2)

print("H_nDist_avg(/100) = ")
print(H_NDist_avg_3)
print("H_nDist_norm_avg(/100) = ")
print(H_NDist_norm_avg_3)

print("vel_last_avg = ")
print(vel_last_avg)
print("vel_last_norm_avg = ")
print(vel_last_norm_avg)
print("H_NDist_per_100 = ")
print(H_NDist_per_100)
print("H_NDist_norm_per_100 = ")
print(H_NDist_norm_per_100)
print("std_for_H_NDist = ")
print(std_for_H_NDist)
print("std_for_H_NDist_norm = ")
print(std_for_H_NDist_norm)
print("std_for_vel_norm = ")
print(std_for_vel_norm)


# save variables
# del f['H_NDist_avg']
# del f['H_NDist_adj_avg']
# del f['H_NDist_norm_avg']
# del f['H_NDist_avg_2']
# del f['H_NDist_adj_avg_2']
# del f['H_NDist_norm_avg_2']
# del f['H_NDist_avg_3']
# del f['H_NDist_adj_avg_3']
# del f['H_NDist_norm_avg_3']
# del f['vel_last_avg']
# del f['vel_last_norm_avg']
# del f['H_NDist_norm_arr']
# del f['neighbour_distances_last']
# del f['neighbour_distances_last_2']
# del f['neighbour_distances_last_3']
# del f['H_NDist_per_100']
# del f['H_NDist_adj_per_100']
# del f['H_NDist_norm_per_100']
# del f['std_for_H_NDist']
# del f['std_for_H_NDist_norm']
# del f['std_for_vel_norm']
#del f['vel_dev_norm_arr']
f.create_dataset('H_NDist_avg', data=H_NDist_avg)
f.create_dataset('H_NDist_adj_avg', data=H_NDist_adj_avg)
f.create_dataset('H_NDist_norm_avg', data=H_NDist_norm_avg)
f.create_dataset('H_NDist_avg_2', data=H_NDist_avg_2)
f.create_dataset('H_NDist_adj_avg_2', data=H_NDist_adj_avg_2)
f.create_dataset('H_NDist_norm_avg_2', data=H_NDist_norm_avg_2)
f.create_dataset('H_NDist_avg_3', data=H_NDist_avg_3)
f.create_dataset('H_NDist_adj_avg_3', data=H_NDist_adj_avg_3)
f.create_dataset('H_NDist_norm_avg_3', data=H_NDist_norm_avg_3)
f.create_dataset('vel_last_avg', data=vel_last_avg)
f.create_dataset('vel_last_norm_avg', data=vel_last_norm_avg)
f.create_dataset('vel_avg_norm_arr', data=vel_avg_norm_arr)

f.create_dataset('H_NDist_norm_arr', data=H_NDist_norm_arr)

f.create_dataset('neighbour_distances_last', data=neighbour_distances_last)
f.create_dataset('neighbour_distances_last_2', data=neighbour_distances_last_2)
f.create_dataset('neighbour_distances_last_3', data=neighbour_distances_last_3)

f.create_dataset('H_NDist_per_100', data=H_NDist_per_100)
f.create_dataset('H_NDist_adj_per_100', data=H_NDist_adj_per_100)
f.create_dataset('H_NDist_norm_per_100', data=H_NDist_norm_per_100)
f.create_dataset('std_for_H_NDist', data=std_for_H_NDist)
f.create_dataset('std_for_H_NDist_norm', data=std_for_H_NDist_norm)
f.create_dataset('std_for_vel_norm', data=std_for_vel_norm)

# f.create_dataset('vel_dev_norm_arr', data=vel_dev_norm_arr)



# %% code for plotting

# plotting the last histogram
count, edge_arr = np.histogram(neighbour_distances, bin_edges_ndist)
H_NDist, H_NDist_adj, H_NDist_norm = compute_h_ndist(neighbour_distances, bin_edges_ndist)
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.bar(bin_edges_ndist[:-1], count, align='edge', width=10)
ax.set_xlabel('neighbor distances', {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of neighbor distances, entropy: {:.3} bits'.format(H_NDist), {'size': 15})
ax.legend(['boid HNDist'])
plt.savefig('histogram of neighbor distances.png')
fig.show()


# plotting H_NDist
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(np.arange(last_num_steps), H_NDist_arr, label='H_NDist')
ax.set_xlabel('steps', {'size': 15})
ax.set_ylabel('H_NDist', {'size': 15})
ax.set_title('H_NDist over steps', {'size': 15})
ax.legend(loc='best')
plt.savefig('H_NDist over steps.png')
fig.show()
# save figure


# plotting H_NDist_norm
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(np.arange(last_num_steps), H_NDist_norm_arr, label='H_NDist_norm')
ax.set_xlabel('steps', {'size': 15})
ax.set_ylabel('H_NDist_norm', {'size': 15})
ax.set_title('H_NDist_norm over steps', {'size': 15})
ax.legend(loc='best')
plt.savefig('H_NDist_norm over steps.png')
fig.show()
# save figure

# plotting H_NDist_norm over 10 frames
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(np.arange(5, last_num_steps+5, 10), H_NDist_norm_arr_10, label='H_NDist_norm over 10 frames')
ax.set_xlabel('steps', {'size': 15})
ax.set_ylabel('H_NDist_norm', {'size': 15})
ax.set_title('H_NDist_norm over 10 steps', {'size': 15})
ax.legend(loc='best')
plt.savefig('H_NDist_norm over 10 steps.png')
fig.show()
# save figure

# plotting H_NDist_norm over 100 frames
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(np.arange(50, last_num_steps+50, 100), H_NDist_norm_arr_100, label='H_NDist_norm over 100 frames')
ax.set_xlabel('steps', {'size': 15})
ax.set_ylabel('H_NDist_norm', {'size': 15})
ax.set_title('H_NDist_norm over 100 steps', {'size': 15})
ax.legend(loc='best')
plt.savefig('H_NDist_norm over 100 steps.png')
fig.show()
# save figure


# plotting average velocity over frames
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(np.arange(last_num_steps), vel_avg_norm_arr, label='average velocity')
ax.set_xlabel('steps', {'size': 15})
ax.set_ylabel('norm of average velocity', {'size': 15})
ax.set_title('norm of average velocity over steps', {'size': 15})
ax.legend(loc='best')
plt.savefig('norm of average velocity over steps.png')
fig.show()
# save figure

# plotting average velocity over 10 frames
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(np.arange(5, last_num_steps+5, 10), vel_avg_norm_arr_10, label='average velocity over 10 frames')
ax.set_xlabel('steps', {'size': 15})
ax.set_ylabel('norm of average velocity', {'size': 15})
ax.set_title('norm of average velocity over 10 steps', {'size': 15})
ax.legend(loc='best')
plt.savefig('norm of average velocity over 10 steps.png')
fig.show()
# save figure

# plotting average velocity over 10 frames
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(np.arange(50, last_num_steps+50, 100), vel_avg_norm_arr_100, label='average velocity over 100 frames')
ax.set_xlabel('steps', {'size': 15})
ax.set_ylabel('norm of average velocity', {'size': 15})
ax.set_title('norm of average velocity over 100 steps', {'size': 15})
ax.legend(loc='best')
plt.savefig('norm of average velocity over 100 steps.png')
fig.show()
# save figure

# plotting H_NDist_norm and v_a over steps
fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
ax2 = ax1.twinx()
ax1.plot(np.arange(last_num_steps), H_NDist_norm_arr, label='H_NDist_norm')
ax2.plot(np.arange(last_num_steps), vel_avg_norm_arr, label='average velocity', c='orangered')
ax1.set_xlabel('steps', {'size': 15})
ax1.set_ylabel('H_NDist_norm', {'size': 15})
ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_ylabel('norm of average velocity', {'size': 15})
ax1.set_title('H_NDist_norm and v_a over steps', {'size': 15})
ax1.legend(loc=1)
ax2.legend(loc=2)
plt.savefig('H_NDist_norm and v_a over steps.png')
fig.show()
# save figure


# s tries to plot the error bar of the average velocity
# plt.errorbar(np.arange(last_num_steps)[-2001:-1], vel_avg_norm_arr[-2001:-1],
#              yerr=vel_dev_norm_arr)
# ax.set_xlabel('steps', {'size': 15})
# ax.set_ylabel('norm of average velocity with error bar', {'size': 15})
# ax.set_title('norm of average velocity with error bar over steps', {'size': 15})
# ax.legend(loc='best')
# plt.savefig('norm of average velocity over last 2000 steps with error bars.png')
# plt.show()

y_v = vel_avg_norm_arr[-2000:] - np.average(vel_avg_norm_arr[-2000:])
fft_y_v = fft(y_v)
N = 2000
x = np.arange(N)
half_x = x[range(int(N / 2))]  # get the half domain

abs_y_v = np.abs(fft_y_v)  # get the mod of the complex number
angle_y_v = np.angle(fft_y_v)  # get the angle of the complex number
normalization_y_v = abs_y_v / N  # normalization
normalization_half_y_v = normalization_y_v[range(int(N / 2))]  # because symmetric, use half of the domain

plt.plot(half_x, normalization_half_y_v)
plt.title('Unilateral amplitude spectrum (after normalization) for last 2000 vel')
plt.savefig('Unilateral amplitude spectrum (after normalization) for last 2000 vel.png')
plt.show()

plt.plot(x, abs_y_v)
plt.title('amplitude spectrum (after normalization) for last 2000 H_NDist_norm')
plt.savefig('amplitude spectrum (after normalization) for last 2000 H_NDist_norm.png')
plt.show()

y_H = H_NDist_norm_arr[-2000:] - np.average(H_NDist_norm_arr[-2000:])
fft_y_H = fft(y_H)
N = 2000
x = np.arange(N)
half_x = x[range(int(N / 2))]  # get the half domain

abs_y_H = np.abs(fft_y_H)  # get the mod of the complex number
angle_y_H = np.angle(fft_y_H)  # get the angle of the complex number
normalization_y_H = abs_y_H / N  # normalization
normalization_half_y_H = normalization_y_H[range(int(N / 2))]  # because symmetric, use half of the domain

plt.plot(half_x, normalization_half_y_H)
plt.title('Unilateral amplitude spectrum (after normalization) for last 2000 H_NDist_norm')
plt.savefig('Unilateral amplitude spectrum (after normalization) for last 2000 H_NDist_norm.png')
plt.show()

plt.plot(x, abs_y_H)
plt.title('amplitude spectrum (after normalization) for last 2000 H_NDist_norm')
plt.savefig('amplitude spectrum (after normalization) for last 2000 H_NDist_norm.png')
plt.show()


# %% close hdf5 file
f.close()


# a_sep, a_ali, a_coh can be adjusted, just put them to be the ratio and then multiply by the steering factor
# put 0.1 < a_sep, a_ali, a_coh < 10
# steering factor, 1e-3, 1e-2, 1e-1

# analyse last 1000 for every 100 have a Hndist and calculate its std
# 10, 100, 1000, 10000

# std for v_a is the std for the last 2000 v
