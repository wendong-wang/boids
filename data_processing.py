import os
import glob
import h5py
import numpy as np
from scipy.spatial import distance as scipy_distance
from scipy.spatial import Voronoi as ScipyVoronoi
import matplotlib.pyplot as plt


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
    p = c / np.float32(np.sum(c))
    p_nonzero = p[np.nonzero(p)]
    w_nonzero = w[np.nonzero(p)]
    h_ndist = - np.sum(p_nonzero * np.log2(p_nonzero))
    h_ndist_adjusted = -np.sum(p_nonzero * np.log2(p_nonzero/w_nonzero))
    h_ndist_normalized = h_ndist_adjusted / np.log2(w_nonzero.sum())
    return h_ndist, h_ndist_adjusted, h_ndist_normalized


# %% load the data file
project_folder = '/Users/wendong/repos/boids_vectorization'
os.chdir(project_folder)
project_folder_tree_gen = os.walk(project_folder)
_, results_folders, _ = next(project_folder_tree_gen)
results_folders.sort()

results_folder_id = -1  # check the last result
os.chdir(results_folders[results_folder_id])

data_file_list = glob.glob('*.hdf5')

f = h5py.File(data_file_list[0], 'r')
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

bin_edges_ndist = np.arange(0, 700, 20).tolist() + [1500]
H_NDist_arr = np.zeros(last_num_steps)
H_NDist_adj_arr = np.zeros(last_num_steps)
H_NDist_norm_arr = np.zeros(last_num_steps)
vel_avg_norm_arr = np.zeros(last_num_steps)

for i in np.arange(last_num_steps):
    vel_avg = np.mean(velocity_saved[:, :, i], axis=0)
    vel_avg_norm_arr[i] = np.sqrt(vel_avg[0]**2 + vel_avg[1]**2)
    vor = ScipyVoronoi(position_saved[:, :, i])
    all_vertices = vor.vertices
    neighbour_pairs = vor.ridge_points
    # ridges are perpendicular between lines drawn between the following input points:
    # row# is the index of a ridge, columns are the two point# that correspond to the ridge
    ridge_vertex_pairs = np.asarray(vor.ridge_vertices)  # used for calculating local areas
    # row# is the index of a ridge, columns are two vertex# of the ridge
    pairwise_distance_matrix = scipy_distance.cdist(position_saved[:, :, i],
                                                    position_saved[:, :, i], 'euclidean')

    neighbour_pairs_mask = neighbour_pair_list_to_matrix(n, neighbour_pairs)
    neighbour_distances = pairwise_distance_matrix[np.nonzero(neighbour_pairs_mask)]

    H_NDist_arr[i], H_NDist_adj_arr[i], H_NDist_norm_arr[i] = \
        compute_h_ndist(neighbour_distances, bin_edges_ndist)


# %% code for plotting

# plotting the last histogram
bin_edges_NDist = np.arange(0, 700, 20).tolist() + [1500]
count, edge_arr = np.histogram(neighbour_distances, bin_edges_ndist)
H_NDist, H_NDist_adj, H_NDist_norm = compute_h_ndist(neighbour_distances, bin_edges_ndist)
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.bar(bin_edges_NDist[:-1], count, align='edge', width=10)
ax.set_xlabel('neighbor distances', {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of neighbor distances, entropy: {:.3} bits'.format(H_NDist), {'size': 15})
ax.legend(['boid HNDist'])
fig.show()


# plotting H_NDist_norm
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(np.arange(last_num_steps), H_NDist_norm_arr, label='H_NDist_norm')
ax.set_xlabel('steps', {'size': 15})
ax.set_ylabel('H_NDist_norm', {'size': 15})
ax.set_title('H_NDist_norm over steps', {'size': 15})
ax.legend(loc='best')
fig.show()


# plotting average velocity over frames
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(np.arange(last_num_steps), vel_avg_norm_arr, label='average velocity')
ax.set_xlabel('steps', {'size': 15})
ax.set_ylabel('norm of average velocity', {'size': 15})
ax.set_title('norm of average velocity over steps', {'size': 15})
ax.legend(loc='best')
fig.show()


# %% close hdf5 file
f.close()

