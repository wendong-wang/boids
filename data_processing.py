import os
import glob
import h5py
import numpy as np
import progressbar
from scipy.spatial import distance as scipy_distance
from scipy.spatial import Voronoi as ScipyVoronoi
import matplotlib.pyplot as plt
from boids_vectorized_9boids_5_3 import enlarged_pos_vel
from scipy.fftpack import fft,ifft
import statistics


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
project_folder = r'E:\Hndist'
os.chdir(project_folder)
project_folder_tree_gen = os.walk(project_folder)
_, results_folders, _ = next(project_folder_tree_gen)
results_folders.sort()

results_folder_id = 2
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
print(f)
# f.close()

# %% set up the variables and parameters

bin_edges_ndist = np.arange(0, 700, 5).tolist() + [2000]
H_NDist_arr = np.zeros(last_num_steps)
H_NDist_adj_arr = np.zeros(last_num_steps)
H_NDist_norm_arr = np.zeros(last_num_steps)
vel_avg_norm_arr = np.zeros(last_num_steps)
vel_dev_norm_arr = np.zeros(2000)  # s

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

    if i > last_num_steps - 1999:
        neighbour_distances_last = np.hstack((neighbour_distances_last, neighbour_distances))
        vel_dev = np.std(velocity_saved[:, :, i], axis=0)  # s
        vel_dev_norm_arr[i-last_num_steps+2000] = np.sqrt(vel_dev[0] ** 2 + vel_dev[1] ** 2)  # s
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
        vel_dev = np.std(velocity_saved[:, :, i], axis=0)  # s
        vel_dev_norm_arr[0] = np.sqrt(vel_dev[0] ** 2 + vel_dev[1] ** 2)  # s


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
vel_last_avg = np.mean(np.mean(velocity_saved[:, :, -1001:-1], axis=0), axis=1)
vel_last_norm_avg = np.sqrt(vel_last_avg[0]**2 + vel_last_avg[1]**2)

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
print("va norm variance last 80000 = ")
print(statistics.variance(vel_avg_norm_arr[-80001:-1]))
print("hndist norm variance last 80000 = ")
print(statistics.variance(H_NDist_norm_arr[-80001:-1]))

# save variables
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

f.create_dataset('H_NDist_norm_arr', data=H_NDist_norm_arr)

f.create_dataset('neighbour_distances_last', data=neighbour_distances_last)
f.create_dataset('neighbour_distances_last_2', data=neighbour_distances_last_2)
f.create_dataset('neighbour_distances_last_3', data=neighbour_distances_last_3)

f.create_dataset('H_NDist_per_100', data=H_NDist_per_100)
f.create_dataset('H_NDist_adj_per_100', data=H_NDist_adj_per_100)
f.create_dataset('H_NDist_norm_per_100', data=H_NDist_norm_per_100)
f.create_dataset('std_for_H_NDist', data=std_for_H_NDist)
f.create_dataset('std_for_H_NDist_norm', data=std_for_H_NDist_norm)

f.create_dataset('vel_dev_norm_arr', data=vel_dev_norm_arr)
f.create_dataset('vel_avg_norm_arr', data=vel_avg_norm_arr)


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


# s tries to plot the error bar of the average velocity
plt.errorbar(np.arange(last_num_steps)[-2001:-1], vel_avg_norm_arr[-2001:-1],
             yerr=vel_dev_norm_arr/np.sqrt(100))
ax.set_xlabel('steps', {'size': 15})
ax.set_ylabel('norm of average velocity with error bar', {'size': 15})
ax.set_title('norm of average velocity with error bar over steps', {'size': 15})
ax.legend(loc='best')
plt.savefig('norm of average velocity over last 2000 steps with error bars.png')
plt.show()



y_v = vel_avg_norm_arr[-20001:-1] - np.average(vel_avg_norm_arr[-20001:-1])
fft_y_v = fft(y_v)
N = 20000
x = np.arange(N)
half_x = x[range(int(N / 2))]  # get the half domain
abs_y_v = np.abs(fft_y_v)  # get the mod of the complex number
abs_y_v[0] = 0
angle_y_v = np.angle(fft_y_v)  # get the angle of the complex number
normalization_y_v = abs_y_v / N  # normalization
normalization_half_y_v = normalization_y_v[range(int(N / 2))]  # because symmetric, use half of the domain
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(half_x[0:100], normalization_half_y_v[0:100])
before = 0
for i,j in zip(half_x[0:100], normalization_half_y_v[0:100]):
    if 1 <= i < 30 and j >= before:
        ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
        ax.annotate('(%s,' %i, xy=(i,j))
    before = j
plt.title('Unilateral amplitude spectrum xy(after normalization) for last 20000 vel')
plt.savefig('Unilateral amplitude spectrum xy(after normalization) for last 20000 vel.png')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x[0:100], abs_y_v[0:100])
before = 0
for i,j in zip(x[0:100], abs_y_v[0:100]):
    if 1 <= i < 30 and j >= before:
        ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
        ax.annotate('(%s,' %i, xy=(i,j))
    before = j
plt.title('amplitude spectrum xy(after normalization) for last 20000 H_NDist_norm')
plt.savefig('amplitude spectrum xy(after normalization) for last 20000 H_NDist_norm.png')
plt.show()

y_v = vel_avg_norm_arr[-40001:-1] - np.average(vel_avg_norm_arr[-40001:-1])
fft_y_v = fft(y_v)
N = 40000
x = np.arange(N)
half_x = x[range(int(N / 2))]  # get the half domain
abs_y_v = np.abs(fft_y_v)  # get the mod of the complex number
abs_y_v[0] = 0
angle_y_v = np.angle(fft_y_v)  # get the angle of the complex number
normalization_y_v = abs_y_v / N  # normalization
normalization_half_y_v = normalization_y_v[range(int(N / 2))]  # because symmetric, use half of the domain
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(half_x[0:100], normalization_half_y_v[0:100])
before = 0
for i,j in zip(half_x[0:100], normalization_half_y_v[0:100]):
    if 1 <= i < 30 and j >= before:
        ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
        ax.annotate('(%s,' %i, xy=(i,j))
    before = j
plt.title('Unilateral amplitude spectrum xy(after normalization) for last 40000 vel')
plt.savefig('Unilateral amplitude spectrum xy(after normalization) for last 40000 vel.png')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x[0:100], abs_y_v[0:100])
before = 0
for i,j in zip(x[0:100], abs_y_v[0:100]):
    if 1 <= i < 30 and j >= before:
        ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
        ax.annotate('(%s,' %i, xy=(i,j))
    before = j
plt.title('amplitude spectrum xy(after normalization) for last 40000 H_NDist_norm')
plt.savefig('amplitude spectrum xy(after normalization) for last 40000 H_NDist_norm.png')
plt.show()

y_v = vel_avg_norm_arr[-80001:-1] - np.average(vel_avg_norm_arr[-80001:-1])
fft_y_v = fft(y_v)
N = 80000
x = np.arange(N)
half_x = x[range(int(N / 2))]  # get the half domain
abs_y_v = np.abs(fft_y_v)  # get the mod of the complex number
abs_y_v[0] = 0
angle_y_v = np.angle(fft_y_v)  # get the angle of the complex number
normalization_y_v = abs_y_v / N  # normalization
normalization_half_y_v = normalization_y_v[range(int(N / 2))]  # because symmetric, use half of the domain
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(half_x[0:100], normalization_half_y_v[0:100])
before = 0
for i,j in zip(half_x[0:100], normalization_half_y_v[0:100]):
    if 1 <= i < 30 and j >= before:
        ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
        ax.annotate('(%s,' %i, xy=(i,j))
    before = j
plt.title('Unilateral amplitude spectrum xy(after normalization) for last 80000 vel')
plt.savefig('Unilateral amplitude spectrum xy(after normalization) for last 80000 vel.png')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x[0:100], abs_y_v[0:100])
before = 0
for i,j in zip(x[0:100], abs_y_v[0:100]):
    if 1 <= i < 30 and j >= before:
        ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
        ax.annotate('(%s,' %i, xy=(i,j))
    before = j
plt.title('amplitude spectrum xy(after normalization) for last 80000 H_NDist_norm')
plt.savefig('amplitude spectrum xy(after normalization) for last 80000 H_NDist_norm.png')
plt.show()

"""生成数据并设置绘图参数"""
x = np.arange(last_num_steps)
y = H_NDist_norm_arr
y2 = vel_avg_norm_arr
fig, axes = plt.subplots()
# 设置两种绘图颜色
c1 = 'b'
c2 = 'r'
fontsize = 12
# 设置字体大小
plt.plot(x, y, color=c1, label='H_NDist_norm')
plt.plot(x, y2, color=c2, label='vel_avg_norm')
axes.set_xlabel("step", fontsize=fontsize)
axes.set_ylabel("H_NDist_norm (b), vel_avg_norm (r)", fontsize=fontsize)
# 设置图表标题
fig.suptitle("H_NDist_norm, vel_avg_norm vs step", fontsize=fontsize+2)
plt.savefig('H_NDist_norm and vel_avg_norm_arr vs step.png')
plt.show()



x2 = x
# 设置刻度线在坐标轴内
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
"""绘图"""
lns = []  # 用于存储绘图句柄以合并图例的list
# 创建画布
fig, axes = plt.subplots()
fig.set_size_inches(10, 8)
# 绘制图1并将绘图句柄返回，以便添加合并图例
lns1 = axes.plot(x, y, color=c1, label=c1)
lns = lns1
# 创建双x轴双y轴
twin_axes = axes.twinx().twiny()  # 使用画布的初始坐标轴对象创建第二对坐标轴，类似于在双x轴的基础上叠加双y轴
# 绘制图2并将绘图句柄返回，以便添加合并图例
lns2 = twin_axes.plot(x2, y2, color=c2, label=c2)
lns += lns2
# 设置坐标轴标注
axes.set_xlabel("step",color=c1, fontsize=fontsize)
axes.set_ylabel("H_NDist_arr",color=c1, fontsize=fontsize)
# twin_axes.set_xlabel("X2",color=c2, fontsize=fontsize)
twin_axes.set_ylabel("vel_avg_norm_arr",color=c2, fontsize=fontsize) # 第二个y轴设置标注无效
# 设置图表标题
fig.suptitle("H_NDist_norm, vel_avg_norm vs step", fontsize=fontsize+2)
# 设置第二个y轴的label；由于set_ylabel无效，因此只能通过该种方式手动添加
loc_text_x=np.min(plt.xlim())+np.ptp(plt.xlim())*1.03
loc_text_y=np.min(plt.ylim())+np.ptp(plt.ylim())*0.5
str_text = 'vel_avg_norm_arr'
twin_axes.text(loc_text_x, loc_text_y, str_text,rotation=90,color=c2,fontsize=fontsize)
# 添加图例
# lns = lns1+lns2
labs = [l.get_label() for l in lns]
axes.legend(lns, labs, loc=0, fontsize=fontsize)
plt.tight_layout()
plt.savefig('H_NDist_arr and vel_avg_norm_arr vs step2.png')
plt.show()


# %% close hdf5 file
f.close()


# a_sep, a_ali, a_coh can be adjusted, just put them to be the ratio and then multiply by the steering factor
# put 0.1 < a_sep, a_ali, a_coh < 10
# steering factor, 1e-3, 1e-2, 1e-1

# analyse last 1000 for every 100 have a Hndist and calculate its std
# 10, 100, 1000, 10000
