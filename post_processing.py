
import numpy as np
from src.plotting import plot_stuff

# Custom's selection
exp = 'exp_synthetic_1_cluster'  # Figure 1 top-left
# exp = 'exp_synthetic_2_clusters_mean_4'  # Figure 1 top-right
# exp = 'exp_synthetic_2_clusters_mean_0'  # Figure 1 bottom-left
# exp = 'exp_synthetic_circle'  # Figure 1 bottom-right
# exp = 'exp_real_lenk'  # Figure 2 left
# exp = 'exp_real_schools'  # Figure 2 right

if exp == 'exp_synthetic_1_cluster':
    methods = ['ITL', 'oracle_unconditional', 'unconditional', 'conditional']
    saved_file_name = 'saved_results/one_cluster_results.npy'
    dataset = 'synthetic-regression'
elif exp == 'exp_synthetic_2_clusters_mean_4':
    methods = ['ITL', 'oracle_unconditional', 'unconditional', 'conditional']
    saved_file_name = 'saved_results/two_clusters_mean_4_results.npy'
    dataset = 'synthetic-regression-multi-clusters'
elif exp == 'exp_synthetic_2_clusters_mean_0':
    methods = ['ITL', 'oracle_unconditional', 'unconditional', 'conditional']
    saved_file_name = 'saved_results/two_clusters_mean_0_results.npy'
    dataset = 'synthetic-regression-multi-clusters-BIS'
elif exp == 'exp_synthetic_circle':
    methods = ['ITL', 'oracle_unconditional', 'unconditional', 'conditional_sin_cos', 'conditional_fourier']
    saved_file_name = 'saved_results/circle_results.npy'
    dataset = 'circle'
elif exp == 'exp_real_lenk':
    methods = ['ITL', 'unconditional', 'conditional']
    saved_file_name = 'saved_results/lenk_results.npy'
    dataset = 'lenk'
elif exp == 'exp_real_schools':
    methods = ['ITL', 'unconditional', 'conditional']
    saved_file_name = 'saved_results/schools_results.npy'
    dataset = 'schools'

results = np.load(saved_file_name, allow_pickle='TRUE').item()
plot_stuff(results, methods, dataset)
