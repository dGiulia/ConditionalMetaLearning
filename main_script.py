
import numpy as np
import matplotlib
from src.data_management import DataHandler, Settings
from src.methods import FixedBias, UnconditionalMetaLearning, ConditionalMetaLearning
from src.plotting import plot_stuff
import time
import datetime


def main():

    # Custom's selection
    exp = 'exp_synthetic_1_cluster'  # Figure 1 top-left
    # exp = 'exp_synthetic_2_clusters_mean_4'  # Figure 1 top-right
    # exp = 'exp_synthetic_2_clusters_mean_0'  # Figure 1 bottom-left
    # exp = 'exp_synthetic_circle'  # Figure 1 bottom-right
    # exp = 'exp_real_lenk'  # Figure 2 left
    # exp = 'exp_real_schools'  # Figure 2 right

    if exp == 'exp_synthetic_1_cluster':
        methods = ['ITL', 'oracle_unconditional', 'unconditional', 'conditional']
        loss_name = 'absolute'
    elif exp == 'exp_synthetic_2_clusters_mean_4':
        methods = ['ITL', 'oracle_unconditional', 'unconditional', 'conditional']
        loss_name = 'absolute'
    elif exp == 'exp_synthetic_2_clusters_mean_0':
        methods = ['ITL', 'oracle_unconditional', 'unconditional', 'conditional']
        loss_name = 'absolute'
    elif exp == 'exp_synthetic_circle':
        methods = ['ITL', 'oracle_unconditional', 'unconditional', 'conditional_sin_cos', 'conditional_fourier']
        loss_name = 'absolute'
    elif exp == 'exp_real_lenk':
        methods = ['ITL', 'unconditional', 'conditional']
        loss_name = 'absolute'
    elif exp == 'exp_real_schools':
        methods = ['ITL', 'unconditional', 'conditional']
        loss_name = 'absolute'

    font = {'size': 26}
    matplotlib.rc('font', **font)
    results = {}

    lambda_par_range = [10 ** i for i in np.linspace(-5, 5, 14)]  # inner regularization parameter lambda
    gamma_par_range = [10 ** i for i in np.linspace(-5, 5, 14)]  # meta-step size gamma

    for curr_method in methods:

        results[curr_method] = []

    tt = time.time()

    trials = 10

    for seed in range(trials):

        print(f'SEED : ', seed, ' ---------------------------------------')
        np.random.seed(seed)
        general_settings = {'seed': seed,
                            'verbose': 1}

        if exp == 'exp_synthetic_1_cluster':

            # synthetic data 1 cluster
            data_settings = {'dataset': 'synthetic-regression',
                             'n_tr_tasks': 300,
                             'n_val_tasks': 100,
                             'n_test_tasks': 80,
                             'n_all_points': 20,
                             'ts_points_pct': 0.5,
                             'n_dims': 20,
                             'noise_std': 0.2}

            settings = Settings(data_settings, 'data')
            settings.add_settings(general_settings)
            data = DataHandler(settings)
            # quantities for generating the feature map
            feature_map_name = 'linear'
            r = None
            W = None

        elif exp == 'exp_synthetic_2_clusters_mean_4':

            # synthetic data MULTI clusters w_\rho = 4
            data_settings = {'dataset': 'synthetic-regression-multi-clusters',
                             'n_tr_tasks': 300,
                             'n_val_tasks': 100,
                             'n_test_tasks': 80,
                             'n_all_points': 20,
                             'ts_points_pct': 0.5,
                             'n_dims': 20,
                             'noise_std': 0.2}

            settings = Settings(data_settings, 'data')
            settings.add_settings(general_settings)
            data = DataHandler(settings)
            # quantities for generating the feature map
            feature_map_name = 'linear'
            r = None
            W = None

        elif exp == 'exp_synthetic_2_clusters_mean_0':

            # synthetic data MULTI clusters w_\rho = 0
            data_settings = {'dataset': 'synthetic-regression-multi-clusters-BIS',
                             'n_tr_tasks': 300,
                             'n_val_tasks': 100,
                             'n_test_tasks': 80,
                             'n_all_points': 20,
                             'ts_points_pct': 0.5,
                             'n_dims': 20,
                             'noise_std': 0.2}

            settings = Settings(data_settings, 'data')
            settings.add_settings(general_settings)
            data = DataHandler(settings)
            # quantities for generating the feature map
            feature_map_name = 'linear'
            r = None
            W = None

        elif exp == 'exp_synthetic_circle':

            # synthetic data - circle
            data_settings = {'dataset': 'circle',
                             'n_tr_tasks': 300,
                             'n_val_tasks': 100,
                             'n_test_tasks': 80,
                             'n_all_points': 20,
                             'ts_points_pct': 0.5,
                             'n_dims': 20,
                             'noise_std': 0.2,
                             'radius_w': 8,
                             'sigma_w': 1}

            settings = Settings(data_settings, 'data')
            settings.add_settings(general_settings)
            data = DataHandler(settings)

        elif exp == 'exp_real_lenk':

            # Lenk dataset
            data_settings = {'dataset': 'lenk',
                             'n_tr_tasks': 100,
                             'n_val_tasks': 40,
                             'n_test_tasks': 30,
                             }

            settings = Settings(data_settings, 'data')
            settings.add_settings(general_settings)
            data = DataHandler(settings)
            # quantities for generating the feature map
            feature_map_name = 'linear_with_labels'
            r = None
            W = None

        elif exp == 'exp_real_schools':

            # Schools dataset
            data_settings = {'dataset': 'schools',
                             'n_tr_tasks': 70,
                             'n_val_tasks': 39,
                             'n_test_tasks': 30,
                             'ts_points_pct': 0.25
                             }

            settings = Settings(data_settings, 'data')
            settings.add_settings(general_settings)
            data = DataHandler(settings)
            # quantities for generating the feature map
            feature_map_name = 'fourier_vector'
            k = 1000
            sigma = 100
            d_size = data.features_tr[0].shape[1]
            r = np.random.uniform(low=0., high=2 * np.pi, size=(k, 1))
            W = np.random.randn(k, d_size) * sigma

        print(f'METHOD: ', settings.data.dataset)

        for curr_method in methods:

            # print(f'method: ', curr_method)

            if curr_method == 'ITL':
                model = FixedBias(np.zeros(data.features_tr[0].shape[1]), lambda_par_range, loss_name)
            elif curr_method == 'oracle_unconditional':
                model = FixedBias(data.oracle_unconditional, lambda_par_range, loss_name)
            elif curr_method == 'unconditional':
                model = UnconditionalMetaLearning(lambda_par_range, gamma_par_range, loss_name)
            elif curr_method == 'conditional':
                model = ConditionalMetaLearning(lambda_par_range, gamma_par_range, loss_name, feature_map_name, r, W,
                                                settings.data.dataset)
            elif curr_method == 'conditional_sin_cos':
                feature_map_name = 'circle_feature_map'
                r = None
                W = None
                model = ConditionalMetaLearning(lambda_par_range, gamma_par_range, loss_name, feature_map_name, r, W,
                                                settings.data.dataset)
            elif curr_method == 'conditional_fourier':
                feature_map_name = 'circle_fourier'
                import math
                s_dim = 50
                sigma = 2 * math.pi * 10
                r = 2 * math.pi * np.random.uniform(0.0, 1.0, s_dim)
                W = sigma * np.random.randn(s_dim)
                model = ConditionalMetaLearning(lambda_par_range, gamma_par_range, loss_name, feature_map_name, r, W,
                                                settings.data.dataset)

            errors = model.fit(data)

            results[curr_method].append(errors)

            print('%s done %5.2f' % (curr_method, time.time() - tt))

        print('seed: %d | %5.2f sec' % (seed, time.time() - tt))

    np.save(settings.data.dataset + '_' + 'temp_test_error' + '_' + str(datetime.datetime.now()).replace(':', '') +
            '.npy', results)
    plot_stuff(results, methods, settings.data.dataset)

    exit()


if __name__ == "__main__":

    main()
