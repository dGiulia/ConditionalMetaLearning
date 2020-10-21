
import numpy as np
from src.general_functions import loss, feature_map
from src.inner_algorithm import inner_algorithm


class FixedBias:

    def __init__(self, fixed_meta_parameter, lambda_par_range, loss_name):

        self.fixed_meta_parameter = fixed_meta_parameter
        self.lambda_par_range = lambda_par_range
        self.loss_name = loss_name

    def fit(self, data):

        # we use the same lambda for each task
        best_perf = np.Inf
        check_val_error = []

        for _, lambda_par in enumerate(self.lambda_par_range):

            # computing the average test error on the validation tasks
            all_validation_errors = []

            for _, task_val in enumerate(data.val_task_indexes):

                x_tr = data.features_tr[task_val]
                y_tr = data.labels_tr[task_val]
                x_ts = data.features_ts[task_val]
                y_ts = data.labels_ts[task_val]

                curr_weights, average_weights = inner_algorithm(x_tr, y_tr, lambda_par, self.fixed_meta_parameter, self.loss_name)
                validation_error = loss(x_ts, y_ts, average_weights, self.loss_name)
                all_validation_errors.append(validation_error)

            average_validation_error = np.mean(all_validation_errors)
            check_val_error.append(average_validation_error)

            if average_validation_error < best_perf:
                best_perf = average_validation_error
                best_lambda = lambda_par

        all_test_errors = []

        for _, task_ts in enumerate(data.test_task_indexes):

            x_tr = data.features_tr[task_ts]
            y_tr = data.labels_tr[task_ts]
            x_ts = data.features_ts[task_ts]
            y_ts = data.labels_ts[task_ts]

            curr_weights, average_weights = inner_algorithm(x_tr, y_tr, best_lambda, self.fixed_meta_parameter, self.loss_name)
            test_error = loss(x_ts, y_ts, average_weights, self.loss_name)
            all_test_errors.append(test_error)

        average_test_error = np.mean(all_test_errors)
        all_best_performances = average_test_error * np.ones(len(data.tr_task_indexes))

        print(f'best lambda: ', best_lambda)
        print(f'best test error: ', all_best_performances[- 1])

        return all_best_performances


class UnconditionalMetaLearning:

    def __init__(self, lambda_par_range, gamma_par_range, loss_name):

        self.lambda_par_range = lambda_par_range
        self.gamma_par_range = gamma_par_range
        self.loss_name = loss_name

    def fit(self, data):

        best_perf = np.Inf

        counter_val = 0

        for _, gamma_par in enumerate(self.gamma_par_range):
            for _, lambda_par in enumerate(self.lambda_par_range):

                counter_val = counter_val + 1
                # print(f'val: ', counter_val, ' on ', len(self.lambda_par_range) * len(self.gamma_par_range))

                all_meta_parameters_temp = []
                all_average_val_errors_temp = []  # temporary memory for the best val error curve
                all_average_test_errors_temp = []  # temporary memory for the best test error curve

                # initialize meta-parameter
                meta_parameter = np.zeros(data.features_tr[0].shape[1])

                for task_tr_index, task_tr in enumerate(data.tr_task_indexes):

                    # print(f'TRAINING task', task_tr_index + 1)

                    x = data.features_tr[task_tr]
                    y = data.labels_tr[task_tr]

                    curr_weights, average_weights = inner_algorithm(x, y, lambda_par, meta_parameter, self.loss_name)

                    # compute the meta-gradient
                    meta_gradient = - lambda_par * (curr_weights - meta_parameter)

                    # update the meta_parameter
                    meta_parameter = meta_parameter - gamma_par * meta_gradient
                    all_meta_parameters_temp.append(meta_parameter)
                    average_meta_parameter = np.mean(all_meta_parameters_temp, axis=0)

                    # compute the error on the validation and test tasks with average_meta_parameter
                    all_val_errors_temp = []
                    for _, task_val in enumerate(data.val_task_indexes):
                        x_tr = data.features_tr[task_val]
                        y_tr = data.labels_tr[task_val]
                        x_ts = data.features_ts[task_val]
                        y_ts = data.labels_ts[task_val]
                        curr_weights, average_weights = inner_algorithm(x_tr, y_tr, lambda_par, average_meta_parameter, self.loss_name)
                        val_error = loss(x_ts, y_ts, average_weights, self.loss_name)
                        all_val_errors_temp.append(val_error)
                    average_val_error = np.mean(all_val_errors_temp)
                    all_average_val_errors_temp.append(average_val_error)

                    all_test_errors_temp = []
                    for _, task_ts in enumerate(data.test_task_indexes):
                        x_tr = data.features_tr[task_ts]
                        y_tr = data.labels_tr[task_ts]
                        x_ts = data.features_ts[task_ts]
                        y_ts = data.labels_ts[task_ts]
                        curr_weights, average_weights = inner_algorithm(x_tr, y_tr, lambda_par, average_meta_parameter, self.loss_name)
                        test_error = loss(x_ts, y_ts, average_weights, self.loss_name)
                        all_test_errors_temp.append(test_error)
                    average_test_error = np.mean(all_test_errors_temp)
                    all_average_test_errors_temp.append(average_test_error)

                # select the hyper-parameters for which the last training task's average validation error is minimized
                if average_val_error < best_perf:
                    best_perf = average_val_error
                    best_lambda_par = lambda_par
                    best_gamma_par = gamma_par
                    all_best_performances = all_average_test_errors_temp

        print(f'best lambda: ', best_lambda_par, '  best gamma: ', best_gamma_par)
        print(f'best test error: ', all_best_performances[- 1])

        return all_best_performances


class ConditionalMetaLearning:

    def __init__(self, lambda_par_range, gamma_par_range, loss_name, feature_map_name, r, W, dataset):

        self.lambda_par_range = lambda_par_range
        self.gamma_par_range = gamma_par_range
        self.loss_name = loss_name
        self.feature_map_name = feature_map_name
        self.dataset = dataset
        self.r = r
        self.W = W

    def fit(self, data):

        best_perf = np.Inf
        counter_val = 0

        for _, gamma_par in enumerate(self.gamma_par_range):
            for _, lambda_par in enumerate(self.lambda_par_range):

                counter_val = counter_val + 1
                # print(f'val: ', counter_val, ' on ', len(self.lambda_par_range) * len(self.gamma_par_range))

                all_meta_parameters_temp = []
                all_average_val_errors_temp = []  # temporary memory for the best val error curve
                all_average_test_errors_temp = []  # temporary memory for the best test error curve

                # initialize meta-parameter
                if self.dataset == 'circle':
                    curr_b = np.zeros(data.features_tr[0].shape[1])
                    sum_b = np.zeros(data.features_tr[0].shape[1])
                    avg_b = np.zeros(data.features_tr[0].shape[1])
                    test_for_shape = feature_map(data.all_side_info[0], data.labels_tr[0], self.feature_map_name, self.r, self.W)
                    curr_M = np.zeros([data.features_tr[0].shape[1], test_for_shape.shape[0]])
                    sum_M = np.zeros([data.features_tr[0].shape[1], test_for_shape.shape[0]])
                    avg_M = np.zeros([data.features_tr[0].shape[1], test_for_shape.shape[0]])
                else:
                    curr_b = np.zeros(data.features_tr[0].shape[1])
                    sum_b = np.zeros(data.features_tr[0].shape[1])
                    avg_b = np.zeros(data.features_tr[0].shape[1])
                    test_for_shape = feature_map(data.features_tr[0], data.labels_tr[0], self.feature_map_name, self.r, self.W)
                    curr_M = np.zeros([data.features_tr[0].shape[1], test_for_shape.shape[0]])
                    sum_M = np.zeros([data.features_tr[0].shape[1], test_for_shape.shape[0]])
                    avg_M = np.zeros([data.features_tr[0].shape[1], test_for_shape.shape[0]])

                idx_avg = 1
                for task_tr_index, task_tr in enumerate(data.tr_task_indexes):

                    # print(f'TRAINING task', task_tr_index + 1)

                    x = data.features_tr[task_tr]
                    y = data.labels_tr[task_tr]
                    if self.dataset == 'circle':
                        s = data.all_side_info[task_tr]

                    if self.dataset == 'circle':
                       x_trasf_feature = feature_map(s, y, self.feature_map_name, self.r, self.W)
                    else:
                       x_trasf_feature = feature_map(x, y, self.feature_map_name, self.r, self.W)

                    # update the meta-parameter
                    curr_meta_parameter = avg_M @ x_trasf_feature + avg_b

                    curr_weights, average_weights = inner_algorithm(x, y, lambda_par, curr_meta_parameter, self.loss_name)

                    # compute the meta-gradient
                    meta_gradient_b = - lambda_par * (curr_weights - curr_meta_parameter)
                    meta_gradient_M = np.tensordot(meta_gradient_b, x_trasf_feature, 0)

                    # update the meta_parameter
                    curr_b = curr_b - gamma_par * meta_gradient_b
                    curr_M = curr_M - gamma_par * meta_gradient_M

                    sum_M = sum_M + curr_M
                    avg_M = sum_M / idx_avg
                    sum_b = sum_b + curr_b
                    avg_b = sum_b / idx_avg

                    idx_avg = idx_avg + 1

                    # compute the error on the validation and test tasks with average_meta_parameter
                    all_val_errors_temp = []
                    for _, task_val in enumerate(data.val_task_indexes):
                        x_tr = data.features_tr[task_val]
                        y_tr = data.labels_tr[task_val]
                        x_ts = data.features_ts[task_val]
                        y_ts = data.labels_ts[task_val]
                        if self.dataset == 'circle':
                            s = data.all_side_info[task_val]

                        if self.dataset == 'circle':
                            x_trasf_feature = feature_map(s, y_tr, self.feature_map_name, self.r, self.W)
                        else:
                            x_trasf_feature = feature_map(x_tr, y_tr, self.feature_map_name, self.r, self.W)

                        curr_meta_parameter = avg_M @ x_trasf_feature + avg_b

                        curr_weights, average_weights = inner_algorithm(x_tr, y_tr, lambda_par, curr_meta_parameter, self.loss_name)
                        val_error = loss(x_ts, y_ts, average_weights, self.loss_name)
                        all_val_errors_temp.append(val_error)
                    average_val_error = np.mean(all_val_errors_temp)
                    all_average_val_errors_temp.append(average_val_error)

                    all_test_errors_temp = []
                    for _, task_ts in enumerate(data.test_task_indexes):
                        x_tr = data.features_tr[task_ts]
                        y_tr = data.labels_tr[task_ts]
                        x_ts = data.features_ts[task_ts]
                        y_ts = data.labels_ts[task_ts]
                        if self.dataset == 'circle':
                            s = data.all_side_info[task_ts]

                        if self.dataset == 'circle':
                            x_trasf_feature = feature_map(s, y_tr, self.feature_map_name, self.r, self.W)
                        else:
                            x_trasf_feature = feature_map(x_tr, y_tr, self.feature_map_name, self.r, self.W)

                        curr_meta_parameter = avg_M @ x_trasf_feature + avg_b

                        curr_weights, average_weights = inner_algorithm(x_tr, y_tr, lambda_par, curr_meta_parameter, self.loss_name)
                        test_error = loss(x_ts, y_ts, average_weights, self.loss_name)
                        all_test_errors_temp.append(test_error)
                    average_test_error = np.mean(all_test_errors_temp)
                    all_average_test_errors_temp.append(average_test_error)

                # select the hyper-parameters for which the average validation error is minimized
                if average_val_error < best_perf:
                    best_perf = average_val_error
                    best_lambda_par = lambda_par
                    best_gamma_par = gamma_par
                    all_best_performances = all_average_test_errors_temp

        print(f'best lambda: ', best_lambda_par, '  best gamma: ', best_gamma_par)
        print(f'best test error: ', all_best_performances[- 1])

        return all_best_performances
