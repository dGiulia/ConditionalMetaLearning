
import numpy as np
import matplotlib.pyplot as plt
import datetime


def plot_stuff(results, methods, dataset):

    font = {'size': 43}
    plt.rc('font', **font)

    my_dpi = 100
    plt.figure(figsize=(1664 / my_dpi, 936 / my_dpi), facecolor='white', dpi=my_dpi)

    if dataset == 'schools':

        for idx, curr_method in enumerate(methods):

            if curr_method == 'ITL':
                color = 'black'
                linestyle = '-'
                curr_method_short = 'ITL'
            elif curr_method == 'oracle_unconditional':
                color = 'red'
                linestyle = '-'
                curr_method_short = 'mean'
            elif curr_method == 'unconditional':
                color = 'tab:green'
                linestyle = '-'
                curr_method_short = 'uncond.'
            elif curr_method == 'conditional':
                color = 'tab:purple'
                linestyle = '-'
                curr_method_short = 'cond.'
            elif curr_method == 'conditional_sin_cos':
                color = 'tab:blue'
                linestyle = '-'
                curr_method_short = 'cond. circle'
            elif curr_method == 'conditional_fourier':
                color = 'tab:orange'
                linestyle = '-'
                curr_method_short = 'cond. rnd'

            diff = np.asarray(np.asarray(results['ITL']) - np.asarray(results[curr_method]))
            diff_relative = np.divide(diff, np.asarray(results['ITL'])) * 100
            mean = np.nanmean(diff_relative, axis=0)
            std = np.nanstd(diff_relative, axis=0)

            plt.plot(mean, color=color, linestyle=linestyle, linewidth=3, label=curr_method_short)
            plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1, edgecolor=color, facecolor=color,
                             antialiased=True)
            plt.xlabel('Training Tasks', fontsize=50, fontweight="normal")
            plt.ylabel('Relative Impr. %', fontsize=50, fontweight="normal")
            plt.legend()

        # plt.ylim(bottom=-10, top=300)
        # plt.ylim(top=4.7)
        plt.tight_layout()
        plt.savefig(dataset + '_' + 'temp_ITL_improvement' + '_' + str(datetime.datetime.now()).replace(':', '')
                    + '.png', format='png')
        plt.pause(0.01)
        plt.close()

    else:

        for idx, curr_method in enumerate(methods):

            if curr_method == 'ITL':
                color = 'black'
                linestyle = '-'
                curr_method_short = 'ITL'
            elif curr_method == 'oracle_unconditional':
                color = 'red'
                linestyle = '-'
                curr_method_short = 'mean'
            elif curr_method == 'unconditional':
                color = 'tab:green'
                linestyle = '-'
                curr_method_short = 'uncond.'
            elif curr_method == 'conditional':
                color = 'tab:purple'
                linestyle = '-'
                curr_method_short = 'cond.'
            elif curr_method == 'conditional_sin_cos':
                color = 'tab:blue'
                linestyle = '-'
                curr_method_short = 'cond. circle'
            elif curr_method == 'conditional_fourier':
                color = 'tab:orange'
                linestyle = '-'
                curr_method_short = 'cond. rnd'

            mean = np.nanmean(results[curr_method], axis=0)
            std = np.nanstd(results[curr_method], axis=0)

            plt.plot(mean, color=color, linestyle=linestyle, linewidth=3, label=curr_method_short)
            plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1, edgecolor=color, facecolor=color,
                             antialiased=True)
            plt.xlabel('Training Tasks', fontsize=50, fontweight="normal")
            plt.ylabel('Test Error', fontsize=50, fontweight="normal")
            plt.legend()

        # plt.ylim(bottom=-10, top=300)
        # plt.ylim(top=4.7)
        plt.tight_layout()
        plt.savefig(dataset + '_' + 'temp_test_error' + '_' + str(datetime.datetime.now()).replace(':', '') + '.png',
                    format='png')
        plt.pause(0.01)
        plt.close()


