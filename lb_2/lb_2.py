import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'F:\Shcool\semestr 4\ТЙіМС\lb\ProbStatLabs\lb_1')# ось тут пропишіть повний шлях до папки lb_1
import lb_1


def confidence_interval_mean(sample, confidence_level):
    # Обчислення середнього вибіркового та стандартного відхилення
    sample_mean = np.mean(sample)
    sample_std = np.std(sample)

    # Обчислення z
    z_value = stats.norm.ppf((1 + confidence_level) / 2, loc=0, scale=1)

    # Обчислення довірчого інтервалу
    lower_bound = sample_mean - z_value * sample_std / np.sqrt(len(sample))
    upper_bound = sample_mean + z_value * sample_std / np.sqrt(len(sample))

    return lower_bound, upper_bound


def confidence_interval_std(sample, confidence_level):
    # Середньоквадратичне відхилення
    sample_std = np.std(sample)

    # Обчислення ступенів свободи
    dof = len(sample) - 1

    # Обчислення значень chi-квадрата
    chi2_lower = stats.chi2.ppf((1 - confidence_level) / 2, dof)
    chi2_upper = stats.chi2.ppf(confidence_level / 2, dof)

    # Обчислення довірчого інтервалу
    lower_bound = np.sqrt((dof * (sample_std ** 2)) / chi2_upper)
    upper_bound = np.sqrt((dof * (sample_std ** 2)) / chi2_lower)

    return lower_bound, upper_bound


def plot_curve_with_shaded_area(ax, conf, sample, lower_limit, upper_limit):
    # Крива
    sample = dict(sorted(sample.items()))
    x = np.array(list(sample.keys()))
    y = np.array(list(sample.values()))
    # Заливка під кривою
    ax.fill_between(x, y, color='orange', alpha=0.5)
    ax.fill_between(x, y, where=((x >= lower_limit) & (x <= upper_limit)), color='blue', alpha=0.5)

    # Налаштування відображення
    ax.set_xlabel('Значення')
    ax.set_ylabel('Частота')
    ax.set_title('Крива з рівнем довірчості ' + str(conf) + ' та обсягом вибірки ' + str(len(sample)))


confidence_levels = [0.8, 0.9, 0.95, 0.99]
sample_sizes = [10, 50, 80, 112]

mean_v = [[], [], [], []]
std_v = [[], [], [], []]
for i in range(len(confidence_levels)):

    for j in sample_sizes:
        print('Рівень довіри:', confidence_levels[i], ' для вибірки розміром:', j)
        mean = confidence_interval_mean(lb_1.sample[:j], confidence_levels[i])
        print("Довірчий інтервал на математичне сподівання: ", mean)
        std = confidence_interval_std(lb_1.sample[:j], confidence_levels[i])
        print("Довірчий інтервал на середньоквадратичне відхилення: ", std)
        print('================================================================================================')
        mean_v[i].append(mean)
        std_v[i].append(std)

for k in range(len(mean_v)):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    plt.tight_layout(pad=4)
    plot_curve_with_shaded_area(axs[0, 0], confidence_levels[k], dict(list(lb_1.sample_dict.items())[:10]),
                                mean_v[k][0][0], mean_v[k][0][1])
    plot_curve_with_shaded_area(axs[0, 1], confidence_levels[k], dict(list(lb_1.sample_dict.items())[:50]),
                                mean_v[k][1][0], mean_v[k][1][1])
    plot_curve_with_shaded_area(axs[1, 0], confidence_levels[k], dict(list(lb_1.sample_dict.items())[:80]),
                                mean_v[k][2][0], mean_v[k][2][1])
    plot_curve_with_shaded_area(axs[1, 1], confidence_levels[k], dict(list(lb_1.sample_dict.items())[:112]),
                                mean_v[k][3][0], mean_v[k][3][1])
    plt.show()

for k in range(len(std_v)):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    plt.tight_layout(pad=4)
    plot_curve_with_shaded_area(axs[0, 0], confidence_levels[k], dict(list(lb_1.sample_dict.items())[:10]),
                                std_v[k][0][0], std_v[k][0][1])
    plot_curve_with_shaded_area(axs[0, 1], confidence_levels[k], dict(list(lb_1.sample_dict.items())[:50]),
                                std_v[k][1][0], std_v[k][1][1])
    plot_curve_with_shaded_area(axs[1, 0], confidence_levels[k], dict(list(lb_1.sample_dict.items())[:80]),
                                std_v[k][2][0], std_v[k][2][1])
    plot_curve_with_shaded_area(axs[1, 1], confidence_levels[k], dict(list(lb_1.sample_dict.items())[:112]),
                                std_v[k][3][0], std_v[k][3][1])
    plt.show()
