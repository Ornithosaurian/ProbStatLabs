import numpy as np
from scipy import stats
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, 'F:\Shcool\semestr 4\ТЙіМС\lb\ProbStatLabs\lb_1')# ось тут пропишіть повний шлях до папки lb_1

import lb_1

def confidence_interval_mean(sample, confidence_level=0.95):

    # Обчислення середнього вибіркового та стандартного відхилення
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)

    # Обчислення допустимої похибки
    z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    margin_of_error = z_value * (sample_std / np.sqrt(len(sample)))

    # Обчислення довірчого інтервалу
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return lower_bound, upper_bound

def confidence_interval_std(sample, confidence_level=0.95):

    # Середньоквадратичне відхилення
    sample_std = np.std(sample, ddof=1)

    # Обчислення ступенів свободи
    dof = len(sample) - 1

    # Обчислення значень chi-квадрата
    chi2_lower = stats.chi2.ppf((1 - confidence_level) / 2, dof)
    chi2_upper = stats.chi2.ppf((1 + confidence_level) / 2, dof)

    # Обчислення довірчого інтервалу
    lower_bound = np.sqrt((dof * (sample_std ** 2)) / chi2_upper)
    upper_bound = np.sqrt((dof * (sample_std ** 2)) / chi2_lower)

    return lower_bound, upper_bound

print("Довірчий інтервал на математичне сподівання: ",confidence_interval_mean(lb_1.sample))
print("Довірчий інтервал на середньоквадратичне відхилення: ",confidence_interval_std(lb_1.sample))

# Задамо рівень довіри та розглянемо різні обсяги вибірки
confidence_levels = [0.8, 0.9, 0.95, 0.99]
sample_sizes = range(10, 101, 10)

mean_intervals = []
std_intervals = []

# Розрахуємо довірчі інтервали для кожної комбінації рівня довіри та обсягу вибірки
for confidence in confidence_levels:
    mean_interval = []
    std_interval = []
    for size in sample_sizes:
        sample = np.random.normal(loc=10, scale=2, size=size)
        mean_interval.append(confidence_interval_mean(sample, confidence))
        std_interval.append(confidence_interval_std(sample, confidence))
    mean_intervals.append(mean_interval)
    std_intervals.append(std_interval)

# Побудуємо графіки залежності довірчих інтервалів від рівня довіри та обсягу вибірки
plt.figure(figsize=(12, 6))

for i, confidence in enumerate(confidence_levels):
    plt.subplot(2, 2, i+1)
    plt.errorbar(sample_sizes, [interval[1]-interval[0] for interval in mean_intervals[i]], label='Mean', marker='o')
    plt.errorbar(sample_sizes, [interval[1]-interval[0] for interval in std_intervals[i]], label='Standard Deviation', marker='o')
    plt.xlabel('Sample Size')
    plt.ylabel('Interval Length')
    plt.title('Confidence Level: {}'.format(confidence))
    plt.legend()

plt.tight_layout()
plt.show()