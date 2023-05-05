import math

import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.ticker import PercentFormatter
import pandas as pd

# * Оголошення варіанту та початкових змінних
variant = 120 % 11 + 11 * 3
n = 122
sigma = 1.7

# * Генерація вибірки об'ємом n
np.random.seed(42)
sample = np.random.normal(0, sigma, n).round(4)

# * Генерація ймовірностей
math_expectations = np.random.dirichlet(np.ones(n), size=1).round(4)

# * Створення словника для спрощення подальшої роботи
sample_dict = dict(zip(sample, math_expectations[0]))

od = collections.OrderedDict(sorted(sample_dict.items()))
odv = dict(sorted(od.items(), key=lambda item: item[1], reverse=True))


#* print function
def print_dict(d:dict):
    print("{:<10}  {:<10}".format("x", "probability"))
    for x, probability in d.items():
        print("{:<10}  {:<10}".format(x, probability))


# * Функція для знаходження математичних сподівань
def calculate_mean(distribution_dict: dict) -> float:
    mean = 0
    for value, probability in distribution_dict.items():
        mean += value * probability
    return mean


#* Вибіркове середнє
def calc_sample_mean(distribution_dict: dict) -> float:
    sample_mean = 0
    for v in distribution_dict.values():
        sample_mean += v
    sample_mean *= 1 / len(distribution_dict)
    return sample_mean


#* Медіана
def calc_median(distribution_dict: dict) -> float:
    sorted_x = sorted(distribution_dict.keys())
    length = len(sorted_x)
    if length % 2 == 0:
        median = (sorted_x[round(length / 2)] + sorted_x[round(length / 2) + 1]) / 2
    else:
        median = sorted_x[round(length / 2)]
    return median


#* Мода
def calc_mode(distribution_dict: dict) -> list:
    mode = []
    list_numbers = {}

    for k in distribution_dict.keys():
        values = list(distribution_dict.keys())
        counter = values.count(k)
        list_numbers.update({k: counter})

    max_counter = max(list_numbers.values())
    for k, v in list_numbers.items():
        if v == max_counter:
            mode.append(k)

    return mode


#* Дисперсія
def calc_variance(distribution_dict: dict) -> float:
    sample_mean = calc_sample_mean(distribution_dict)
    variance = 0
    for x in distribution_dict.keys():
        variance += (x - sample_mean) ** 2
    variance /= len(distribution_dict) - 1
    return variance


#* Середньоквадратичне відхилення
def calc_standard_deviation(distribution_dict: dict) -> float:
    variance=calc_variance(distribution_dict)
    return math.sqrt(variance)


# * Полігон
# plt.plot(od.keys(), od.values())
# plt.suptitle("Полігон")

# * Гістограма
# fig = plt.figure()
# ax = fig.add_subplot (111)
# ax.hist (sample, edgecolor='black')
# plt.suptitle("Гістограма")

# * Розмаху
# plt.boxplot(od.keys(), od.values())
# plt.suptitle("Розмаху")

# * Парета
# df = pd.DataFrame.from_dict({'value': odv.values()})

# df = df.sort_values(by='value',ascending=False)
# df["cumpercentage"] = df["value"].cumsum()/df["value"].sum()*100


# fig, ax = plt.subplots()
# ax.bar(df.index, df["value"], color="C0")
# ax2 = ax.twinx()
# ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
# ax2.yaxis.set_major_formatter(PercentFormatter())

# ax.tick_params(axis="y", colors="C0")
# ax2.tick_params(axis="y", colors="C1")
# plt.suptitle("Парета")


# * Кругова
# * Створення списку для подальшої роботи
data = []
for key, value in od.items():
    if key >= 0:
        data.append((key, value))
    else:
        data.append((-key, value))

# * Створення кругової діаграми з абсолютними значеннями
plt.figure(figsize=(7, 6))
plt.pie([x[0] for x in data], labels=[str(x[0]) for x in data])


# * Функція для зворотнього відображення значень на графіку
def backmapping(value):
    for x in data:
        if x[0] == value:
            if key < 0:
                return -x[0]
            else:
                return x[0]


# * Додавання зворотнього відображення на графіку
plt.suptitle("Кругова")
plt.gca().set_aspect('equal')
# plt.gca().legend(
#     [f"{backmapping(v)} ({v:.2f})" for v in [x[0] for x in data]],
#     title="Values",
#     loc='upper right'
# )

plt.show()






#* Вивід результатів
print(f"Номер варіанту: {variant}\nВідповідно до варіанту n = {n} sigma = {sigma}")
print(f"Математичне сподівання: {calculate_mean(sample_dict).round(4)}")
print(f"Вибіркове середнє:{calculate_mean(sample_dict)}")
print(f"Медіана:{calc_median(sample_dict)}")
print(f"Мода:{calc_mode(sample_dict)}")
print(f"Дисперсія:{calc_variance(sample_dict)}")
print(f"Середньоквадратичне відхилення:{calc_standard_deviation(sample_dict)}")
print_dict(sample_dict)
