# -*- coding: utf-8 -*-

"""
@Time    : 2019/4/20 14:00
@Author  : Cassie
@Description :
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
database = 'F:/NLP-dataset/titanic/train.csv'
content = pd.read_csv(database)
content = content.dropna()
age_with_fares = content[
    (content['Age'] > 22) & (content['Fare'] < 400) & (content['Fare'] > 130)
]

# 这里不是列表，是取值
sub_fare = age_with_fares['Fare']
sub_age = age_with_fares['Age']
# plt.scatter(sub_age ,sub_fare)
# plt.show()


def func(age, k, b):
    return k * age + b


def loss(y, yhat):
    """

    :param y: real fares
    :param yhat: the estimated fares
    :return:  how good is the estimated fares
    """
    return np.mean(np.abs(y - yhat))


losses = []


# 随机选择
def exercise1():
    min_error_rate = float('inf')
    loop_times = 10000
    while loop_times > 0:
        k_hat = random.randint(-10, 10)
        b_hat = random.randint(-10, 10)
        estimated = func(sub_age, k_hat, b_hat)
        error_rate = loss(y=sub_fare, yhat=estimated)

        if error_rate < min_error_rate:
            min_error_rate = error_rate
            best_k, best_b = k_hat, b_hat
            print('loop: {}'.format(loop_times))
            losses.append(error_rate)
            print('f(age) = {} * age + {}, with error rate : {}'.format(best_k, best_b, min_error_rate))

        loop_times -= 1

    plt.scatter(sub_age, sub_fare)
    plt.plot(sub_age,estimated,c='r')
    plt.show()


# 选定方向后能移动的步长
def step():
    # return random.random() * 2 - 1  # -1到1
    return random.random() * 1  # 0 到 1


# 设置方向，沿着能让loss下降的方向的最好的k,b的方向
def exercise2():
    min_error_rate = float('inf')
    loop_times = 1000
    change_directions = [
        (+1,-1),
        (+1,+1),
        (-1,+1),
        (-1,-1)
    ]
    k_hat = random.random() * 20 - 10  # -10 到 10  这只是初始值的范围，之后会根据Loss进行调整，就不会在这个范围内，每次都会在随机的步长值内有变动
    b_hat = random.random() * 20 - 10
    best_b = b_hat
    best_k = k_hat
    direction = random.choice(change_directions)
    while loop_times > 0:
        k_delta_direction, b_delta_direction = direction
        k_delta = k_delta_direction * step()
        b_delta = b_delta_direction * step()

        new_k = best_k + k_delta
        new_b = best_b + b_delta
        estimated = func(sub_age, new_k, new_b)
        error_rate = loss(y=sub_fare, yhat=estimated)

        if error_rate < min_error_rate:
            min_error_rate = error_rate
            best_k, best_b = new_k, new_b
            direction = (k_delta_direction, b_delta_direction)
            print('loop: {}'.format(loop_times))
            losses.append(error_rate)
            print('f(age) = {} * age + {}, with error rate : {}'.format(best_k, best_b, min_error_rate))
        else:
            # direction = random.choice(change_directions)
            # 防止又选了跟上次一样的那个
            direction = random.choice(list(set(change_directions) - {(k_delta_direction, b_delta_direction)}))

        loop_times -= 1

    plt.scatter(sub_age, sub_fare)
    plt.plot(sub_age, estimated, c='r')
    plt.show()


def derivate_k(y, yhat, x):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.mean([a * -x_i for a, x_i in zip(abs_values, x)])


def derivate_b(y,yhat):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.mean([a * -1 for a in abs_values])


learning_rate = 1e-3


def exercise3():

    loop_times = 1000
    k_hat = random.random() * 20 - 10  # -10 到 10
    b_hat = random.random() * 20 - 10

    while loop_times > 0:
        # × -1的原因是要让它沿着导数下降的方向，所以导数是越来越小的
        k_delta = -1 * learning_rate * derivate_k(
            sub_fare, func(sub_age,k_hat, b_hat), sub_age
        )
        b_delta = -1 * learning_rate * derivate_b(sub_fare, func(sub_age, k_hat,b_hat))

        k_hat += k_delta
        b_hat += b_delta
        estimated = func(sub_age, k_hat, b_hat)
        error_rate = loss(y=sub_fare, yhat=estimated)
        print('loop: {}'.format(loop_times))
        losses.append(error_rate)
        print('f(age) = {} * age + {}, with error rate : {}'.format(k_hat, b_hat, error_rate))

        loop_times -= 1

    plt.scatter(sub_age, sub_fare)
    plt.plot(sub_age, estimated, c='r')
    plt.show()

exercise3()
plt.plot(range(len(losses)), losses)
plt.show()





