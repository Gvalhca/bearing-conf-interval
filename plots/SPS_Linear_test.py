import numpy as np
import matplotlib.pyplot as plt
from random import randint
from tqdm import tqdm

# Set parameters of bearing observation

a0 = 1  # initial level of alternation
a1 = 1.2  # alternation of stable regime
a2 = 8

t0 = 0  # initial time
t1 = t0 + 3  # end of period
t2 = t1 + 2

t_end = t2

k1 = (a1 - a0) / (t1 - t0)
k2 = (a2 - a1) / (t2 - t1)


# ************* Set linear function ***************************

def phi(k, a0, t0, t):
    res = a0 + k * (t - t0)
    return res


def linear_func(k, a0, t0, t):
    res = np.zeros(len(t))
    for i in range(len(t)):
        res[i] = a0 + k * (t[i] - t0)
    return res


num_points = 0
t = np.ndarray
alt_noised = np.ndarray
alt = np.ndarray
M = 200  # Quantity of random sets (less than 2^N)
q = 10  # Confidence parameter
conf = str(int((1 - q / M) * 100))

N = 3  # Number of data points to SPS analysis

Beta = np.ones([M, N])

# ************* Set noised observation ************************
def set_values(_input_data):
    global num_points, t, alt_noised
    num_points = _input_data.shape[0]
    t = np.linspace(t0, t_end, num_points)
    alt = _input_data
    alt_noised = np.zeros(num_points)

    for i in range(num_points):
        # if t[i] <= t1:
        #     alt[i] = phi(k1, a0, t0, t[i])
        # if t1 < t[i] <= t2:
        #     alt[i] = phi(k2, a1, t1, t[i])

        alt_noised[i] = alt[i] + 0.02 * randint(-10, 10) * a1
    #    alt_noised[i] = alt[i] * (1 + 0.01 * randint(-10, 10))


    # ************* Set SPS parameters ****************************

    M = 200  # Quantity of random sets (less than 2^N)
    q = 10  # Confidence parameter
    conf = str(int((1 - q / M) * 100))

    N = 3  # Number of data points to SPS analysis

    Beta = np.ones([M, N])

    for j in range(M - 1):
        for i in range(N):
            if randint(0, 1) > 0.5:
                Beta[j + 1, i] = 1
            else:
                Beta[j + 1, i] = -1
        # print(j + 1, Beta[j + 1])


# ********* SPS-indicator procedure ***************************

def sps_indicator(y0, x0, tExp, altExp, Beta, N, M, q):
    # Set interval for search of the k value

    low_ang = round(-np.pi / 2, 2)
    high_ang = round(np.pi / 2, 2)

    low_k = np.tan(low_ang)
    high_k = np.tan(high_ang)

    k_q = 360

    # Constructing of the confidence set
    coin = 0
    inl = 1
    inr = 1
    wid2 = np.tan(low_ang) - np.tan(high_ang)
    wid1 = 1.1 * wid2
    loop_num = 1
    while inr - inl < k_q - 2 and loop_num < 10:
        if inr == 1:  # first loop condition
            ang = np.linspace(low_ang, high_ang, k_q)
            k_test = np.tan(ang)
        #           print(k_test)
        else:
            k_q = 50
            k_test = np.linspace(low_k, high_k, k_q)
        Rank = np.ones(k_q)
        for s in range(k_q):
            delta = np.zeros(N)
            for i in range(N):
                delta[i] = phi(k_test[s], y0, x0, tExp[i]) - altExp[i]
            coin = 0
            H = np.zeros(M)
            for j in range(M):
                for i in range(N):
                    H[j] = H[j] + Beta[j, i] * delta[i]
                if j != 0:
                    if abs(H[j]) <= abs(H[0]):
                        Rank[s] = Rank[s] + 1
                    if abs(H[j]) == abs(H[0]):
                        coin += 1
            #       if coin != 0:
            #            tinyplus = randrange(coin + 1)
            #            Rank[k] = Rank[k] + tinyplus
            if inr == 1:  # first loop condition
                if 1 < s < k_q - 1:
                    if Rank[s - 1] == M and Rank[s] < M:
                        inl = s - 2
                    #                      print('inl =', inl)
                    if Rank[s - 1] < M and Rank[s] == M:
                        inr = s + 1
            #                      print('inr =', inr)
            else:
                if 0 < s < k_q:
                    if Rank[s - 1] >= M - q and Rank[s] < M - q:
                        inl = s - 1
                    if Rank[s - 1] < M - q and Rank[s] >= M - q:
                        inr = s
        low_k = k_test[inl]
        high_k = k_test[inr]
        wid1 = wid2
        wid2 = high_k - low_k
        #       print(Rank)
        #       print(inl, inr, low_k, high_k, coin, loop_num)
        loop_num += 1
    return [low_k, high_k]


# **************** Data analysis via SPS *********
#
# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w
#
#
# #y0 = alt_noised[0]
# x0 = t0
# y0 = alt_noised[0]
# tExp = t
# altExp = alt_noised
#
# k_min = np.zeros(num_points - N)
# k_max = np.zeros(num_points - N)
# alt_min = np.zeros(num_points - N)
# alt_max = np.zeros(num_points - N)
#
# y0 = np.zeros(num_points - N)
#
# key = 0            # number of intervals greater than previous one
#
# for i in tqdm(range(num_points - N)):
#     tExp = t[i:i + N]
#     altExp = alt_noised[i:i + N]
#     y0_test_q = 20
#     y0_test = np.linspace(0.05 * alt_noised[0], 5 * alt_noised[0], y0_test_q)
#     wide = 10 * (max(altExp) - min(altExp))
#     alt_min0 = 0
#     alt_max0 = 0
# #    print('initial wide = {0}'.format(wide))
#     for j in range(y0_test_q):
#         [k_min[i], k_max[i]] = sps_indicator(y0_test[j], x0, tExp, altExp, Beta, N, M, q)
#         alt_min0 = phi(k_min[i], y0_test[j], x0, t[i + N])
#         alt_max0 = phi(k_max[i], y0_test[j], x0, t[i + N])
#         if alt_max0 - alt_min0 < wide:
#             wide = alt_max0 - alt_min0
#             y0[i] = y0_test[j]
#             alt_max[i] = alt_max0
#             alt_min[i] = alt_min0
#  #           print('y0 = {0}; wide = {1}'.format(y0[i], wide))
#  #           print('altmin = {0}; altmax = {1}'.format(alt_min[i], alt_max[i]))


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def run():

    # y0 = alt_noised[0]

    k_min = np.zeros(num_points - N)
    k_max = np.zeros(num_points - N)
    alt_min = np.zeros(num_points - N)
    alt_max = np.zeros(num_points - N)

    key = 0  # number of intervals greater than previous one

    for i in tqdm(range(num_points - N)):
        tExp = t[i:i + N]
        altExp = alt_noised[i:i + N]
        # y0 = np.average(altExp)
        y0 = altExp[0]
        x0 = tExp[0]
        #    print('y0 = {0}; x0 = {1}'.format(round(y0, 2), round(x0, 2)))
        [k_min[i], k_max[i]] = sps_indicator(y0, x0, tExp, altExp, Beta, N, M, q)
        #    print('\n k_min = {0}; k_max = {1} \n'.format(round(k_min[i], 2), round(k_max[i], 2)))
        alt_min[i] = phi(k_min[i], y0, x0, t[i + N])
        alt_max[i] = phi(k_max[i], y0, x0, t[i + N])
    return alt_max, alt_min, k_max, k_min