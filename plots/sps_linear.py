import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def _phi(k, x0, y0, x):
    return y0 + k * (x - x0)


def _find_k(x0, y0, x, y):
    return (y - y0) / (x - x0)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


class SpsInterval:
    def __init__(self, upper_bound, lower_bound):
        self.upper = upper_bound
        self.lower = lower_bound


class SpsData:
    def __init__(self, angle_min, angle_max, conf_min, conf_max):
        self.angle_interval = SpsInterval(angle_max, angle_min)
        self.conf_interval = SpsInterval(conf_max, conf_min)


class SpsLinear:
    """
    Sps(num_points_sps=7, confidence_param=10,
                    num_random_sets=200, num_lines_sps=360)

        Linear implementation of SPS algorithm

        Attributes
        ----------
        _num_points_sps : int
            Number of data points to SPS analysis (N)
        _confidence : int
            Confidence parameter (q)
        _num_random_sets : int
            Quantity of random sets, ideally less than 2^N (M)
        _num_lines : int
            Quantity of lines that should be used in SPS step (k_q)
         _alt_sum : int
            Alternation sum of distances between SPS hypothesis and true data (H)
    """
    _x_data: np.ndarray
    _y_data: np.ndarray
    sps_data: SpsData

    def __init__(self, num_points_sps: int = 7, confidence_param: int = 10,
                 num_random_sets: int = 200, num_lines_sps: int = 360):

        self._num_points_sps = num_points_sps
        self._confidence = confidence_param
        self._num_random_sets = num_random_sets
        self._Beta = np.random.randint(2, size=(self._num_random_sets, self._num_points_sps)).astype(np.float64)
        self._Beta[self._Beta == 0] = -1.
        self._Beta[0, :] = 1.
        self._num_lines = num_lines_sps
        self._alt_sum = np.zeros((self._num_random_sets, self._num_lines))
        self._confidence_percent = str(int((1 - self._confidence / self._num_random_sets) * 100))

    # def broadcast_to_matrices(self, ):

    def _indicator(self, x0, y0, x_data, y_data):
        low_angle = np.round(-np.pi / 2, 2)
        high_angle = np.round(np.pi / 2, 2)
        low_k = np.tan(low_angle)
        high_k = np.tan(high_angle)
        left_border = 0
        right_border = 0
        loop_num = 1

        x0_matrix = np.broadcast_to(x0, self._num_points_sps - 1)
        y0_matrix = np.broadcast_to(y0, self._num_points_sps - 1)
        k_vec = _find_k(x0_matrix, y0_matrix, x_data[1:self._num_points_sps], y_data[1:self._num_points_sps])
        # start_k_interval = SpsInterval(np.max(k_vec), np.min(k_vec))
        low_k = np.min(k_vec)
        high_k = np.max(k_vec)

        x_data = np.broadcast_to(x_data[:, None], (x_data.shape[0], self._num_lines))
        y_data = np.broadcast_to(y_data[:, None], (y_data.shape[0], self._num_lines))

        while right_border - left_border < self._num_lines - 2 and loop_num < 10:
            # if right_border == 0:  # first loop condition
            #     ang = np.linspace(low_angle, high_angle, self._num_lines)
            #     k_test = np.tan(ang)
            # else:
            k_test = np.linspace(low_k, high_k, self._num_lines)

            self._angle_matrix = np.broadcast_to(k_test, (self._num_points_sps, k_test.shape[0]))
            delta = _phi(self._angle_matrix, x0, y0, x_data) - y_data

            self._alt_sum = np.zeros((self._num_random_sets, self._num_lines))
            self._alt_sum = self._alt_sum + self._Beta @ delta

            h_abs = np.abs(self._alt_sum)
            rank = (h_abs <= h_abs[0, :]).sum(axis=0)

            if right_border == 0:
                max_rank = self._num_random_sets
            else:
                max_rank = self._num_random_sets - self._confidence

            left_border = max(0, np.argmax(rank < max_rank) - 2)
            right_border = min(rank.shape[0] - 1, rank.shape[0] - np.argmax(rank[::-1] < max_rank) + 1)
            low_k = k_test[left_border]
            high_k = k_test[right_border]
            loop_num += 1
        return [low_k, high_k]

    def process_data(self, x_data: np.ndarray, y_data: np.ndarray):
        self._x_data = x_data
        self._y_data = y_data
        num_points = self._y_data.shape[0]

        k_min = np.zeros(num_points - self._num_points_sps)
        k_max = np.zeros(num_points - self._num_points_sps)
        alt_min = np.zeros(num_points - self._num_points_sps)
        alt_max = np.zeros(num_points - self._num_points_sps)

        key = 0  # number of intervals greater than previous one

        for i in tqdm(range(num_points - self._num_points_sps)):
            x_data_slice = self._x_data[i:i + self._num_points_sps]
            y_data_slice = self._y_data[i:i + self._num_points_sps]
            # y0 = np.average(altExp)
            y0 = y_data_slice[0]
            x0 = x_data_slice[0]
            #    print('y0 = {0}; x0 = {1}'.format(round(y0, 2), round(x0, 2)))
            [k_min[i], k_max[i]] = self._indicator(x0, y0, x_data_slice, y_data_slice)
            #    print('\n k_min = {0}; k_max = {1} \n'.format(round(k_min[i], 2), round(k_max[i], 2)))
            alt_min[i] = _phi(k_min[i], x0, y0, self._x_data[i + self._num_points_sps])
            alt_max[i] = _phi(k_max[i], x0, y0, self._x_data[i + self._num_points_sps])
        print(alt_max)
        self.sps_data = SpsData(k_min, k_max, alt_min, alt_max)

    def plot(self, sps_data: SpsData = None, plots_num: int = 20, thresholds: List = None,
             sps_path: str = './sps_plots', test_name: str = 'Bearing Test', plot_dpi=200):
        if sps_data is None:
            sps_data = self.sps_data
        for ii in range(plots_num):
            pp = plots_num - ii

            fig = plt.figure()

            plt.ylabel('Roll, unit', fontsize=14)
            plt.xlabel('time, unit', fontsize=14)
            label1 = 'Model data'
            label2 = 'Industrial data'
            label_SPS1 = 'SPS prediction -' + self._confidence_percent + '%'
            label_SPS2 = 'SPS prediction, confidence = ' + self._confidence_percent + '%'

            plot_title = test_name

            plt.title(plot_title)
            plt.plot(self._x_data[self._num_points_sps:], sps_data.conf_interval.upper, 'mo:', markersize=2,
                     label=label_SPS1, linewidth=1)
            plt.plot(self._x_data[self._num_points_sps:], sps_data.conf_interval.lower, 'mo:', markersize=2)
            plt.plot(self._x_data, self._y_data, 'ro-', markersize=2, label=label2, linewidth=1)

            flag = True
            xlim_right_border = np.max(self._x_data)

            for thr in thresholds:
                y_trsh2 = thr

                # plt.hlines(threshold1, 0, self._x_data[-1], colors='green', linestyles='dotted', label='Threshold#1', lw=2)
                plt.axhline(y_trsh2, color='green', linestyle='dotted', label=f'Threshold {thr}',
                            lw=2)

                t_pr_low = (y_trsh2 - self._y_data[-(pp + self._num_points_sps)]) / sps_data.angle_interval.upper[-pp] + \
                           self._x_data[-(pp + self._num_points_sps)]
                t_pr_high = (y_trsh2 - self._y_data[-(pp + self._num_points_sps)]) / sps_data.angle_interval.lower[
                    -pp] + \
                            self._x_data[-(pp + self._num_points_sps)]

                # FIXME
                if t_pr_low > t_pr_high:
                    t_pr_high = 10 * t_pr_low
                    xlim_right_border = 2 * t_pr_low

                x_l = [self._x_data[-(pp + self._num_points_sps)], t_pr_low]
                y_l = [self._y_data[-(pp + self._num_points_sps)], y_trsh2]

                x_h = [self._x_data[-(pp + self._num_points_sps)], t_pr_high]
                y_h = [self._y_data[-(pp + self._num_points_sps)], y_trsh2]

                plt.plot(x_l, y_l, 'b--', linewidth=0.5)
                plt.plot(x_h, y_h, 'b--', linewidth=0.5)

                # plt.plot(t[-(pp + N):], linear_func(k_min[-pp],  alt_noised[-(pp + N)], t[-(pp + N)], t[-(pp + N):]), 'b--', linewidth=0.5)
                # plt.plot(t[-(pp + N):], linear_func(k_max[-pp],  alt_noised[-(pp + N)], t[-(pp + N)], t[-(pp + N):]), 'b--', linewidth=0.5)
                if flag:
                    plt.hlines(y_trsh2, t_pr_low, t_pr_high, colors='blue', linestyles='solid', label=label_SPS2, lw=3)
                    flag = False
                else:
                    plt.hlines(y_trsh2, t_pr_low, t_pr_high, colors='blue', linestyles='solid', lw=3)
            # plt.plot(t[N:-8], moving_average(alt_max - alt_min, 9), 'b:', markersize=2, label=label_SPS, linewidth=2)

            # plt.xlim(t0, 1.3 * t_end)
            # plt.ylim(-1, 3)

            plt.scatter(self._x_data[-(pp + self._num_points_sps):-pp], self._y_data[-(pp + self._num_points_sps):-pp],
                        s=20, color='blue', marker='s')

            plt.legend(loc=2, markerscale=1.1, fontsize=8, shadow=True)

            plt.xlim(np.min(self._x_data) - 10, xlim_right_border + 10)

            # plt.plot(self._x_data[self._num_points_sps:],
            #          sps_data.conf_interval.upper - sps_data.conf_interval.lower, color='b', markersize=2, linewidth=2)

            file_name = os.path.join(sps_path, 'plot_' + test_name + str(ii) + '_.png')
            plt.savefig(file_name, dpi=plot_dpi)
            plt.show()


if __name__ == '__main__':
    np.random.seed(239)
    random.seed(239)

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

    num_points = 50
    x = np.linspace(t0, t_end, num_points)
    alt = np.zeros(num_points)
    y = np.zeros(num_points)

    for i in range(num_points):
        if x[i] <= t1:
            alt[i] = _phi(k1, t0, a0, x[i])
        if t1 < x[i] <= t2:
            alt[i] = _phi(k2, t1, a1, x[i])
        y[i] = alt[i] + 0.02 * random.randint(-10, 10) * a1

    x = np.loadtxt('x_data_tfs_04.txt')
    y = np.loadtxt('y_data_tfs_04.txt')

    conf_step = 7
    q = 10
    M = 200
    sps = SpsLinear(
        num_points_sps=conf_step,
        confidence_param=q,
        num_random_sets=M,
        num_lines_sps=360
    )
    sps.process_data(x, y)
    sps.plot(
        plots_num=x.shape[0] - conf_step,
        thresholds=[50, 70]
    )
