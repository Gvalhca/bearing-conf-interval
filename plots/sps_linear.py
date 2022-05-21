import os
import pathlib
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.optimize import minimize, Bounds
from scipy.spatial import ConvexHull
from tqdm import tqdm


def _phi(k, x0, y0, x):
    return y0 + k * (x - x0)


def _get_signed_dist(k, x0, y0, x, y):
    x0 = x0[:, np.newaxis]
    y0 = y0[:, np.newaxis]
    p1 = np.array((x0, y0))
    p2 = np.array((x, _phi(k, x0, y0, x)))
    p3 = np.array((x, y))
    x1 = (k ** 2 * x0 + x + k * (y - y0)) / (k ** 2 + 1)
    y1 = _phi(-1 / k, x, y, x1)
    p4 = np.array((x1, y1))
    dist = np.sqrt((y1 - y) ** 2 + (x1 - x) ** 2)
    line_k_test = p2 - p1
    line_normal = p4 - p3
    dot_prod = np.einsum('ijk,ijk->jk', line_normal, line_k_test)
    sign = np.sign(dot_prod)
    return dist * sign


def _get_k_interval(x0, y0, x, y):
    delta = x - x0
    # delta[delta == 0] = 1e-15
    return (y - y0) / delta


def _get_intersection_with_thresh(start_point, k, thresh):
    return (thresh - start_point[1] + k * start_point[0]) / k


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
    _x_data_slice: np.ndarray
    _y_data_slice: np.ndarray
    sps_data: SpsData

    def __init__(
            self,
            num_points_sps: int = 7,
            confidence_param: int = 10,
            num_random_sets: int = 200,
            num_k_samples: int = 360,
            num_b_samples: int = 360,
            thresholds: List = None
    ):
        self._num_points_sps = num_points_sps
        self._confidence = confidence_param
        self._num_random_sets = num_random_sets
        self._Beta = np.random.randint(2, size=(self._num_points_sps, self._num_random_sets)).astype(np.float64)
        self._Beta[self._Beta == 0] = -1.
        self._Beta[:, 0] = 1.
        self._num_k_samples = num_k_samples
        self._num_b_samples = num_b_samples
        # self._alt_sum = np.zeros((self._num_random_sets, self._num_lines))
        self._confidence_percent = str(int((1 - self._confidence / self._num_random_sets) * 100))
        self._start_x = []
        self._start_y = []
        self._thresholds = thresholds
        self._horizontal_intervals = []
        self._prediction_lines = []
        self._conf_maps = []
        self._sum_maps = []
        self.meshgrid_x = []
        self.meshgrid_y = []
        # def broadcast_to_matrices(self, ):

    def _indicator_to_minimize(self, start_point):
        low_k, high_k = self._indicator(start_point[0], start_point[1])
        x_thresh_high = _get_intersection_with_thresh(start_point, high_k, self._thresholds[0])
        x_thresh_low = _get_intersection_with_thresh(start_point, low_k, self._thresholds[0])
        if x_thresh_low < x_thresh_high:
            x_thresh_low = np.inf
        # return 0.5 * (x_thresh_low - x_thresh_high) ** 2
        return 0.5 * ((high_k - low_k) ** 2)
        # return 1 / 3 * ((-np.inf - low_k) ** 2 + (np.inf - high_k) ** 2 + (high_k - low_k)**2)

    def _minimize(self, x0, y0):
        vars_to_minimize = np.array([x0, y0])
        bounds = Bounds([0, 0], [self._x_data_slice[0, 0], self._thresholds[0]])
        indicator_minimized = minimize(self._indicator_to_minimize, vars_to_minimize, method='nelder-mead',
                                       bounds=bounds, options={'xatol': 1e-8, 'disp': True})
        print(f'Minimization result: {indicator_minimized.x}')
        return indicator_minimized.x[0], indicator_minimized.x[1]

    def _indicator(self, x0, y0):
        low_angle = np.round(-np.pi / 2, 2)
        high_angle = np.round(np.pi / 2, 2)
        low_k = np.tan(low_angle)
        high_k = np.tan(high_angle)
        left_border = 0
        right_border = 0
        loop_num = 1

        # x0_matrix = np.broadcast_to(x0, self._num_points_sps)
        # y0_matrix = np.broadcast_to(y0, self._num_points_sps)
        # k_interval = _get_k_interval(x0_matrix, y0_matrix, self._x_data_slice,
        #                              self._y_data_slice)
        # start_k_interval = SpsInterval(np.max(k_vec), np.min(k_vec))
        # low_k = np.min(k_interval)
        # high_k = np.max(k_interval)

        # while right_border - left_border < self._num_lines - 2 and loop_num < 10:
        # if right_border == 0:  # first loop condition
        #     ang = np.linspace(low_angle, high_angle, self._num_lines)
        #     k_test = np.tan(ang)
        # else:
        k_test = np.linspace(0, 2, self._num_k_samples)
        b_test = np.linspace(-x0, self._thresholds[0], self._num_b_samples)
        # k_test = np.linspace(0, 0.5, self._num_lines)
        # b_test = np.linspace(-10, 50, self._num_lines)

        vars_to_sps = np.array(np.meshgrid(k_test, b_test)).reshape(2, -1)
        # k_test[k_test == 0] = 1e-15

        # phi = np.atleast_2d(_phi(self._angle_matrix, x0, y0, self._x_data_slice)[:, 0])
        theta = np.atleast_2d(vars_to_sps)
        phi = np.squeeze(np.atleast_2d([self._x_data_slice, np.broadcast_to(1.0, shape=self._x_data_slice.shape)]))

        outer_prod = np.einsum('ij,jk->ikj', phi, phi.T)
        R = 1 / self._num_points_sps * outer_prod.sum(axis=-1)
        # R_sqrt_inv = np.atleast_2d(np.power(R, -1/2)).T
        R_sqrt = np.real(scipy.linalg.sqrtm(R))
        # np.savetxt('R_matrix.txt', R_sqrt)
        # R_sqrt[R_sqrt == 0] = 1e-5
        if np.linalg.det(R_sqrt) == 0:
            return [np.nan, np.nan, np.nan, np.nan], np.zeros((theta.shape[0], 0))
        R_sqrt_inv = np.linalg.inv(R_sqrt)

        predict = phi.T @ theta
        epsilon = self._y_data_slice.T - predict

        eps_shape = (
            self._num_points_sps, theta.shape[0], self._num_random_sets, self._num_k_samples * self._num_b_samples)
        epsilon = np.broadcast_to(epsilon[:, np.newaxis, np.newaxis, :], shape=eps_shape)
        epsilon = np.swapaxes(epsilon, 0, 1)
        check = self._Beta[np.newaxis, :, :, np.newaxis] * phi[:, :, np.newaxis, np.newaxis]
        residual = (1 / self._num_points_sps * (check * epsilon).sum(axis=1))
        residual = np.swapaxes(residual, 0, 1)
        sps_sums = R_sqrt_inv @ residual

        normalized_sps_sums = np.linalg.norm(sps_sums, axis=1) ** 2
        # permuted_sps_sums = np.random.permutation(normalized_sps_sums)
        # sps_sum = (self._Beta @ (np.broadcast_to(phi.T, (self._num_points_sps, self._num_points_sps)) @ delta_old)).sum(axis=-1)

        # self._angle_matrix = np.broadcast_to(k_test, (self._num_points_sps, k_test.shape[0]))
        # delta_old = _phi(self._angle_matrix, x0, y0, self._x_data_slice.T) - self._y_data_slice.T

        # delta = _get_signed_dist(self._angle_matrix, x0_matrix, y0_matrix, self._x_data_slice, self._y_data_slice)

        # self._alt_sum = np.zeros((self._num_random_sets, self._num_lines))
        # self._alt_sum = self._alt_sum + self._Beta @ delta_old

        h_abs = normalized_sps_sums
        zero_sum = h_abs[0, :]
        rank = (h_abs < zero_sum).sum(axis=-2)
        equal_sums = (h_abs == zero_sum).sum(axis=-2)
        normalized_bias = np.trunc(np.random.uniform(0, equal_sums + 1))
        rank = rank + normalized_bias
        # if right_border == 0:
        #     max_rank = self._num_random_sets
        # else:
        max_rank = self._num_random_sets - self._confidence

        left_border = max(0, np.argmax(rank < max_rank) - 2)
        right_border = min(rank.shape[0] - 1, rank.shape[0] - np.argmax(rank[::-1] < max_rank) + 1)

        sps_mask = rank <= max_rank

        confidence_map = theta[:, sps_mask]

        low_k = vars_to_sps[0, left_border]
        high_k = vars_to_sps[0, right_border]
        low_y0 = vars_to_sps[1, left_border]
        high_y0 = vars_to_sps[1, right_border]
        loop_num += 1

        # np.savetxt('norm_sps_sums.txt', normalized_sps_sums)
        meshgrid_x, meshgrid_y = np.meshgrid(k_test, b_test)
        return [low_k, high_k, low_y0, high_y0], confidence_map, rank, meshgrid_x, meshgrid_y

    def _broadcast_data(self):
        # self._x_data_slice[self._x_data_slice == 0] = 1e-15
        self._x_data_slice = np.atleast_2d(self._x_data_slice)
        self._y_data_slice = np.atleast_2d(self._y_data_slice)
        # self._x_data_slice = np.broadcast_to(self._x_data_slice[:, None],
        #                                      (self._x_data_slice.shape[0], self._num_lines))
        # self._y_data_slice = np.broadcast_to(self._y_data_slice[:, None],
        #                                      (self._y_data_slice.shape[0], self._num_lines))

    def process_data(self, x_data: np.ndarray, y_data: np.ndarray):
        # self._y_data = y_data[~np.isnan(y_data)]
        self._y_data = y_data
        self._x_data = x_data
        num_points = self._y_data.shape[0]
        # self._x_data = x_data[0:num_points]

        k_min = np.zeros(num_points - self._num_points_sps)
        k_max = np.zeros(num_points - self._num_points_sps)

        b_min = np.zeros(num_points - self._num_points_sps)
        b_max = np.zeros(num_points - self._num_points_sps)

        alt_min = np.zeros(num_points - self._num_points_sps)
        alt_max = np.zeros(num_points - self._num_points_sps)

        key = 0  # number of intervals greater than previous one

        for i in tqdm(range(num_points - self._num_points_sps)):
            self._x_data_slice = self._x_data[i:i + self._num_points_sps]
            self._y_data_slice = self._y_data[i:i + self._num_points_sps]

            # y0 = np.average(altExp)
            y0 = self._y_data_slice[0]
            x0 = self._x_data_slice[0]

            self._broadcast_data()

            # x0, y0 = self._minimize(x0, y0)
            # x0, y0 = 10, -10
            self._start_x.append(x0)
            self._start_y.append(y0)
            #    print('y0 = {0}; x0 = {1}'.format(round(y0, 2), round(x0, 2)))
            result, confidence_map, sum_map, meshgrid_X, meshgrid_Y = self._indicator(x0, y0)
            self._conf_maps.append(confidence_map)
            self._sum_maps.append(sum_map)
            self.meshgrid_x.append(meshgrid_X)
            self.meshgrid_y.append(meshgrid_Y)
            if np.isnan(np.sum(result)):
                continue
            k_min[i], k_max[i], b_min[i], b_max[i] = result
            #    print('\n k_min = {0}; k_max = {1} \n'.format(round(k_min[i], 2), round(k_max[i], 2)))
            if confidence_map.shape[-1] != 0:
                solving_matrix = np.atleast_2d(self._thresholds[0] - confidence_map[-1, :]).T
                k_matrix = np.atleast_2d(confidence_map[0, :]).T
                k_matrix[k_matrix == 0] = 1e-05
                intersections_x = solving_matrix / k_matrix
                self._horizontal_intervals.append(SpsInterval(np.max(intersections_x), np.min(intersections_x)))
                self._prediction_lines.append(SpsInterval(confidence_map[:, np.argmax(intersections_x)],
                                                          confidence_map[:, np.argmin(intersections_x)]))
            else:
                self._horizontal_intervals.append(SpsInterval(0, 0))
                self._prediction_lines.append(SpsInterval(np.zeros((2, 1)), np.zeros((2, 1))))

            # alt_min[i] = _phi(k_min[i], x0, b_max[i], self._x_data[i + self._num_points_sps])
            # alt_max[i] = _phi(k_max[i], x0, b_min[i], self._x_data[i + self._num_points_sps])
        print(alt_max)
        self.sps_data = SpsData(k_min, k_max, alt_min, alt_max)

    def plot_3d(self, path='./3d_plots'):
        plots_num = self._x_data.shape[0] - self._num_points_sps
        for ii in range(plots_num):
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            sum_map = self._sum_maps[ii].T.reshape(self._num_b_samples, -1)
            meshgrid_x = self.meshgrid_x[ii]
            meshgrid_y = self.meshgrid_y[ii]
            # Plot the surface.
            surf = ax.plot_surface(meshgrid_x, meshgrid_y, sum_map, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)

            # Customize the z axis.
            # ax.set_zlim(-1.01, 1.01)
            ax.zaxis.set_major_locator(LinearLocator(10))
            # A StrMethodFormatter is used automatically
            ax.zaxis.set_major_formatter('{x:.02f}')

            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)

            plt.show()

    def plot(self, sps_data: SpsData = None, plots_num: int = 20,
             sps_path: str = './sps_plots', test_name: str = 'Bearing Test', plot_dpi=200, need_show=True):
        if sps_data is None:
            sps_data = self.sps_data

        pathlib.Path(sps_path).mkdir(parents=True, exist_ok=True)

        plots_num = self._x_data.shape[0] - self._num_points_sps

        for ii in range(plots_num):
            pp = plots_num - ii
            fig, ax = plt.subplots(3, figsize=(10, 10))

            ax[0].set(xlabel='time, unit', ylabel='Roll, unit')
            ax[1].set(xlabel='k', ylabel='b')
            ax[2].set(xlabel='k', ylabel='b')

            label1 = 'Model data'
            label2 = 'Industrial data'
            label_SPS1 = 'SPS prediction -' + self._confidence_percent + '%'
            label_SPS2 = 'SPS prediction, confidence = ' + self._confidence_percent + '%'

            plot_title = test_name

            ax[0].set_title(plot_title)
            # plt.plot(self._x_data[self._num_points_sps:], sps_data.conf_interval.upper, 'mo:', markersize=2,
            #          label=label_SPS1, linewidth=1)
            # plt.plot(self._x_data[self._num_points_sps:], sps_data.conf_interval.lower, 'mo:', markersize=2)
            ax[0].plot(self._x_data, self._y_data, 'ro-', markersize=2, label=label2, linewidth=1)

            indicator_points = self._conf_maps[ii].T

            # if ii <= plots_num - self._num_points_sps:
            ax[1].plot(indicator_points[:, 0], indicator_points[:, 1], '.', color='k')
            ax[2].plot(indicator_points[:, 0], indicator_points[:, 1], '.', color='k')
            try:
                hull = ConvexHull(indicator_points)
                for simplex in hull.simplices:
                    ax[1].plot(indicator_points[simplex, 0], indicator_points[simplex, 1], 'c')
                    ax[2].plot(indicator_points[simplex, 0], indicator_points[simplex, 1], 'c')
                ax[1].plot(indicator_points[hull.vertices, 0], indicator_points[hull.vertices, 1], 'o', mec='r',
                              color='none', lw=1, markersize=10)
                ax[2].plot(indicator_points[hull.vertices, 0], indicator_points[hull.vertices, 1], 'o', mec='r',
                              color='none', lw=1, markersize=10)
            except Exception as e:
                print(e)

            flag = True
            xlim_right_border = np.max(self._x_data)

            for thr in self._thresholds:
                y_trsh2 = thr

                # plt.hlines(threshold1, 0, self._x_data[-1], colors='green', linestyles='dotted', label='Threshold#1', lw=2)
                ax[0].axhline(y_trsh2, color='green', linestyle='dotted', label=f'Threshold {thr}',
                                 lw=2)

                # t_pr_low = (y_trsh2 - self._start_y[ii]) / sps_data.angle_interval.upper[ii] + \
                #            self._start_x[ii]
                # t_pr_high = (y_trsh2 - self._start_y[ii]) / sps_data.angle_interval.lower[ii] + \
                #             self._start_x[ii]

                # FIXME
                # if t_pr_low > t_pr_high:
                #     t_pr_high = 10 * t_pr_low
                #     xlim_right_border = 2 * t_pr_low

                # x_l = [self._start_x[ii], t_pr_low]
                # y_l = [self._start_y[ii], y_trsh2]
                #
                # x_h = [self._start_x[ii], t_pr_high]
                # y_h = [self._start_y[ii], y_trsh2]
                #
                # plt.plot(x_l, y_l, 'b--', linewidth=0.5)
                # plt.plot(x_h, y_h, 'b--', linewidth=0.5)
                #

                y_pred = lambda x, k, b: k * x + b

                x_l = [0, 10_000]
                y_l = [*y_pred(np.array(x_l), *self._prediction_lines[ii].lower)]

                x_h = [0, 10_000]
                y_h = [*y_pred(np.array(x_l), *self._prediction_lines[ii].upper)]

                ax[0].plot(x_l, y_l, 'b--', linewidth=0.5)
                ax[0].plot(x_h, y_h, 'b--', linewidth=0.5)

                if ii < len(self._horizontal_intervals):
                    if flag:
                        ax[0].hlines(y_trsh2, self._horizontal_intervals[ii].lower,
                                        self._horizontal_intervals[ii].upper,
                                        colors='blue', linestyles='solid', label=label_SPS2, lw=3)
                        flag = False
                    else:
                        ax[0].hlines(y_trsh2, self._horizontal_intervals[ii].lower,
                                        self._horizontal_intervals[ii].upper,
                                        colors='blue', linestyles='solid', lw=3)

            ax[0].scatter(self._x_data[-(pp + self._num_points_sps):-pp],
                             self._y_data[-(pp + self._num_points_sps):-pp],
                             s=20, color='blue', marker='s')

            # plt.scatter(self._start_x[ii], self._start_y[ii],
            #             s=20, color='y', marker='o')

            ax[0].legend(loc=2, markerscale=1.1, fontsize=8, shadow=True)

            y_data_no_nan = self._y_data[~np.isnan(self._y_data)]

            ax[0].set_xlim(np.min(self._x_data) - 10, xlim_right_border + 10)
            ax[0].set_ylim(-(np.max(np.abs(y_data_no_nan)) + 1), np.max(np.abs(
                y_data_no_nan if np.max(y_data_no_nan) > np.max(self._thresholds) else np.max(self._thresholds))) + 1)
            # plt.plot(self._x_data[self._num_points_sps:],
            #          sps_data.conf_interval.upper - sps_data.conf_interval.lower, color='b', markersize=2, linewidth=2)

            ax[1].set_xlim(0, 1.5)
            ax[1].set_ylim(-self._x_data[np.argmax(y_data_no_nan)] - 1, self._thresholds[0] + 1)

            file_name = os.path.join(sps_path,
                                     'plot_' + test_name + str(ii) + '_N_' + str(
                                         self._num_points_sps) + '_conf_' + self._confidence_percent + '_.png')
            plt.tight_layout()
            plt.savefig(file_name, dpi=plot_dpi)
            if need_show:
                plt.show()


from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)

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
    M = 128
    q = 22
    for q in range(0, 64):
        sps = SpsLinear(
            num_points_sps=conf_step,
            confidence_param=q,
            num_random_sets=M,
            num_k_samples=1_000,
            num_b_samples=100,
            thresholds=[50, 70],
        )
        sps.process_data(x, y)
        sps.plot(
            sps_path=f'sps_plots/area_limits_N_{conf_step}',
            test_name='Bearing Test Area',
            plot_dpi=200,
            need_show=False,
        )
    # sps.plot_3d()
