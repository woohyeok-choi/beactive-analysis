import numpy as np
import networkx as nx
from typing import Callable, Iterable, List, Optional, Dict, Tuple, Set, NamedTuple
from math import log, floor
from itertools import product
import logging
import sys
import random
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib.collections import PathCollection
from copy import deepcopy

logging.basicConfig(stream=sys.stdout, format='[%(funcName)s] %(message)s')
logger = logging.getLogger("DStream")


def __is_neighbor__(grid_a: np.ndarray, grid_b: np.ndarray) -> bool:
    """
    One grid is a neighbor with another grid if these grids' coordinates are different by 1 in 1-dimension
    """
    assert grid_a.shape[0] == grid_b.shape[0]

    diff = grid_a - grid_b
    where = np.argwhere(diff)
    if where.shape[0] == 1:
        idx = where[0, 0]
        return np.abs(diff[idx]) == 1

    return False


def __check_neighbor__(grid_a: np.ndarray, grid_b: np.ndarray) -> Optional[Tuple[int, bool]]:
    assert grid_a.shape[0] == grid_b.shape[0]

    diff = grid_a - grid_b
    where = np.argwhere(diff)
    if where.shape[0] == 1:
        idx = where[0, 0]
        if diff[idx] == 1:
            return idx, True
        elif diff[idx] == -1:
            return idx, False
    return None


class GridCharacteristic:
    ATTR_SPARSE = 0
    ATTR_TRANSITIONAL = 1
    ATTR_DENSE = 2

    def __init__(self, dim: int):
        """
        :param dim: dimension of grid coordinates
        - last_updated_time: last time that a grid receives data
        - is_sporadic: whether a grid is sporadic grid
        - label: class label
        - attribute: density attribute (sparse, transitional, or dense)
        - density: density decayed over time
        - attraction: attraction btw. neighboring grids (NBs); # of NBs can be 2 * (dim - 1).
                      attraction[dim, 0] means NB different by -1 at dim,
                      and attraction[dim, 1] means NB different by 1 at dim.

        """
        self._dim: int = dim
        self.last_updated_time: int = -1
        self.label: Optional[int] = None
        self.density: float = 0.0
        self.attraction: np.ndarray = np.zeros((dim, 2))
        self.attribute = -1
        self.is_changed = False
        self.n_data = 0
        self.is_marked_sporadic = False
        self.last_sporadic_marked_time = -1
        self.last_removal_time = -1

    def to_string(self) -> str:
        return 'T_g = {}, label = {}, density = {}, attribute = {}, N_data = {}'.format(
            self.last_updated_time, self.label, self.density, self.attribute, self.n_data
        )

    def reset(self):
        self.last_updated_time = -1
        self.label = None
        self.density = 0.0
        self.attraction = np.zeros((self._dim, 2))
        self.attribute = -1
        self.is_changed = False
        self.n_data = 0
        self.is_marked_sporadic = False
        self.last_sporadic_marked_time = -1

    def receive_new_datum(self, cur_time: int, decay_factor: float, attraction_ini: Optional[np.ndarray] = None):
        if attraction_ini is not None:
            assert attraction_ini.shape == (self._dim, 2)

        self.density = self.density * (decay_factor ** (cur_time - self.last_updated_time)) + 1

        if attraction_ini is not None:
            self.attraction = self.attraction * (decay_factor ** (cur_time - self.last_updated_time)) + attraction_ini

        self.n_data += 1
        self.last_updated_time = cur_time

    def get_decayed_density(self, cur_time: int, decay_factor: float) -> float:
        return self.density * (decay_factor ** (cur_time - self.last_updated_time))

    def get_decayed_attraction(self, cur_time: int, decay_factor: float) -> np.ndarray:
        return self.attraction * (decay_factor ** (cur_time - self.last_updated_time))

    def prepare_clustering(self, cur_time: int, decay_factor: float, d_m: float, d_l: float):
        density = self.density * (decay_factor ** (cur_time - self.last_updated_time))

        if density >= d_m:
            attribute = GridCharacteristic.ATTR_DENSE
        elif density <= d_l:
            attribute = GridCharacteristic.ATTR_SPARSE
        else:
            attribute = GridCharacteristic.ATTR_TRANSITIONAL

        self.is_changed = self.attribute != attribute
        self.attribute = attribute


class StepStat(NamedTuple):
    cur_time: int
    n_grids: int
    n_removed: int
    datum: np.ndarray
    grid: Tuple
    label: int
    d_m: float
    d_l: float
    is_updated: bool


class DStream:
    def __init__(self,
                 dim: int,
                 decay_factor: float,
                 c_m: float,
                 c_l: float,
                 beta: float,
                 epsilon: Optional[float] = None,
                 min_cluster_size: int = 4,
                 is_debug: bool = False,
                 ):
        """
        :param dim: a dimension of data
        :param decay_factor: Decay factor from Chen and Tu, 2009
        :param c_m: Threshold for dense grid from Chen and Tu, 2009
        :param c_l: Threshold for sparse grid from Chen and Tu, 2009


        Each grid has a width of 1.0, and it is centered at 0.5. Therefore, it is required to choose right dist_func.
        In addition, data sequentially incomes by 1 timestamp.
        """

        self._dim = dim
        self._decay_factor = decay_factor
        self._c_m = c_m
        self._c_l = c_l
        self._beta = beta
        self._epsilon = epsilon

        self._grid_list: Dict[Tuple, GridCharacteristic] = dict()
        self._removed_list: Dict[Tuple, GridCharacteristic] = dict()
        self._is_initialized = False
        self._cur_time = 0
        self._min_cluster_size = min_cluster_size

        self._label_keys = []

        logger.setLevel(logging.DEBUG if is_debug else logging.FATAL)

        if self._epsilon is not None:
            assert self._epsilon <= 0.5

        assert self._c_m > 1
        assert 0 < self._c_l < 1

    def is_initialized(self):
        return self._is_initialized

    def add_datum(self, datum: np.ndarray) -> StepStat:
        assert datum.shape[0] == self._dim

        is_updated = False

        grid = self.get_grid(datum)
        cv = self._grid_list.get(grid)

        if cv is None:
            cv = self._removed_list.get(grid)
            if cv is None:
                cv = GridCharacteristic(self._dim)
            else:
                cv.reset()
            self._grid_list[grid] = cv

        cv.receive_new_datum(
            cur_time=self._cur_time,
            decay_factor=self._decay_factor,
            attraction_ini=self.__get_init_attraction__(datum)
        )

        gap = self.__calculate_time_gap__()

        if self._cur_time != 0 and gap != 0:
            if self._cur_time >= gap and not self._is_initialized:
                self._is_initialized = self.__init_clustering__(self._cur_time)
                is_updated = self._is_initialized

            if np.mod(self._cur_time, gap) == 0 and self._is_initialized:
                logger.debug('[{}] Adjust clustering: (gap = {}, grids = {})'.format(
                    self._cur_time, gap, len(self._grid_list.keys()))
                )

                self.__handle_sporadic_grids__(self._cur_time)
                self.__detect_sporadic_grids__(self._cur_time)

                self.__adjust_clustering__(self._cur_time)
                is_updated = True

        ret = StepStat(
            cur_time=self._cur_time,
            d_l=self._c_l / (len(self._grid_list.keys()) * (1 - self._decay_factor)),
            d_m=self._c_m / (len(self._grid_list.keys()) * (1 - self._decay_factor)),
            datum=datum,
            grid=grid,
            label=self._grid_list[grid].label if self._grid_list.get(grid) else None,
            n_grids=len(self._grid_list.keys()),
            n_removed=len(self._removed_list.keys()),
            is_updated=is_updated
        )

        self._cur_time += 1

        return ret

    def get_grid(self, datum: np.ndarray) -> Tuple:
        grid = np.floor(datum).astype(np.float64)

        return tuple(grid)

    def get_decay_factor(self) -> float:
        return self._decay_factor

    def get_grid_info(self, datum: np.ndarray) -> Tuple[Tuple, GridCharacteristic]:
        grid = self.get_grid(datum)

        return grid, self._grid_list.get(grid)

    def get_all_grids_info(self) -> Dict[Tuple, GridCharacteristic]:
        return deepcopy(self._grid_list)

    def get_cluster_label(self, datum: np.ndarray) -> Optional[int]:
        if not self._is_initialized:
            return None
        grid = self.get_grid(datum)
        cv = self._grid_list.get(grid)

        if cv is not None:
            return cv.label
        return None

    def get_all_cluster_labels(self) -> List[int]:
        return self._label_keys

    def __get_init_attraction__(self, datum: np.ndarray) -> Optional[np.ndarray]:
        if self._epsilon is None:
            return None

        g = self.get_grid(datum)
        g_c = np.array(g) + 0.5

        diff_x_c = datum - g_c
        diff_r_e = 0.5 - self._epsilon
        two_eps = 2 * self._epsilon

        omegas = np.ones((self._dim, self._dim)) - np.identity(self._dim) * 2

        attrs = np.where(np.abs(diff_x_c) < diff_r_e,
                         (1 + omegas) / 2,
                         1 / 2 + omegas * (0.5 / two_eps - np.abs(diff_x_c) / two_eps)
                         ).prod(axis=1)

        left = np.where(diff_x_c <= 0, attrs, 0)
        right = np.where(diff_x_c > 0, attrs, 0)

        return np.vstack([left, right]).transpose()

    def __generate_unique_key__(self) -> int:
        key = np.int(np.round(random.uniform(0, 1), 8) * 10 ** 8)
        while key in self._label_keys:
            key = np.int(np.round(random.uniform(0, 1), 8) * 10 ** 8)
        return key

    def __is_correlated__(self, cur_time: int, g: Tuple, h: Tuple, theta: float) -> bool:
        if self._epsilon is None:
            return True

        cv_g = self._grid_list.get(g)
        cv_h = self._grid_list.get(h)

        attr_idx = __check_neighbor__(np.array(g), np.array(h))

        if attr_idx is None:
            return False

        k, is_left = attr_idx

        attr_g_h = cv_g.get_decayed_attraction(cur_time, self._decay_factor)[k, 0 if is_left else 1]
        attr_h_g = cv_h.get_decayed_attraction(cur_time, self._decay_factor)[k, 0 if is_left else 1]

        print(
            'Theta = {} / g = {} / h = {} \nattr_g_h: \n{} \nattr_h_g: \n{} \n attr_g_h = {}, attr_h_g = {}'
            .format(theta, g, h, cv_g.get_decayed_attraction(cur_time, self._decay_factor),
                    cv_h.get_decayed_attraction(cur_time, self._decay_factor), attr_g_h, attr_h_g)
        )

        return attr_g_h > theta and attr_h_g > theta

    def __init_clustering__(self, cur_time: int) -> bool:
        neighbor_tree = self.__build_neighbor_tree__()
        p = neighbor_tree.number_of_edges() * 2
        theta = self._c_m / (p * (1 - self._decay_factor))

        n = len(self._grid_list.keys())
        d_m = self._c_m / (n * (1 - self._decay_factor))
        d_l = self._c_l / (n * (1 - self._decay_factor))

        for cv in self._grid_list.values():
            cv.prepare_clustering(cur_time, self._decay_factor, d_m, d_l)

        dense_grids = [
            grid for grid, cv in self._grid_list.items() if cv.attribute == GridCharacteristic.ATTR_DENSE
        ]

        if len(dense_grids) < self._min_cluster_size:
            return False

        for dense_grid in dense_grids:
            cv = self._grid_list.get(dense_grid)
            cv.label = self.__generate_unique_key__()
            self._label_keys.append(cv.label)

        # logger.debug('[{}] n_init_clusters = {}, n_pairs = {}'.format(cur_time, cluster_idx, n_pairs))

        while True:
            is_changed = False
            clusters = self.__build_cluster_dict__()

            for grids in clusters.values():
                _, outside_grids = self.__differentiate_grid_types__(grids)

                for g in outside_grids:
                    for h in neighbor_tree.neighbors(g):
                        cv_g = self._grid_list.get(g)
                        cv_h = self._grid_list.get(h)

                        label_g = cv_g.label
                        label_h = cv_h.label

                        is_correlated = self.__is_correlated__(self._cur_time, g, h, theta)
                        is_different_cluster = label_g != label_h and label_h is not None

                        if is_different_cluster and is_correlated:
                            grids_in_cluster_g = clusters.get(label_g)
                            grids_in_cluster_h = clusters.get(label_h)

                            if len(grids_in_cluster_g) > len(grids_in_cluster_h):
                                self.__adjust_label__(grids_in_cluster_h, label_g)
                            else:
                                self.__adjust_label__(grids_in_cluster_g, label_h)
                            is_changed = True
                        elif cv_h.attribute == GridCharacteristic.ATTR_TRANSITIONAL and is_correlated:
                            if label_g != label_h and label_g is not None:
                                cv_h.label = label_g
                                is_changed = True

            if not is_changed:
                break

        return True

    def __handle_sporadic_grids__(self, cur_time: int):
        sporadic_grids: Dict[Tuple, GridCharacteristic] = {
            grid: cv for grid, cv in self._grid_list.items() if cv.is_marked_sporadic
        }

        for grid, cv in sporadic_grids.items():
            if cv.last_sporadic_marked_time != -1 and cv.last_sporadic_marked_time + 1 <= cur_time:
                if cv.last_updated_time != cur_time:
                    self._grid_list.pop(grid)
                    cv.last_removal_time = cur_time
                    self._removed_list[grid] = cv
                else:
                    if not self.__is_sporadic__(grid, cur_time):
                        cv.is_marked_sporadic = False

        # logger.debug('[{}] remove sporadic grids = {}'.format(cur_time, [__to_array__(g) for g in sporadic_grids]))

    def __detect_sporadic_grids__(self, cur_time):
        for grid, cv in self._grid_list.items():
            if self.__is_sporadic__(grid, cur_time):
                cv.is_marked_sporadic = True
                cv.last_sporadic_marked_time = cur_time
            else:
                cv.is_marked_sporadic = False

    def __adjust_clustering__(self, cur_time: int):
        n = len(self._grid_list.keys())
        d_m = self._c_m / (n * (1 - self._decay_factor))
        d_l = self._c_l / (n * (1 - self._decay_factor))

        for cv in self._grid_list.values():
            cv.prepare_clustering(cur_time, self._decay_factor, d_m, d_l)

        # logger.debug('[{}] n_grid = {}, d_m = {}, d_l = {}'.format(cur_time, n, d_m, d_l))

        changed_grids: Dict[Tuple, GridCharacteristic] = {
            grid: cv for grid, cv in self._grid_list.items() if cv.is_changed
        }
        for grid, cv in changed_grids.items():
            if cv.attribute == GridCharacteristic.ATTR_SPARSE:
                self.__adjust_sparse_grid__(grid)
            elif cv.attribute == GridCharacteristic.ATTR_DENSE:
                self.__adjust_dense_grid__(grid)
            elif cv.attribute == GridCharacteristic.ATTR_TRANSITIONAL:
                self.__adjust_transitional_grid__(grid)

    def __adjust_sparse_grid__(self, g: Tuple):
        cv_g = self._grid_list.get(g)

        if cv_g.label is None:
            return

        clusters = self.__build_cluster_dict__()
        cluster_g = clusters.get(cv_g.label)
        tree = self.__build_neighbor_tree__(cluster_g)
        tree.remove_node(g)

        if tree.number_of_nodes() > 0 and not nx.is_connected(tree):
            for idx, sub_graph in enumerate(nx.connected_component_subgraphs(tree)):
                if idx == 0:
                    continue
                new_label = self.__generate_unique_key__()
                for node in sub_graph:
                    cv_h = self._grid_list.get(node)
                    cv_h.label = new_label
                self._label_keys.append(new_label)

        cv_g.label = None

    def __adjust_dense_grid__(self, g: Tuple):
        tree = self.__build_neighbor_tree__()
        p = tree.number_of_edges() * 2
        theta = self._c_m / (p * (1 - self._decay_factor))
        print(p)

        clusters = self.__build_cluster_dict__()
        cv_g = self._grid_list.get(g)

        max_size = -1
        h = None

        logging.debug(
            'Adjust a dense grid: {} / Neighbors = {}'.format(
                g, set(tree.neighbors(g))
            )
        )

        for _h in tree.neighbors(g):
            is_correlated = self.__is_correlated__(self._cur_time, g, _h, theta)
            cv_h = self._grid_list.get(_h)

            if cv_h.label is not None and is_correlated:
                c = clusters.get(cv_h.label)
                size = len(c)

                if max_size < size:
                    max_size = size
                    h = _h

        if h is not None:
            cv_h = self._grid_list.get(h)

            if cv_g.label != cv_h.label:
                if cv_h.attribute == GridCharacteristic.ATTR_DENSE:
                    if cv_g.label is None:
                        cv_g.label = cv_h.label
                    else:
                        cluster_g = clusters.get(cv_g.label)
                        cluster_h = clusters.get(cv_h.label)

                        if len(cluster_g) > len(cluster_h):
                            self.__adjust_label__(cluster_h, cv_g.label)
                        else:
                            self.__adjust_label__(cluster_g, cv_h.label)
                elif cv_h.attribute == GridCharacteristic.ATTR_TRANSITIONAL:
                    cluster_h = clusters.get(cv_h.label)
                    exp_cluster_h = cluster_h.union([g])
                    _, outside = self.__differentiate_grid_types__(exp_cluster_h)

                    if cv_g.label is None and g in outside:
                        cv_g.label = cv_h.label
                    elif cv_g.label is not None:
                        cluster_g = clusters.get(cv_g.label)
                        cluster_h = clusters.get(cv_h.label)

                        if len(cluster_g) >= len(cluster_h):
                            cv_h.label = cv_g.label
        elif cv_g.label is None:
            new_cluster = set()
            new_label = self.__generate_unique_key__()
            cv_g.label = new_label

            new_cluster.add(g)
            for h in tree.neighbors(g):
                if h not in new_cluster:
                    cv_h = self._grid_list.get(g)
                    is_correlated = self.__is_correlated__(self._cur_time, g, h, theta)

                    if cv_h.attribute == GridCharacteristic.ATTR_TRANSITIONAL and is_correlated:
                        new_cluster.add(h)
                        cv_h.label = new_label

    def __adjust_transitional_grid__(self, g: Tuple):
        tree = self.__build_neighbor_tree__()
        p = tree.number_of_edges() * 2
        theta = self._c_m / (p * (1 - self._decay_factor))

        clusters = self.__build_cluster_dict__()
        cv_g = self._grid_list.get(g)

        max_size = -1
        h = None

        for _h in tree.neighbors(g):
            cv_h = self._grid_list.get(_h)
            is_correlated = self.__is_correlated__(self._cur_time, g, _h, theta)

            if cv_h.label is not None and is_correlated:
                c = clusters.get(cv_h.label)
                size = len(c)
                exp_c = c.union([g])
                _, outside = self.__differentiate_grid_types__(exp_c)

                if g in outside and max_size < size:
                    max_size = size
                    h = _h

        if h is not None:
            cv_h = self._grid_list.get(h)
            cv_g.label = cv_h.label

    def __adjust_label__(self, grids: Iterable[Tuple], new_label: Optional[int]):
        for g in grids:
            self._grid_list[g].label = new_label

    def __build_neighbor_tree__(self, grid_list: Set[Tuple] = None) -> nx.Graph:
        graph = nx.empty_graph()
        grids = grid_list if grid_list else self._grid_list.keys()

        for g, h in product(grids, grids):

            if not graph.has_node(g):
                graph.add_node(g)

            if not graph.has_node(h):
                graph.add_node(h)

            if __is_neighbor__(np.array(g), np.array(h)):
                if not graph.has_edge(g, h):
                    graph.add_edge(g, h)

        return graph

    def __build_cluster_dict__(self) -> Dict[int, Set[Tuple]]:
        clusters: Dict[int, Set[Tuple]] = dict()

        for g, cv in self._grid_list.items():
            label = cv.label

            if label is not None:
                if clusters.get(label) is None:
                    clusters[label] = set()
                clusters[label].add(g)

        return clusters

    def __calculate_time_gap__(self) -> int:
        size = len(self._grid_list.keys())
        arg1 = self._c_l / self._c_m
        arg2 = (size - self._c_m) / (size - self._c_l)

        return floor(log(max(arg1, arg2), self._decay_factor))

    def __is_sporadic__(self, grid: Tuple, cur_time: int):
        cv = self._grid_list.get(grid)
        n = len(self._grid_list.keys())

        if cv is None:
            return False

        density = cv.get_decayed_density(cur_time, self._decay_factor)
        numerator = self._c_l * (1 - self._decay_factor ** (cur_time - cv.last_updated_time + 1))
        denominator = n * (1 - self._decay_factor)
        threshold = numerator / denominator

        return density < threshold and cur_time >= (1.0 + self._beta) * cv.last_removal_time

    def __differentiate_grid_types__(self, grids: Set[Tuple]) -> Tuple[Set[Tuple], Set[Tuple]]:
        """
        :param grids: list of grid keys (Tuple)
        :return: Tuple of inside grids and outside grids where both has a shape of (n_grids, dim)
        """
        inside_grids = set()
        outside_grids = set()
        neighbor_tree = self.__build_neighbor_tree__(grids)

        for grid in grids:
            neighbors = set(neighbor_tree.neighbors(grid))

            if len(neighbors) == self._dim * 2:
                inside_grids.add(grid)
            else:
                outside_grids.add(grid)

        return inside_grids, outside_grids


def visualize(dstream: DStream, data: np.ndarray, pause: Optional[float] = None):
    assert data.shape[1] == 2, "Only two dimensional data can be visualized."

    plt.ion()

    plot: Tuple[Figure, Axes] = plt.subplots()
    fig, axes = plot

    x = []
    y = []

    left_panel: Text = plt.gcf().text(0.01, 0.5, '')
    scatter_plots: PathCollection = axes.scatter(x, y)
    grids_plots: Dict[Tuple, (Rectangle, Text)] = dict()

    def __update_left_panel__(stat: StepStat):
        left_panel.set_text(
            'Is Updated: {}\nN_Grids: {}\nN_Removed: {}\nCur time: {}\nDatum: ({:.2f}, {:.2f})\n'
            'Grid: ({:.2f}, {:.2f})\nLabel: {}\nD_l: {:.2f}\nD_m: {:.2f}'.format(
                stat.is_updated, stat.n_grids, stat.n_removed, stat.cur_time, stat.datum[0], stat.datum[1],
                stat.grid[0], stat.grid[1], stat.label, stat.d_l, stat.d_m
            )
        )

    def __update_scatter_plot__(d: np.ndarray):
        x.append(d[0])
        y.append(d[1])
        scatter_plots.set_offsets(np.c_[x, y])

    def __update_grid__(time: int):
        all_grids = dstream.get_all_grids_info()
        for grid, cv in all_grids.items():

            if grids_plots.get(grid) is None:
                _x, _y = grid
                rect = Rectangle((_x, _y), 1.0, 1.0, linewidth=1.0)
                text = axes.annotate('', (_x + 0.5, _y + 0.5), fontsize=5, ha='center', va='center')
                grids_plots[grid] = (rect, text)
                axes.add_patch(rect)
            else:
                rect, text = grids_plots.get(grid)

            text.set_text('({:.2f}, {:.2f})\nDensity = {:.2f}\nLabel = {}'.format(
                grid[0], grid[1], cv.get_decayed_density(time, dstream.get_decay_factor()), cv.label
            )
            )

            rect.set_alpha(0.9)

            if cv.attribute == GridCharacteristic.ATTR_DENSE:
                rect.set_color('#004d40' if cv.label is None else '#b71c1c')
            elif cv.attribute == GridCharacteristic.ATTR_SPARSE:
                rect.set_color('#b2dfdb' if cv.label is None else '#ffcdd2')
            else:
                rect.set_color('#26a69a' if cv.label is None else '#e57373')

        plot_keys = set(grids_plots.keys())
        real_keys = set(all_grids.keys())
        diff = plot_keys.difference(real_keys)

        for d in diff:
            rect, text = grids_plots.pop(d)
            rect.remove()
            text.set_text('')

    x_min = np.min(data[:, 0])
    y_min = np.min(data[:, 1])
    x_max = np.max(data[:, 0])
    y_max = np.max(data[:, 1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.draw()
    plt.show()
    for i in range(data.shape[0]):
        datum = data[i]
        step_stat = dstream.add_datum(datum)

        __update_left_panel__(step_stat)
        __update_scatter_plot__(datum)
        __update_grid__(step_stat.cur_time)

        if pause is None:
            plt.waitforbuttonpress()
        else:
            plt.pause(pause)
