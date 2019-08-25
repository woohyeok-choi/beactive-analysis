import numpy as np
from typing import Optional, Tuple, List, Dict, Set, NamedTuple, Union, Iterable, Callable
from sklearn.neighbors import BallTree, KDTree, DistanceMetric
from functools import wraps
from threading import RLock
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from time import perf_counter
import networkx as nx
from itertools import combinations, product, groupby
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.patches import Circle
from matplotlib.collections import PathCollection
from string import ascii_uppercase
import logging
import sys
from uuid import uuid4


logging.basicConfig(stream=sys.stdout, format='[%(funcName)s] %(message)s')
logger = logging.getLogger("DBStream")


def func_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        s = perf_counter()
        ret = func(*args, **kwargs)
        print('{}: {} s'.format(func.__name__, perf_counter() - s))
        return ret
    return wrapper


class MicroCluster(NamedTuple):
    center: Tuple
    label: str
    n_data: int
    last_update_time: int
    weight: float


class MacroCluster(NamedTuple):
    label: str
    n_micro_clusters: int
    sum_weights: float


class DBStream:
    def __init__(self,
                 dim: int,
                 radius: float,
                 decay_factor: float,
                 time_gap: int,
                 min_weight: float,
                 alpha: float,
                 sigma: Optional[float] = None,
                 tree_type: str = 'ball',
                 metric: str = 'euclidean',
                 is_debug: bool = True):

        assert tree_type in ['ball', 'kd']

        if tree_type == 'ball':
            assert metric in BallTree.valid_metrics
        elif tree_type == 'kd':
            assert metric in KDTree.valid_metrics

        self._dim = dim
        self._radius = radius
        self._decay_factor = decay_factor
        self._time_gap = time_gap
        self._min_weight = min_weight
        self._alpha = alpha
        self._sigma = sigma if sigma is not None else radius / 3.0

        self._tree_type = tree_type
        self._metric = metric
        self._dist_func: DistanceMetric = DistanceMetric.get_metric(metric)
        self._cur_time = 0
        self._micro_clusters: nx.Graph = nx.empty_graph()

        self._label_keys: List[str] = []

        logger.setLevel(logging.DEBUG if is_debug else logging.FATAL)

    def is_initialized(self) -> bool:
        return self._cur_time > self._time_gap

    def get_number_of_micro_clusters(self) -> int:
        return self._micro_clusters.number_of_nodes()

    def get_number_of_macro_clusters(self) -> int:
        counter = Counter([node['label'] for node in self._micro_clusters.nodes if node['label']])
        return len(counter.most_common())

    def get_micro_clusters(self) -> List[MicroCluster]:
        return [
            MicroCluster(center=node,
                         label=data['label'],
                         n_data=data['n_data'],
                         last_update_time=data['last_update_time'],
                         weight=self.__calculate_decayed_weight__(
                             self._cur_time, data['weight'], data['last_update_time'])
                         )
            for node, data in self._micro_clusters.nodes(data=True)
        ]

    def get_macro_clusters(self) -> List[MacroCluster]:
        return [
            MacroCluster(
                label=k,
                n_micro_clusters=len(list(g)),
                sum_weights=sum([
                    self.__get_decayed_weight__(self._cur_time, n) for n in g
                ])
            ) for k, g in groupby(self._micro_clusters.nodes, lambda x: self._micro_clusters.nodes[x]['label']) if k
        ]

    def get_micro_cluster(self, datum: np.ndarray) -> Optional[Tuple]:
        if self._micro_clusters.number_of_nodes() == 0:
            return None

        centers = np.array(list(self._micro_clusters.nodes))

        tree = self.__build_tree__(centers)

        distances, indices = tree.query([datum], k=1, return_distance=True)

        index = indices[0, 0]
        distance = distances[0, 0]

        if distance < self._radius:
            return tuple(centers[index])
        return None

    def get_macro_cluster(self, datum: np.ndarray) -> Optional[MacroCluster]:
        center = self.get_micro_cluster(datum)

        if center is not None:
            label = self._micro_clusters.nodes[center]['label']
            mcs = [
                mc
                for mc in self._micro_clusters.nodes
                if self._micro_clusters.nodes[mc]['label'] == label
            ]

            return MacroCluster(
                label=label,
                n_micro_clusters=len(mcs),
                sum_weights=sum([self.__get_decayed_weight__(self._cur_time, n) for n in mcs])
            )
        return None

    def get_label(self, datum: np.ndarray) -> Optional[str]:
        center = self.get_micro_cluster(datum)

        if center is not None:
            return self._micro_clusters.nodes[center]['label']

    def add_datum(self, datum: np.ndarray):
        self.__update__(datum)

        if self._cur_time >= self._time_gap and np.mod(self._cur_time, self._time_gap) == 0:
            self.__clean_up__()
            self.__re_cluster__()

        self._cur_time += 1

    def __generate_unique_key__(self) -> str:
        key = uuid4().hex[:6].upper()
        while key in self._label_keys:
            key = uuid4().hex[:6].upper()
        return key

    def __calculate_decayed_weight__(self, cur_time: int, weight: float, last_update_time: int) -> float:
        return weight * (2 ** (- self._decay_factor * (cur_time - last_update_time)))

    def __get_decayed_weight__(self, cur_time: int, center: Tuple) -> float:
        mc = self._micro_clusters.nodes[center]
        weight = mc['weight']
        last_update_time = mc['last_update_time']

        return self.__calculate_decayed_weight__(cur_time, weight, last_update_time)

    def __get_decayed_shared_density__(self, cur_time, g: Tuple, h: Tuple) -> float:
        s = self._micro_clusters.edges[g, h]
        shared_density = s['shared_density']
        last_update_time = s['last_update_time']

        return shared_density * (2 ** (- self._decay_factor * (cur_time - last_update_time)))

    def __add_new_micro_cluster__(self, cur_time: int, center: np.ndarray):
        self._micro_clusters.add_node(
            tuple(center),
            prev_center=None,
            n_data=1,
            last_update_time=cur_time,
            weight=1,
            label=None
        )

    def __add_new_shared_density(self, cur_time: int, g: Tuple, h: Tuple):
        self._micro_clusters.add_edge(g, h,
                                      last_update_time=cur_time,
                                      shared_density=1)

    def __update_micro_cluster__(self, cur_time: int, center: Tuple, datum: np.ndarray) -> np.ndarray:
        mc = self._micro_clusters.nodes[center]
        np_center = np.array(center)
        last_update_time = mc['last_update_time']
        weight = mc['weight']
        n_data = mc['n_data']

        new_center = np_center + np.exp(-((datum - np_center) ** 2) / (2 * (self._sigma ** 2))) * (datum - np_center)
        new_weight = weight * (2 ** (- self._decay_factor * (cur_time - last_update_time))) + 1
        n_data += 1

        mc['prev_center'] = center
        mc['n_data'] = n_data
        mc['last_update_time'] = cur_time
        mc['weight'] = new_weight

        nx.relabel_nodes(self._micro_clusters, {center: tuple(new_center)}, copy=False)
        return new_center

    def __update_shared_density__(self, cur_time, g: Tuple, h: Tuple):
        s = self._micro_clusters.edges[g, h]

        last_update_time = s['last_update_time']
        shared_density = s['shared_density']

        new_shared_density = shared_density * (2 ** (- self._decay_factor * (cur_time - last_update_time))) + 1

        s['last_update_time'] = cur_time
        s['shared_density'] = new_shared_density

    def __revert_micro_cluster__(self, center: Tuple):
        mc = self._micro_clusters.nodes[center]
        prev_center = mc['prev_center']

        if prev_center is not None:
            nx.relabel_nodes(self._micro_clusters, {center: prev_center}, copy=False)

    def __build_tree__(self, centers: np.ndarray) -> Optional[Union[KDTree, BallTree]]:
        if self._tree_type == 'ball':
            tree = BallTree(centers, metric=self._metric)
        else:
            tree = KDTree(centers, metric=self._metric)
        return tree

    def __update__(self, datum: np.ndarray):
        if self._micro_clusters.number_of_nodes() == 0:
            self.__add_new_micro_cluster__(self._cur_time, datum)
            return

        centers = np.array(list(self._micro_clusters.nodes))
        tree = self.__build_tree__(centers)

        indices = tree.query_radius([datum], self._radius)[0]

        if indices.shape[0] == 0:
            self.__add_new_micro_cluster__(self._cur_time, datum)
        else:
            for i in indices:
                c = tuple(centers[i])
                new_center = self.__update_micro_cluster__(self._cur_time, c, datum)
                centers[i] = new_center

            for i, j in combinations(indices, 2):
                c_i = tuple(centers[i])
                c_j = tuple(centers[j])

                if self._micro_clusters.has_edge(c_i, c_j):
                    self.__update_shared_density__(self._cur_time, c_i, c_j)
                else:
                    self.__add_new_shared_density(self._cur_time, c_i, c_j)

        all_nodes = np.array(list(self._micro_clusters.nodes))
        distances = self._dist_func.pairwise(all_nodes)

        reverted_pairs = np.argwhere((0 < distances) & (distances < self._radius))
        reverted_nodes = np.unique(reverted_pairs)

        for i in reverted_nodes:
            self.__revert_micro_cluster__(tuple(all_nodes[i]))

    def __clean_up__(self):
        self._is_updated = True

        w_weak = 2 ** (- self._decay_factor * self._time_gap)
        aw_weak = w_weak * self._alpha

        removed_nodes: List[Tuple] = []
        removed_edges: List[Tuple[Tuple, Tuple]] = []

        for node in self._micro_clusters.nodes:
            weight = self.__get_decayed_weight__(self._cur_time, node)
            if weight < w_weak:
                removed_nodes.append(node)

        for node in removed_nodes:
            self._micro_clusters.remove_node(node)

        for i, j in self._micro_clusters.edges:
            shared_density = self.__get_decayed_shared_density__(self._cur_time, i, j)
            if shared_density < aw_weak:
                removed_edges.append((i, j))

        for i, j in removed_edges:
            self._micro_clusters.remove_edge(i, j)

    def __re_cluster__(self):
        tree: nx.Graph = nx.empty_graph()
        for i in self._micro_clusters.nodes:
            mc_i_weight = self._micro_clusters.nodes[i]['weight']

            # Strong MC
            if mc_i_weight >= self._min_weight:
                tree.add_node(i)

        for i, j in self._micro_clusters.edges:
            mc_i_weight = self._micro_clusters.nodes[i]['weight']
            mc_j_weight = self._micro_clusters.nodes[j]['weight']

            if mc_i_weight >= self._min_weight and mc_j_weight >= self._min_weight:
                denominator = (mc_i_weight + mc_j_weight) / 2
                shared_density = self._micro_clusters.edges[i, j]['shared_density']
                c = shared_density / denominator

                if c >= self._alpha:
                    tree.add_edge(i, j)

        all_nodes = set(self._micro_clusters.nodes)

        connected_nodes = set([
            node for components in nx.connected_components(tree) for node in self.__label_nodes__(components)
        ])

        # Clear labels of non-connected nodes
        for n in all_nodes.difference(connected_nodes):
            self._micro_clusters.nodes[n]['label'] = None

    def __label_nodes__(self, nodes: Iterable) -> Iterable:
        labels = [self._micro_clusters.nodes[n]['label'] for n in nodes]

        counter = dict()
        for i in labels:
            if i is not None:
                if i in counter:
                    counter[i] += 1
                else:
                    counter[i] = 1

        if len(counter.keys()):
            label, _ = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0]
        else:
            label = self.__generate_unique_key__()

        for n in nodes:
            self._micro_clusters.nodes[n]['label'] = label

        return nodes

    @classmethod
    def visualize(cls, db_stream: 'DBStream', data: np.ndarray, pause: Optional[float] = None):
        assert data.shape[1] == 2, "Only two dimensional data can be visualized."
        plt.ion()

        plot: Tuple[Figure, Axes] = plt.subplots()
        fig, axes = plot

        x = []
        y = []

        left_panel: Text = plt.gcf().text(0.01, 0.5, '')
        scatter_plots: PathCollection = axes.scatter(x, y)
        cluster_patches: List[Tuple[Circle, Text]] = []

        def __update_left_panel__(params: Dict):
            left_panel.set_text(
                '\n'.join(['{}: {}'.format(k, v) for k, v in params.items()])
            )

        def __update_scatter_plot__(d: np.ndarray):
            x.append(d[0])
            y.append(d[1])
            scatter_plots.set_offsets(np.c_[x, y])

        def __update_cluster__():
            for c, t in cluster_patches:
                c.remove()
                t.remove()
            cluster_patches.clear()

            micro_clusters = db_stream.get_micro_clusters()

            for m in micro_clusters:
                _x, _y = m.center
                circle = Circle((_x, _y), radius=db_stream._radius, linewidth=1.0, alpha=0.9)
                circle.set_color('#004d40' if m.label is None else '#26a69a')
                annotation = ['({:.2f}, {:.2f})'.format(m.center[0], m.center[1]),
                              'W = {:.2f}'.format(db_stream.__get_decayed_weight__(db_stream._cur_time - 1, m.center)),
                              'N = {}'.format(m.n_data),
                              'Label = {}'.format(m.label)
                              ]
                text = axes.annotate('\n'.join(annotation), (_x, _y), fontsize=5, ha='center', va='center')
                axes.add_patch(circle)
                cluster_patches.append((circle, text))

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
            cur_time = db_stream._cur_time

            db_stream.add_datum(datum)

            label = db_stream.get_label(datum)

            mc = db_stream.get_micro_cluster(datum)
            n = db_stream.get_number_of_micro_clusters()

            __update_left_panel__({
                'N': n,
                'Time': cur_time,
                'Datum': '{:.2f}, {:.2f}'.format(datum[0], datum[1]),
                'MC': '{:.2f}, {:.2f}'.format(mc[0], mc[1]),
                'Label': label
            })
            __update_scatter_plot__(datum)
            __update_cluster__()

            if pause is None:
                plt.waitforbuttonpress()
            else:
                plt.pause(pause)
