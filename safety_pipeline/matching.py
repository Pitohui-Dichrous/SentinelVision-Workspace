"""Deterministic maximum-cardinality, maximum-weight bipartite matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Mapping, Tuple, TypeVar


Left = TypeVar("Left", bound=Hashable)
Right = TypeVar("Right", bound=Hashable)


@dataclass
class _Edge:
    to: int
    reverse: int
    capacity: int
    cost: float


def _min_cost_matching(
    left_nodes: Iterable[Left],
    right_nodes: Iterable[Right],
    weights: Mapping[Tuple[Left, Right], float],
    *,
    require_maximum_cardinality: bool,
) -> Dict[Left, Right]:
    left = tuple(left_nodes)
    right = tuple(right_nodes)
    if not left or not right or not weights:
        return {}
    source = 0
    left_offset = 1
    right_offset = left_offset + len(left)
    sink = right_offset + len(right)
    graph: List[List[_Edge]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int, cost: float) -> _Edge:
        forward = _Edge(end, len(graph[end]), capacity, cost)
        backward = _Edge(start, len(graph[start]), 0, -cost)
        graph[start].append(forward)
        graph[end].append(backward)
        return forward

    left_index = {node: index for index, node in enumerate(left)}
    right_index = {node: index for index, node in enumerate(right)}
    for index in range(len(left)):
        add_edge(source, left_offset + index, 1, 0.0)
    for index in range(len(right)):
        add_edge(right_offset + index, sink, 1, 0.0)

    pair_edges: Dict[Tuple[Left, Right], _Edge] = {}
    for left_node in left:
        for right_node in right:
            pair = (left_node, right_node)
            if pair not in weights:
                continue
            edge = add_edge(
                left_offset + left_index[left_node],
                right_offset + right_index[right_node],
                1,
                -float(weights[pair]),
            )
            pair_edges[pair] = edge

    node_count = len(graph)
    while True:
        distance = [float("inf")] * node_count
        previous: List[Tuple[int, int] | None] = [None] * node_count
        distance[source] = 0.0
        # Bellman-Ford handles negative forward weights and residual reverse
        # edges.  Stable node/edge iteration supplies deterministic tie breaks.
        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distance[node] == float("inf"):
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate = distance[node] + edge.cost
                    if candidate < distance[edge.to] - 1e-12:
                        distance[edge.to] = candidate
                        previous[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        # In optional-unmatched mode, an augmenting path is useful only when
        # it strictly increases the matching's total weight.  A non-negative
        # path cost would preserve or reduce that total merely to add another
        # weak pair, which is exactly the identity-switch failure the tracker
        # must avoid.
        if not require_maximum_cardinality and distance[sink] >= -1e-12:
            break
        node = sink
        while node != source:
            parent, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent

    result: Dict[Left, Right] = {}
    for (left_node, right_node), edge in pair_edges.items():
        if edge.capacity == 0:
            result[left_node] = right_node
    return result


def maximum_cardinality_weight_matching(
    left_nodes: Iterable[Left],
    right_nodes: Iterable[Right],
    weights: Mapping[Tuple[Left, Right], float],
) -> Dict[Left, Right]:
    """Match as many pairs as possible, then maximize their total weight.

    This objective is appropriate for mutually-exclusive detection conflict
    resolution, where every geometrically valid pair removes one duplicate
    logical object.
    """

    return _min_cost_matching(
        left_nodes,
        right_nodes,
        weights,
        require_maximum_cardinality=True,
    )


def maximum_total_weight_matching(
    left_nodes: Iterable[Left],
    right_nodes: Iterable[Right],
    weights: Mapping[Tuple[Left, Right], float],
) -> Dict[Left, Right]:
    """Maximize total edge weight while allowing either side to stay unmatched.

    Unlike a cardinality-first objective, this will not trade one near-perfect
    tracking association for two barely admissible IoU matches.  That makes it
    suitable for identity-preserving tracking with an explicit IoU gate.
    """

    return _min_cost_matching(
        left_nodes,
        right_nodes,
        weights,
        require_maximum_cardinality=False,
    )
