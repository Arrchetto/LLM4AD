from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .cvrplib import CVRPInstance, validate_routes


def validate_customer_permutation(
    instance: CVRPInstance, permutation: Any
) -> tuple[int, ...]:
    """Validate a strict one-dimensional customer permutation without repair."""
    if not isinstance(permutation, list):
        raise ValueError("candidate output must be a one-dimensional Python list")
    if len(permutation) != instance.dimension - 1:
        raise ValueError("candidate permutation must contain every customer exactly once")

    normalized: list[int] = []
    for position, raw_node in enumerate(permutation):
        if isinstance(raw_node, (bool, np.bool_)) or not isinstance(
            raw_node, (int, np.integer)
        ):
            raise ValueError(f"customer at position {position} must be an integer")
        node = int(raw_node)
        if node <= 0 or node >= instance.dimension:
            raise ValueError(f"customer {node} is outside [1, {instance.dimension})")
        normalized.append(node)

    if len(set(normalized)) != len(normalized):
        raise ValueError("candidate permutation contains duplicate customers")
    if set(normalized) != set(range(1, instance.dimension)):
        raise ValueError("candidate permutation has missing customers")
    return tuple(normalized)


def _best_fit_assign(
    instance: CVRPInstance, assignment_order: Sequence[int]
) -> list[list[int]] | None:
    bins = [[] for _ in range(instance.max_vehicles)]
    loads = [0] * instance.max_vehicles
    for customer in assignment_order:
        demand = int(instance.demands[customer])
        feasible = [
            index
            for index, load in enumerate(loads)
            if load + demand <= instance.capacity
        ]
        if not feasible:
            return None
        vehicle = max(feasible, key=lambda index: (loads[index], -index))
        bins[vehicle].append(customer)
        loads[vehicle] += demand
    return bins


def decode_customer_permutation(
    instance: CVRPInstance, permutation: Any
) -> tuple[tuple[int, ...], ...]:
    """Decode a strict customer priority into fleet- and capacity-feasible routes."""
    priority = validate_customer_permutation(instance, permutation)
    bins = _best_fit_assign(instance, priority)
    if bins is None:
        rank = {customer: index for index, customer in enumerate(priority)}
        fallback_order = sorted(
            priority,
            key=lambda customer: (-int(instance.demands[customer]), rank[customer]),
        )
        bins = _best_fit_assign(instance, fallback_order)
        if bins is None:
            raise ValueError("customer permutation cannot be packed within max_vehicles")
        for route in bins:
            route.sort(key=rank.__getitem__)

    routes = tuple((0, *route, 0) for route in bins if route)
    return validate_routes(instance, routes)


__all__ = ["decode_customer_permutation", "validate_customer_permutation"]
