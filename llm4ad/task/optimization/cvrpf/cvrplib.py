from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_VEHICLE_PATTERN = re.compile(r"-k([1-9][0-9]*)$")
_NATURAL_PATTERN = re.compile(r"([0-9]+)")
_ROUTE_PATTERN = re.compile(r"^Route\s+#([1-9][0-9]*)\s*:\s*(.*)$", re.IGNORECASE)
_COST_PATTERN = re.compile(r"^Cost\s*:?[ \t]+(.+?)\s*$", re.IGNORECASE)
_SECTION_NAMES = {"NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"}


@dataclass(frozen=True)
class CVRPInstance:
    name: str
    dataset: str
    comment: str
    dimension: int
    capacity: int
    max_vehicles: int
    edge_weight_type: str
    coordinates: np.ndarray
    demands: np.ndarray
    distance_matrix: np.ndarray
    best_known_routes: tuple[tuple[int, ...], ...] = ()
    best_known_cost: float | None = None


@dataclass(frozen=True)
class CVRPSolution:
    name: str
    routes: tuple[tuple[int, ...], ...]
    cost: float


def natural_key(value: str) -> tuple[object, ...]:
    """Return a case-insensitive key with numeric components compared as integers."""
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_PATTERN.split(value)
    )


def build_euc_2d_distance_matrix(coordinates: Any) -> np.ndarray:
    """Build a TSPLIB EUC_2D integer distance matrix."""
    try:
        points = np.asarray(coordinates, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("coordinates must contain numeric values") from error
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
        raise ValueError("coordinates must be a non-empty n-by-2 array")
    if not np.isfinite(points).all():
        raise ValueError("coordinates must be finite")
    delta = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    with np.errstate(over="ignore", invalid="ignore"):
        lengths = np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float64))
    if not np.isfinite(lengths).all():
        raise ValueError("EUC_2D distances must be finite")
    return np.asarray(lengths + 0.5, dtype=np.int64)


def _parse_positive_int(value: str, field: str, path: Path) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{path}: {field} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{path}: {field} must be positive")
    return parsed


def _split_vrp(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    headers: dict[str, str] = {}
    sections = {name: [] for name in _SECTION_NAMES}
    active: str | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line == "EOF":
            continue
        if line in _SECTION_NAMES:
            active = line
            continue
        if active is not None:
            sections[active].append(line)
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: invalid header line {line!r}")
        key, value = (part.strip() for part in line.split(":", 1))
        if key in headers:
            raise ValueError(f"{path}:{line_number}: duplicate field {key}")
        headers[key] = value
    return headers, sections


def _parse_indexed_rows(
    rows: Sequence[str], expected_width: int, section: str, path: Path
) -> dict[int, tuple[float, ...]]:
    parsed: dict[int, tuple[float, ...]] = {}
    for row in rows:
        parts = row.split()
        if len(parts) != expected_width:
            raise ValueError(f"{path}: invalid row in {section}: {row!r}")
        try:
            node_id = int(parts[0])
            values = tuple(float(value) for value in parts[1:])
        except ValueError as error:
            raise ValueError(f"{path}: non-numeric row in {section}: {row!r}") from error
        if node_id in parsed:
            raise ValueError(f"{path}: duplicate node identifier {node_id} in {section}")
        parsed[node_id] = values
    return parsed


def parse_vrp(path: str | Path, *, dataset: str) -> CVRPInstance:
    """Parse one explicit CVRPLIB CVRP instance without skipping invalid data."""
    vrp_path = Path(path)
    if not vrp_path.is_file():
        raise FileNotFoundError(f"CVRPLIB instance does not exist: {vrp_path}")
    headers, sections = _split_vrp(vrp_path)
    required = ("NAME", "COMMENT", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE", "CAPACITY")
    missing = [field for field in required if not headers.get(field)]
    missing.extend(section for section in _SECTION_NAMES if not sections[section])
    if missing:
        raise ValueError(f"{vrp_path}: missing required fields/sections: {', '.join(sorted(missing))}")

    name = headers["NAME"]
    if vrp_path.stem != name:
        raise ValueError(f"{vrp_path}: NAME {name!r} does not match file stem {vrp_path.stem!r}")
    if headers["TYPE"].upper() != "CVRP":
        raise ValueError(f"{vrp_path}: TYPE must be CVRP")
    edge_weight_type = headers["EDGE_WEIGHT_TYPE"].upper()
    if edge_weight_type != "EUC_2D":
        raise ValueError(f"{vrp_path}: unsupported EDGE_WEIGHT_TYPE {edge_weight_type!r}")
    dimension = _parse_positive_int(headers["DIMENSION"], "DIMENSION", vrp_path)
    capacity = _parse_positive_int(headers["CAPACITY"], "CAPACITY", vrp_path)
    vehicle_match = _VEHICLE_PATTERN.search(name)
    if vehicle_match is None:
        raise ValueError(f"{vrp_path}: NAME must end with -kN to define max vehicles")
    max_vehicles = int(vehicle_match.group(1))

    coordinates_by_id = _parse_indexed_rows(
        sections["NODE_COORD_SECTION"], 3, "NODE_COORD_SECTION", vrp_path
    )
    demand_rows = _parse_indexed_rows(
        sections["DEMAND_SECTION"], 2, "DEMAND_SECTION", vrp_path
    )
    expected_ids = set(range(1, dimension + 1))
    if set(coordinates_by_id) != expected_ids:
        raise ValueError(f"{vrp_path}: NODE_COORD_SECTION identifiers must be 1..{dimension}")
    if set(demand_rows) != expected_ids:
        raise ValueError(f"{vrp_path}: DEMAND_SECTION identifiers must be 1..{dimension}")

    depot_tokens: list[int] = []
    terminated = False
    for row in sections["DEPOT_SECTION"]:
        try:
            depot_id = int(row)
        except ValueError as error:
            raise ValueError(f"{vrp_path}: invalid depot identifier {row!r}") from error
        if depot_id == -1:
            terminated = True
            continue
        if terminated:
            raise ValueError(f"{vrp_path}: data after DEPOT_SECTION terminator")
        depot_tokens.append(depot_id)
    if not terminated or len(depot_tokens) != 1 or depot_tokens[0] not in expected_ids:
        raise ValueError(f"{vrp_path}: depot must contain exactly one valid node followed by -1")
    depot_file_id = depot_tokens[0]
    if demand_rows[depot_file_id][0] != 0:
        raise ValueError(f"{vrp_path}: depot demand must be zero")

    file_order = [depot_file_id] + sorted(expected_ids - {depot_file_id})
    coordinates = np.asarray([coordinates_by_id[node] for node in file_order], dtype=np.float64)
    demand_values: list[int] = []
    for node in file_order:
        raw_demand = demand_rows[node][0]
        if not math.isfinite(raw_demand) or raw_demand < 0 or not raw_demand.is_integer():
            raise ValueError(f"{vrp_path}: demand for node {node} must be a non-negative integer")
        demand_values.append(int(raw_demand))
    demands = np.asarray(demand_values, dtype=np.int64)
    distance_matrix = build_euc_2d_distance_matrix(coordinates)
    coordinates.setflags(write=False)
    demands.setflags(write=False)
    distance_matrix.setflags(write=False)
    return CVRPInstance(
        name=name,
        dataset=str(dataset),
        comment=headers["COMMENT"],
        dimension=dimension,
        capacity=capacity,
        max_vehicles=max_vehicles,
        edge_weight_type=edge_weight_type,
        coordinates=coordinates,
        demands=demands,
        distance_matrix=distance_matrix,
    )


def parse_sol(path: str | Path, *, instance_name: str, dimension: int) -> CVRPSolution:
    """Parse one CVRPLIB solution using its depot-omitting customer labels."""
    solution_path = Path(path)
    if not solution_path.is_file():
        raise FileNotFoundError(f"CVRPLIB solution does not exist: {solution_path}")
    if solution_path.stem != instance_name:
        raise ValueError(
            f"{solution_path}: solution name {solution_path.stem!r} does not match "
            f"instance name {instance_name!r}"
        )
    routes: list[tuple[int, ...]] = []
    route_numbers: list[int] = []
    cost: float | None = None
    for line_number, raw_line in enumerate(solution_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        route_match = _ROUTE_PATTERN.match(line)
        if route_match:
            route_number = int(route_match.group(1))
            if route_number in route_numbers:
                raise ValueError(f"{solution_path}:{line_number}: duplicate route number")
            tokens = route_match.group(2).split()
            if not tokens:
                raise ValueError(f"{solution_path}:{line_number}: route must contain customers")
            try:
                customers = tuple(int(token) for token in tokens)
            except ValueError as error:
                raise ValueError(f"{solution_path}:{line_number}: invalid route node") from error
            if any(customer < 1 or customer >= dimension for customer in customers):
                raise ValueError(f"{solution_path}:{line_number}: route node outside 1..{dimension - 1}")
            route_numbers.append(route_number)
            routes.append((0, *customers, 0))
            continue
        cost_match = _COST_PATTERN.match(line)
        if cost_match:
            if cost is not None:
                raise ValueError(f"{solution_path}:{line_number}: duplicate Cost")
            try:
                cost = float(cost_match.group(1))
            except ValueError as error:
                raise ValueError(f"{solution_path}:{line_number}: invalid Cost") from error
            if not math.isfinite(cost):
                raise ValueError(f"{solution_path}:{line_number}: Cost must be finite")
            continue
        raise ValueError(f"{solution_path}:{line_number}: unrecognized solution line {line!r}")
    if not routes or cost is None:
        raise ValueError(f"{solution_path}: solution requires Route lines and Cost")
    if sorted(route_numbers) != list(range(1, len(routes) + 1)):
        raise ValueError(f"{solution_path}: route numbers must be consecutive from 1")
    ordered = tuple(route for _, route in sorted(zip(route_numbers, routes)))
    return CVRPSolution(name=instance_name, routes=ordered, cost=cost)


def load_instance_pair(
    vrp_path: str | Path, sol_path: str | Path, *, dataset: str
) -> CVRPInstance:
    instance = parse_vrp(vrp_path, dataset=dataset)
    solution = parse_sol(sol_path, instance_name=instance.name, dimension=instance.dimension)
    recomputed_cost = route_set_cost(instance, solution.routes)
    if not math.isclose(recomputed_cost, solution.cost, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{sol_path}: declared Cost {solution.cost} does not match "
            f"recomputed route cost {recomputed_cost}"
        )
    return replace(
        instance,
        best_known_routes=solution.routes,
        best_known_cost=solution.cost,
    )


def validate_routes(
    instance: CVRPInstance, routes: Any
) -> tuple[tuple[int, ...], ...]:
    """Validate complete CVRP routes without coercing or repairing candidate output."""
    if isinstance(routes, (str, bytes)) or not isinstance(routes, (list, tuple)):
        raise ValueError("routes must be a two-dimensional list or tuple of routes")
    if not routes:
        raise ValueError("routes must not be empty")

    normalized: list[tuple[int, ...]] = []
    seen_customers: set[int] = set()
    used_vehicle_count = 0
    for route_index, raw_route in enumerate(routes):
        if isinstance(raw_route, (str, bytes)) or not isinstance(raw_route, (list, tuple)):
            raise ValueError(f"route {route_index} must be a list or tuple")
        if len(raw_route) < 2:
            raise ValueError(f"route {route_index} must start and end at the depot")
        route: list[int] = []
        for position, raw_node in enumerate(raw_route):
            if isinstance(raw_node, (bool, np.bool_)) or not isinstance(
                raw_node, (int, np.integer)
            ):
                raise ValueError(
                    f"route {route_index} node at position {position} must be an integer"
                )
            node = int(raw_node)
            if node < 0 or node >= instance.dimension:
                raise ValueError(
                    f"route {route_index} node {node} is outside [0, {instance.dimension})"
                )
            route.append(node)
        if route[0] != 0 or route[-1] != 0:
            raise ValueError(f"route {route_index} must start and end at depot 0")
        if any(node == 0 for node in route[1:-1]):
            raise ValueError(f"route {route_index} contains an interior depot")
        customers = route[1:-1]
        duplicate_customers = seen_customers.intersection(customers)
        duplicate_customers.update(
            customer for customer in customers if customers.count(customer) > 1
        )
        if duplicate_customers:
            raise ValueError(f"duplicate customers: {sorted(duplicate_customers)}")
        seen_customers.update(customers)
        if customers:
            used_vehicle_count += 1
        load = sum(int(instance.demands[node]) for node in customers)
        if load > instance.capacity:
            raise ValueError(
                f"route {route_index} load {load} exceeds vehicle capacity {instance.capacity}"
            )
        normalized.append(tuple(route))

    if used_vehicle_count > instance.max_vehicles:
        raise ValueError(
            f"solution uses {used_vehicle_count} vehicles, maximum is {instance.max_vehicles}"
        )
    required = set(range(1, instance.dimension))
    missing = required - seen_customers
    if missing:
        raise ValueError(f"missing customers: {sorted(missing)}")
    return tuple(normalized)


def route_set_cost(instance: CVRPInstance, routes: Any) -> float:
    """Return finite total integer route distance after strict feasibility checks."""
    matrix = np.asarray(instance.distance_matrix)
    if matrix.ndim != 2 or matrix.shape != (instance.dimension, instance.dimension):
        raise ValueError("distance matrix shape does not match instance dimension")
    if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError("distance matrix and route cost must be finite numeric values")
    normalized = validate_routes(instance, routes)
    edge_costs = [
        float(matrix[start, end])
        for route in normalized
        for start, end in zip(route[:-1], route[1:])
    ]
    cost = math.fsum(edge_costs)
    if not math.isfinite(cost):
        raise ValueError("route cost must be finite")
    return cost


def load_cvrplib_sets(
    data_root: str | Path,
    set_names: Sequence[str],
    expected_counts: Mapping[str, int],
) -> tuple[CVRPInstance, ...]:
    """Load exactly the requested CVRPLIB sets with complete `.vrp`/`.sol` pairing."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"CVRPLIB data root does not exist: {root}")
    instances: list[CVRPInstance] = []
    seen_names: set[str] = set()
    for set_name in set_names:
        if set_name not in expected_counts:
            raise ValueError(f"missing expected count for CVRPLIB set {set_name}")
        set_dir = root / set_name
        if not set_dir.is_dir():
            raise FileNotFoundError(f"CVRPLIB set directory does not exist: {set_dir}")
        vrp_files = list(set_dir.rglob("*.vrp"))
        sol_files = list(set_dir.rglob("*.sol"))
        vrp_by_stem = _unique_by_stem(vrp_files, set_name, ".vrp")
        sol_by_stem = _unique_by_stem(sol_files, set_name, ".sol")
        if set(vrp_by_stem) != set(sol_by_stem):
            missing_solutions = sorted(set(vrp_by_stem) - set(sol_by_stem), key=natural_key)
            missing_instances = sorted(set(sol_by_stem) - set(vrp_by_stem), key=natural_key)
            raise ValueError(
                f"CVRPLIB set {set_name} has unmatched instance/solution pairs; "
                f"missing solutions={missing_solutions}, missing instances={missing_instances}"
            )
        expected_count = int(expected_counts[set_name])
        if len(vrp_by_stem) != expected_count:
            raise ValueError(
                f"CVRPLIB set {set_name} expected {expected_count} instances, "
                f"found {len(vrp_by_stem)}"
            )
        for name in sorted(vrp_by_stem, key=natural_key):
            if name in seen_names:
                raise ValueError(f"duplicate CVRPLIB instance name across sets: {name}")
            seen_names.add(name)
            instances.append(
                load_instance_pair(vrp_by_stem[name], sol_by_stem[name], dataset=set_name)
            )
    return tuple(sorted(instances, key=lambda instance: natural_key(instance.name)))


def _unique_by_stem(paths: Sequence[Path], set_name: str, suffix: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        if path.stem in result:
            raise ValueError(
                f"CVRPLIB set {set_name} contains duplicate {suffix} stem {path.stem!r}"
            )
        result[path.stem] = path
    return result


__all__ = [
    "CVRPInstance",
    "CVRPSolution",
    "build_euc_2d_distance_matrix",
    "load_cvrplib_sets",
    "load_instance_pair",
    "natural_key",
    "parse_sol",
    "parse_vrp",
    "route_set_cost",
    "validate_routes",
]
