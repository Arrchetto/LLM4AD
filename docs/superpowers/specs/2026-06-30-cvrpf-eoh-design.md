# CVRPF Complete EoH Task Design

## Goal

Add a production `cvrpf` optimization task to LLM4AD in which EoH evolves one complete, deterministic CVRP solver function. The evaluator trains only on the 27 CVRPLIB A instances and 23 CVRPLIB B instances and maximizes negative mean route distance.

## Scope

The task supports EoH only. It does not add LLaMEA class candidates, a validation set, random instances, smoke gates, P/X evaluation, best-sample callbacks, gap computation, paid API calls, or long-running experiments. EoH population management, operators, registration, stopping, and best-sample selection remain unchanged.

## Package Structure

Create `llm4ad/task/optimization/cvrpf/` with:

- `cvrplib.py`: immutable instance/solution records, natural ordering, strict `.vrp`/`.sol` parsing, TSPLIB EUC_2D distance construction, paired dataset loading, and strict route scoring.
- `template.py`: the single top-level `solve` candidate function and the complete EoH prompt contract.
- `evaluation.py`: fixed A+B training-set loading, candidate execution, aggregate fitness, and evaluation artifacts.
- `__init__.py`: export `CVRPFEvaluation` and reusable parser/scorer APIs needed by a future separate P-set tool.
- `paras.yaml`: GUI-visible evaluator configuration.

Add focused tests in `tests/test_cvrpf.py`. Update the English and Chinese CVRPLIB source records with the correct A/B/P and X attribution.

## Candidate Contract

The template contains exactly one top-level function:

```python
def solve(
    distance_matrix: np.ndarray,
    demands: np.ndarray,
    vehicle_capacity: int,
    max_vehicles: int,
) -> list[list[int]]:
```

Node `0` is the depot and customers are `1..n-1`. Each returned inner list is one used vehicle route and must start and end at `0`. Every customer must occur exactly once across all routes, route load may not exceed `vehicle_capacity`, and the number of returned nonempty routes may not exceed `max_vehicles`. The function returns routes only, not fitness or metadata. It receives no BKS, files, network access, or external helper functions. All evolvable logic remains inside `solve`.

The initial template implementation is a deterministic complete baseline suitable for all 50 training instances. The prompt states the strict overall evaluation timeout, requires bounded computation, and explains that EoH maximizes `-mean_distance`.

## Dataset Discovery and Isolation

`CVRPFEvaluation(data_root=None, ...)` accepts an explicit extracted CVRPLIB root. When omitted, it derives `data/benchmarks/cvrp/extracted` from the checked-out workspace layout rather than embedding a machine-specific absolute path.

The evaluator calls a dedicated A+B loader with the set names fixed in code as `("A", "B")` and expected counts fixed as A=27 and B=23. The loader recursively locates paired `.vrp` and `.sol` files within only those two set directories, rejects missing/extra/unpaired/duplicate stems, and returns all 50 instances in natural instance-name order. Evaluation iterates only this stored tuple. P and X paths are never traversed by evaluator initialization, `evaluate_program`, or `evaluate`.

## CVRPLIB Parsing

The `.vrp` parser requires `NAME`, `COMMENT`, `TYPE`, `DIMENSION`, `EDGE_WEIGHT_TYPE`, `CAPACITY`, `NODE_COORD_SECTION`, `DEMAND_SECTION`, and `DEPOT_SECTION`. It accepts only `TYPE=CVRP` and `EDGE_WEIGHT_TYPE=EUC_2D`, rejects duplicate/missing/out-of-range node identifiers, requires one depot, and normalizes the depot to internal node `0`. File identifiers are converted from 1-based to 0-based in a deterministic mapping.

The vehicle limit is parsed from the instance-name suffix `-kN`. The `.sol` parser requires one or more `Route #n` lines and one finite `Cost` and validates the solution stem against the instance stem. CVRPLIB route lines omit the depot and label customers `1..dimension-1`, which already matches the internal customer indices after the `.vrp` depot (file node 1) is mapped to internal node 0. The parser therefore preserves those customer labels and adds internal depot `0` at both ends; subtracting one from route-line labels would incorrectly turn customer 1 into the depot. Parsed known routes are rescored using the same strict scorer; `A-n32-k5.sol` must reproduce cost 784.

For coordinates `(x_i, y_i)` and `(x_j, y_j)`, the distance matrix uses the TSPLIB nearest-integer rule:

```python
int(np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2) + 0.5)
```

It does not use floating Euclidean costs or Python `round()`.

## Strict Scoring

The scorer accepts only a Python list or tuple of routes (and may accept a two-dimensional NumPy integer array when it represents the same structure). Strings, bytes, scalars, ragged non-route objects, and arbitrary iterables are invalid. Every route must be a nonempty one-dimensional sequence of integer nodes; booleans and floating values are invalid even when numerically integral.

The scorer rejects missing depot endpoints, out-of-range nodes, depot appearances inside a route, duplicate or omitted customers, overloaded routes, and excess vehicles. It performs no coercion, deletion, completion, depot insertion, splitting, or fallback construction. Route distance is accumulated in a finite numeric type and must remain finite.

Any exception, timeout, invalid structure, infeasible route set, nonfinite instance cost, or nonfinite aggregate invalidates the complete candidate under `SecureEvaluator`; no partial mean is computed.

## Evaluation and Artifacts

For each stored A/B instance, the evaluator calls the candidate with copies of the distance matrix and demands plus scalar capacity and vehicle limit. It records:

```text
instance_name, candidate_cost
```

The aggregate artifact contains `fitness`, `mean_distance`, and ordered `instance_costs`. The standard EoH `evaluate_program` and `evaluate` interfaces return the finite scalar fitness required by existing population and profiler code:

```text
mean_distance = mean(candidate_cost_i for all 50 A+B instances)
fitness = -mean_distance
```

BKS and gap do not participate in training and are not passed to candidates. A public artifact-producing evaluation helper preserves per-instance results for direct inspection and future tooling without changing EoH's score contract.

`CVRPFEvaluation.supported_methods = ("EoH",)` lets the existing GUI compatibility guard reject other methods. No class-candidate metadata is added.

## Testing and Verification

Development follows red-green-refactor. Tests cover exact A/B counts and isolation, deterministic natural ordering, missing/mismatched file failures, required parser fields, `A-n32-k5` metadata and BKS reconstruction, a finite deterministic baseline, every strict feasibility rejection, no repair behavior, exact fitness direction/formula, one-function template conversion, standard safe execution, EoH-only GUI compatibility, absence of random generation, and absence of P-output or automatic P evaluation.

Verification runs the focused CVRPF tests, related optimization/GUI compatibility tests, Python `compileall`, and the complete unit suite if its runtime is reasonable. No real LLM or network operation is involved.
