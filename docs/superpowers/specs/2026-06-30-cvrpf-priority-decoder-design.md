# CVRPF Priority-Decoder EoH Design

## Objective

Revise the EoH-only `cvrpf` task so EoH evolves a deterministic customer-priority algorithm instead of constructing capacity-feasible routes directly. The evaluator converts each valid customer permutation into CVRP routes with a deterministic capacity decoder. This adopts the useful search-space reduction from the prototype without copying its permissive node repair, random data, floating-point distances, smoke gate, or unlimited-fleet formulation.

The task continues to train on exactly the 27 CVRPLIB A instances and 23 CVRPLIB B instances. It continues to maximize negative mean route distance and does not use BKS or gap during training.

## Candidate Contract

The template contains exactly one top-level function:

```python
def solve(
    distance_matrix: np.ndarray,
    demands: np.ndarray,
    vehicle_capacity: int,
    max_vehicles: int,
) -> list[int]:
    ...
```

The function returns a one-dimensional Python list containing every customer node `1..n-1` exactly once. Node `0` must not appear. Each element must be a Python or NumPy integer, and `bool` is invalid. Floating-point values, nested sequences, duplicates, omissions, and out-of-range nodes invalidate the entire candidate.

The permutation is a complete priority policy for the evaluator-owned decoder. The candidate does not return routes, fitness, BKS values, or metadata. All evolved logic remains inside `solve`, and the candidate remains compatible with `TextFunctionProgramConverter` and standard EoH function evolution.

## Deterministic Capacity Decoder

For each instance, the evaluator first validates the permutation without changing it. It then attempts order-driven best-fit assignment:

1. Create exactly `max_vehicles` empty bins with zero load.
2. Process customers in candidate order.
3. Assign each customer to the feasible bin having the greatest current load; break ties by vehicle index.
4. Preserve candidate-relative order among customers assigned to the same bin.

If this pass fails because no bin has enough residual capacity, the evaluator retries from empty bins using deterministic best-fit decreasing:

1. Sort customers by descending demand.
2. Break equal-demand ties by their position in the candidate permutation.
3. Apply the same best-fit assignment and deterministic vehicle-index tie-break.
4. Within each resulting bin, restore candidate-relative customer order.

The fallback changes capacity grouping but never deletes, inserts, duplicates, renumbers, or rounds a customer. It exists only to remove order-induced capacity fragmentation while preserving the candidate's route-order signal. If the fallback cannot pack all customers into `max_vehicles`, the candidate is invalid for that instance and therefore invalid overall.

Empty bins are omitted. Every emitted route is `[0, *customers, 0]`. The existing strict route scorer remains the final authority for exact coverage, depot placement, capacity, vehicle count, bounds, integer nodes, and finite cost.

## Evaluation Flow

For each naturally sorted A+B instance:

1. Call the candidate with copies of the distance matrix and demands plus integer capacity and vehicle count.
2. Strictly validate the returned customer permutation.
3. Decode it into routes with order-driven best-fit and the deterministic BFD fallback.
4. Run the existing strict route validation and integer-distance cost calculation.
5. Record the instance name and candidate cost.

Any candidate exception, timeout, invalid permutation, decoder failure, strict route validation failure, or non-finite value invalidates the entire candidate. No instance is skipped.

The aggregate remains:

```text
mean_distance = mean(candidate_cost_i for all 50 A+B instances)
fitness = -mean_distance
```

EoH maximizes fitness. BKS and gap remain excluded from training.

## Preserved Benchmark Semantics

- Load only CVRPLIB A and B; never load or evaluate P or X.
- Keep deterministic natural instance ordering.
- Keep TSPLIB `EUC_2D` integer distances: `int(sqrt(dx*dx + dy*dy) + 0.5)`.
- Keep each instance's `max_vehicles` parsed from `-kN`.
- Keep the configurable data root and explicit dataset integrity errors.
- Keep `supported_methods = ("EoH",)`.
- Do not add random instances, smoke gates, validation sets, early stopping, candidate reranking, or post-training P-set evaluation.
- Do not modify EoH population maintenance, selection, crossover, mutation, registration, termination, or best-sample selection.
- Do not add warm-start behavior or LLaMEA class mode.

## Template and Prompt

The template remains short and contains a deterministic nearest-neighbor permutation baseline. It describes only the precise permutation contract, timeout, fitness direction, and evaluator decoding responsibility. It does not prescribe Clarke-Wright, local search, genetic operators, or another specific high-performance algorithm.

The task description states that EoH should design a complete deterministic customer-priority algorithm whose permutation is decoded under capacity and fleet constraints. This is an explicit change from the previous direct-route contract and must not be described as an independently executable full CVRP solver.

## Testing

Tests will retain parser, dataset, distance, fitness, GUI compatibility, and no-P coverage. Direct-route contract tests will be replaced or supplemented with tests proving:

- the template has exactly one top-level function returning a customer permutation;
- a valid permutation decodes into strictly feasible routes;
- order-driven best-fit is deterministic;
- BFD fallback handles a permutation that fragments capacity in the first pass;
- the decoder never exceeds `max_vehicles`;
- candidate-relative order is preserved within decoded routes;
- empty, nested, duplicate, missing, depot-containing, out-of-range, floating-point, and boolean permutations are invalid;
- no customer insertion, deletion, or renumbering occurs;
- any single-instance failure invalidates the whole candidate;
- the 50-instance fitness remains finite for the template baseline and equals negative mean decoded route distance;
- no random data, smoke gate, P-set access, or EoH-core modification is introduced.

Verification will run the focused CVRPF tests, related optimization/GUI tests, Python `compileall`, and the complete unit suite when runtime remains reasonable. No real LLM or long EoH experiment will be run.

## Scope Change

This design deliberately changes the research claim. EoH evolves the complete customer-ordering component of a CVRP algorithm, while the evaluator supplies a fixed deterministic capacity decoder. It no longer claims that every aspect of route grouping and feasibility is generated by the candidate. This tradeoff is required to obtain the prototype's high-feasibility search behavior while retaining CVRPLIB data, integer distances, strict customer-permutation validation, and fleet-size limits.
