# CVRPF Priority Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `cvrpf` from direct route generation to strict customer-permutation evolution with a deterministic, fleet-limited best-fit/BFD capacity decoder.

**Architecture:** Add a focused decoder module that strictly validates the candidate permutation, performs order-driven best-fit packing, falls back to deterministic best-fit decreasing when order fragmentation prevents packing, and sends decoded routes through the existing strict route validator. The evaluator calls this decoder before computing the unchanged negative-mean-distance fitness; the short EoH template evolves only the customer permutation.

**Tech Stack:** Python 3.11, NumPy, `unittest`, LLM4AD `Evaluation`/`SecureEvaluator`/`TextFunctionProgramConverter`, CVRPLIB A+B.

---

### Task 1: Strict Permutation Validation and Capacity Decoder

**Files:**
- Create: `llm4ad/task/optimization/cvrpf/decoder.py`
- Modify: `tests/test_cvrpf.py`

- [ ] **Step 1: Write failing permutation and decoder tests**

Add imports for `CVRPInstance`, `decode_customer_permutation`, and `validate_customer_permutation`. Add a tiny deterministic fixture and tests covering the public decoder contract:

```python
def make_decoder_instance() -> CVRPInstance:
    coordinates = np.zeros((5, 2), dtype=np.float64)
    demands = np.array([0, 4, 4, 6, 6], dtype=np.int64)
    distance_matrix = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0],
        ],
        dtype=np.int64,
    )
    return CVRPInstance(
        name="Tiny-n5-k2",
        dataset="test",
        comment="decoder fixture",
        dimension=5,
        capacity=10,
        max_vehicles=2,
        edge_weight_type="EUC_2D",
        coordinates=coordinates,
        demands=demands,
        distance_matrix=distance_matrix,
    )


class CustomerPermutationDecoderTests(unittest.TestCase):
    def setUp(self):
        self.instance = make_decoder_instance()

    def test_valid_permutation_is_preserved(self):
        self.assertEqual(
            validate_customer_permutation(self.instance, [1, 2, 3, 4]),
            (1, 2, 3, 4),
        )

    def test_invalid_permutations_are_rejected_without_customer_repair(self):
        invalid = (
            [], (1, 2, 3, 4), np.array([1, 2, 3, 4]),
            [0, 1, 2, 3], [1, 1, 3, 4], [1, 2, 3], [1, 2, 3, 5],
            [True, 2, 3, 4], [1.0, 2, 3, 4], [[1], 2, 3, 4],
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_customer_permutation(self.instance, value)

    def test_bfd_fallback_recovers_order_fragmentation_deterministically(self):
        first = decode_customer_permutation(self.instance, [1, 2, 3, 4])
        second = decode_customer_permutation(self.instance, [1, 2, 3, 4])
        self.assertEqual(first, ((0, 1, 3, 0), (0, 2, 4, 0)))
        self.assertEqual(second, first)
        self.assertEqual(validate_routes(self.instance, first), first)

    def test_decoder_never_exceeds_fleet_or_changes_customer_set(self):
        routes = decode_customer_permutation(self.instance, [4, 3, 2, 1])
        customers = [node for route in routes for node in route[1:-1]]
        self.assertLessEqual(len(routes), self.instance.max_vehicles)
        self.assertEqual(sorted(customers), [1, 2, 3, 4])
```

- [ ] **Step 2: Run the focused tests and verify the missing-module failure**

Run:

```bash
.venv/bin/python -m unittest tests.test_cvrpf.CustomerPermutationDecoderTests -v
```

Expected: FAIL because `llm4ad.task.optimization.cvrpf.decoder` does not exist.

- [ ] **Step 3: Implement strict validation and deterministic decoding**

Create `decoder.py` with these public functions and a private assignment helper:

```python
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .cvrplib import CVRPInstance, validate_routes


def validate_customer_permutation(
    instance: CVRPInstance, permutation: Any
) -> tuple[int, ...]:
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
            raise ValueError(
                f"customer {node} is outside [1, {instance.dimension})"
            )
        normalized.append(node)
    if len(set(normalized)) != len(normalized):
        raise ValueError("candidate permutation contains duplicate customers")
    required = set(range(1, instance.dimension))
    if set(normalized) != required:
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
```

Export only `decode_customer_permutation` and `validate_customer_permutation` through `__all__`.

- [ ] **Step 4: Run decoder tests and all strict scorer tests**

Run:

```bash
.venv/bin/python -m unittest \
  tests.test_cvrpf.CustomerPermutationDecoderTests \
  tests.test_cvrpf.StrictRouteScoringTests -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit decoder behavior**

```bash
git add llm4ad/task/optimization/cvrpf/decoder.py tests/test_cvrpf.py
git commit -m "feat: add deterministic CVRPF priority decoder"
```

### Task 2: Evaluator Integration and All-or-Nothing Fitness

**Files:**
- Modify: `llm4ad/task/optimization/cvrpf/evaluation.py`
- Modify: `tests/test_cvrpf.py`

- [ ] **Step 1: Replace route-candidate tests with permutation-candidate tests**

Update evaluator tests so `compile_template_solve()` returns a permutation and invalid candidates include malformed permutations. Add a direct integration assertion that evaluator costs equal decoded route costs:

```python
def test_evaluator_decodes_permutation_before_strict_costing(self):
    instance = self.evaluation.training_instances[0]
    permutation = self.solve(
        instance.distance_matrix.copy(),
        instance.demands.copy(),
        instance.capacity,
        instance.max_vehicles,
    )
    expected_routes = decode_customer_permutation(instance, permutation)
    expected_cost = route_set_cost(instance, expected_routes)
    artifact = self.evaluation.evaluate_with_artifact(self.solve)
    self.assertEqual(artifact["instance_costs"][0]["candidate_cost"], expected_cost)
```

Change the mutation test variable from `routes` to `permutation`. Keep exception, timeout, deterministic fitness, array-copy, 50-instance artifact, and all-or-nothing failure coverage.

- [ ] **Step 2: Run evaluator tests and verify direct-route integration fails**

Run:

```bash
.venv/bin/python -m unittest tests.test_cvrpf.CVRPFEvaluatorTests -v
```

Expected: FAIL because `evaluate_with_artifact()` passes the permutation directly to `route_set_cost()`.

- [ ] **Step 3: Decode candidate output in the evaluator**

Import the decoder and change only the per-instance evaluation boundary:

```python
from .decoder import decode_customer_permutation

candidate_output = candidate(
    instance.distance_matrix.copy(),
    instance.demands.copy(),
    int(instance.capacity),
    int(instance.max_vehicles),
)
routes = decode_customer_permutation(instance, candidate_output)
cost = route_set_cost(instance, routes)
```

Update the evaluator class docstring to describe customer-priority algorithms. Preserve the existing exception-to-`None` behavior, artifact schema, fitness formula, data loading, timeout, and supported method.

- [ ] **Step 4: Run evaluator tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_cvrpf.CVRPFEvaluatorTests -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit evaluator integration**

```bash
git add llm4ad/task/optimization/cvrpf/evaluation.py tests/test_cvrpf.py
git commit -m "feat: decode CVRPF customer priorities during evaluation"
```

### Task 3: Short EoH Template and Accurate Task Contract

**Files:**
- Modify: `llm4ad/task/optimization/cvrpf/template.py`
- Modify: `tests/test_cvrpf.py`

- [ ] **Step 1: Write failing template contract tests**

Replace direct-route wording assertions with permutation-decoder wording. Require a short one-function template, `list[int]` return annotation, exclusion of depot zero, strict exact coverage, decoder responsibility, runtime, and fitness direction. Remove assertions that force a particular grouping algorithm from the prompt:

```python
def test_template_documents_priority_decoder_contract(self):
    template_text = " ".join(template_program.lower().split())
    self.assertLessEqual(len(template_program.splitlines()), 34)
    for required in (
        "30-second", "customer permutation", "exactly once", "exclude depot 0",
        "capacity decoder", "max_vehicles", "negative mean", "eoh maximizes",
    ):
        self.assertIn(required, template_text)

def test_task_description_is_neutral_about_search_algorithm(self):
    description = " ".join(task_description.lower().split())
    self.assertIn("27 cvrplib a", description)
    self.assertIn("23 cvrplib b", description)
    self.assertIn("customer permutation", description)
    self.assertIn("deterministic capacity decoder", description)
    for forbidden in ("clarke-wright", "2-opt", "genetic algorithm"):
        self.assertNotIn(forbidden, description)
```

- [ ] **Step 2: Run template tests and verify old route contract fails**

Run:

```bash
.venv/bin/python -m unittest \
  tests.test_cvrpf.CVRPFEvaluatorTests.test_template_has_exactly_one_top_level_function \
  tests.test_cvrpf.CVRPFEvaluatorTests.test_template_documents_priority_decoder_contract \
  tests.test_cvrpf.CVRPFEvaluatorTests.test_task_description_is_neutral_about_search_algorithm -v
```

Expected: FAIL because the template returns route lists and describes candidate-owned grouping.

- [ ] **Step 3: Implement a short deterministic nearest-neighbor permutation baseline**

Use this structure, keeping one top-level function and no top-level helper:

```python
template_program = '''import numpy as np


def solve(
    distance_matrix: np.ndarray,
    demands: np.ndarray,
    vehicle_capacity: int,
    max_vehicles: int,
) -> list[int]:
    """Return a customer permutation within the 30-second evaluation.

    Include every customer exactly once and exclude depot 0. The evaluator's
    deterministic capacity decoder enforces vehicle_capacity and max_vehicles.
    EoH maximizes negative mean distance, so shorter decoded routes are better.
    """
    remaining = set(range(1, len(demands)))
    permutation = []
    current = 0
    while remaining:
        customer = min(
            remaining,
            key=lambda node: (float(distance_matrix[current, node]), node),
        )
        permutation.append(customer)
        remaining.remove(customer)
        current = customer
    return permutation
'''
```

Write a concise `task_description` under 1300 characters. It must state the exact function signature, strict one-dimensional customer-permutation contract, A+B counts, deterministic capacity/fleet decoding, all-or-nothing invalidation, 30-second timeout, no external state/BKS/fitness return, and maximization of negative mean decoded distance. Do not prescribe a named routing heuristic.

- [ ] **Step 4: Run template and secure-evaluation tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_cvrpf.CVRPFEvaluatorTests -v
```

Expected: all tests PASS, including execution through `SecureEvaluator`.

- [ ] **Step 5: Commit template contract**

```bash
git add llm4ad/task/optimization/cvrpf/template.py tests/test_cvrpf.py
git commit -m "refactor: evolve CVRPF customer permutations"
```

### Task 4: Package Exports and Regression Verification

**Files:**
- Modify: `llm4ad/task/optimization/cvrpf/__init__.py`
- Modify: `tests/test_cvrpf.py`

- [ ] **Step 1: Add failing public-export and forbidden-behavior assertions**

Assert the package exports both decoder functions and task sources contain no random generation, smoke gate, P-set evaluation, candidate class metadata, or permissive customer cleanup terms. Keep GUI compatibility checks for EoH-only support.

```python
def test_priority_decoder_is_publicly_reusable(self):
    from llm4ad.task.optimization import cvrpf

    self.assertIs(cvrpf.decode_customer_permutation, decode_customer_permutation)
    self.assertIs(cvrpf.validate_customer_permutation, validate_customer_permutation)
```

- [ ] **Step 2: Run integration tests and verify exports fail**

Run:

```bash
.venv/bin/python -m unittest tests.test_cvrpf.CVRPFIntegrationTests -v
```

Expected: FAIL because decoder functions are not yet package exports.

- [ ] **Step 3: Export decoder APIs**

Add imports and `__all__` entries in `cvrpf/__init__.py`:

```python
from .decoder import decode_customer_permutation, validate_customer_permutation
```

Do not change `paras.yaml`; it remains EoH function mode with a 30-second timeout.

- [ ] **Step 4: Run focused and related tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_cvrpf -v
.venv/bin/python -m unittest \
  tests.test_gui_llamea_compat \
  tests.test_gui_parameter_parsing -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run compile and full regression checks**

Run:

```bash
.venv/bin/python -m compileall -q llm4ad tests
.venv/bin/python -m unittest discover -s tests -v
git diff --check
git status --short --branch
```

Expected: compileall exits 0, the complete test suite passes, `git diff --check` has no output, and status contains only the intended task/test changes before commit.

- [ ] **Step 6: Commit exports and final test updates**

```bash
git add llm4ad/task/optimization/cvrpf/__init__.py tests/test_cvrpf.py
git commit -m "test: verify CVRPF priority-decoder integration"
```

- [ ] **Step 7: Inspect final scope**

```bash
git status --short --branch
git diff HEAD~4..HEAD --stat
git log -5 --oneline
```

Expected: clean worktree; changes are limited to the CVRPF decoder, evaluator, template, package exports, tests, and the committed design/plan documents. No EoH core file, random benchmark generator, P-set script, smoke gate, LLaMEA class mode, or long-running experiment is added.
