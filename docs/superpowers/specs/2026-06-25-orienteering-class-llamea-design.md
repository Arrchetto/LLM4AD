# LLaMEA Orienteering Optimizer Class Design

## Goal

Add a separate Orienteering Problem task whose LLaMEA candidates are complete
optimizer classes, while preserving the existing function-based
`OrienteeringEvaluation` and all existing search methods.

## User-visible behavior

The GUI exposes two independent OP tasks:

- `OrienteeringEvaluation`: existing function mode. Candidates implement
  `select_next_node(...)`.
- `OrienteeringClassEvaluation`: new class mode. Candidates implement
  `OrienteeringOptimizer` with `__call__(self, instance)`.

The GUI continues to expose one `LLaMEA` method. LLaMEA reads an explicit
evaluator contract and automatically selects function or class handling.

Existing function-mode tasks and methods retain their current behavior.
Methods that do not support class candidates fail before starting when paired
with `OrienteeringClassEvaluation`, with a clear compatibility error.

## Architecture

### Explicit candidate contract

Evaluation objects may declare:

```python
candidate_type = "function"
```

or:

```python
candidate_type = "class"
candidate_name = "OrienteeringOptimizer"
candidate_call_signature = ("instance",)
supported_methods = ("LLaMEA",)
```

Evaluators without `candidate_type` are treated as function evaluators for
backward compatibility. LLaMEA must not infer candidate type from the task or
class name.

### New task package

Create `llm4ad/task/optimization/orienteering_class/` containing:

- `template.py`: complete `OrienteeringOptimizer` example and class-specific
  task description.
- `evaluation.py`: `OrienteeringClassEvaluation`, dataset construction, route
  validation, and fitness calculation.
- `paras.yaml`: GUI defaults matching the existing OP task.
- `__init__.py`: exports the new evaluator and template.

The new evaluator reuses the existing deterministic OP `GetData` generator.
It does not modify the original OP package.

### Candidate class contract

Every valid candidate must define:

```python
class OrienteeringOptimizer:
    def __init__(self):
        ...

    def __call__(self, instance: dict) -> list[int]:
        ...
```

The evaluator creates a fresh optimizer instance for every dataset instance.
`instance` contains:

- `coordinates`
- `distance_matrix`
- `prizes`
- `start_node`
- `end_node`
- `max_length`

The optimizer returns the complete route only. It does not return or report
fitness.

### Route validation

The evaluator rejects a candidate result when any of these conditions holds:

- the optimizer cannot be instantiated without arguments;
- the candidate does not expose callable `__call__(instance)`;
- the returned value is not a one-dimensional sequence;
- route entries are not integer node identifiers;
- the route is empty;
- the route does not start and end at the declared depot;
- a node identifier is outside the dataset range;
- a non-depot customer appears more than once;
- total travel length exceeds `max_length`, allowing a small floating-point
  tolerance.

Unvisited customer nodes are allowed because OP selects a profitable subset.
A route containing only `[depot, depot]` is valid and receives zero prize.

Fitness is the mean collected prize across all configured instances. Higher is
better, matching existing LLM4AD maximization behavior.

## LLaMEA integration

The existing LLaMEA class remains the only GUI method. Its constructor selects
the evaluator-provided template and task description as it does today.

`generate_evaluator()` branches on `candidate_type`:

- Function mode keeps current parsing, signature validation, execution, and
  evaluation unchanged.
- Class mode executes the code, finds the explicitly named class, validates
  constructor and `__call__` signatures, and passes the class to
  `OrienteeringClassEvaluation.evaluate()`.

The LLM parser already accepts both `def` and `class` names. Class mode adds
an expected-name check so helper functions or helper classes cannot become the
candidate accidentally.

Invalid generated classes receive `-inf` fitness, an actionable feedback
message, and a profiler record with no score. They do not terminate the
evolutionary run.

## GUI profiler and logs

Each class sample is saved intact in LLaMEA's `code/` directory. The filename
uses `OrienteeringOptimizer`.

The function-oriented profiler representation cannot parse a class as a
`Function`. For class candidates, the adapter registers a synthetic function
record named `OrienteeringOptimizer` solely for GUI sample counting and
fitness plotting, while passing the complete class source through the
profiler's `program` field. The original source remains available in both
LLaMEA logs and LLM4AD sample records.

Convergence CSV and PNG generation remains unchanged because it consumes
profiler scores rather than candidate syntax.

## Compatibility guard

Before a run starts, the GUI/backend checks `supported_methods` when present.
Selecting `OrienteeringClassEvaluation` with a method other than LLaMEA raises
a clear error stating that the task requires class-candidate support.

No compatibility restriction is added to `OrienteeringEvaluation`.

## Testing

Add tests covering:

- dynamic export and deterministic datasets for
  `OrienteeringClassEvaluation`;
- a valid optimizer class receiving a positive finite score;
- zero-customer route behavior;
- invalid start/end depot, duplicate customer, invalid node, non-integer
  node, over-budget route, malformed return type, constructor arguments, and
  incorrect `__call__` signature;
- LLaMEA function mode remaining unchanged;
- LLaMEA class mode loading and evaluating the expected class;
- syntax errors and malformed class candidates producing `-inf` without
  escaping;
- profiler registration retaining full class source;
- GUI/backend compatibility rejection for unsupported methods;
- task parameter discovery for the new GUI task.

Run the focused tests first, then the full repository test suite.

## Non-goals

- Converting CVRP or any other task to class mode.
- Replacing the existing function-based OP task.
- Duplicating the LLaMEA evolutionary algorithm as `LLaMEAClass`.
- Allowing generated code to calculate or supply its own fitness.
- Adding persistence or cross-instance state to optimizer objects.
