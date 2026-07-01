# BP 1D Complete Optimizer Experiment Design

## Objective

Add a GUI-discoverable experiment task in which EoH evolves a complete
one-dimensional bin-packing solver rather than a scoring rule or another
partial heuristic. The task uses the local BPPLIB snapshot under
`/Users/a1-6/Desktop/LLM_Platform/data/benchmarks/bp_1d`.

The experiment trains only on instances with approximately 100 items. Held-out
families and larger sizes measure cross-family and cross-scale generalization.
A later LLaMEA task will generate a complete optimizer class and reuse the same
dataset, split, validation, and scoring protocol.

## Package Structure

Create the following packages beside `llm4ad/task/optimization`:

```text
llm4ad/task/experiment/
├── __init__.py
├── bp_1d_common/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluation_core.py
│   ├── split_manifest.json
│   └── best_known.csv
└── bp_1d_eoh_full/
    ├── __init__.py
    ├── evaluation.py
    ├── template.py
    └── paras.yaml
```

`bp_1d_common` is a neutral dependency. It contains no `evaluation.py` or
`paras.yaml`, so the GUI must not display it as a runnable task. The future
`bp_1d_llamea_class` package will import this common package instead of
importing the EoH task.

## Candidate Contract

The EoH template exposes exactly one top-level candidate function:

```python
def solve(
    instance_id: str,
    bin_capacity: int,
    num_items: int,
    items: list[int],
) -> dict:
    ...
```

It returns:

```python
{
    "num_bins": int,
    "bins": list[list[int]],
}
```

Item indices are one-based. Every item must appear exactly once, all indices
must be integers in range, `num_bins` must equal `len(bins)`, empty bins are
invalid, and no bin may exceed `bin_capacity`. The function constructs the
complete packing for one instance. Helper logic must be nested inside `solve`
because the current EoH converter evolves one function body.

The task declares `supported_methods = ("EoH",)`.

## Dataset Discovery and Parsing

The evaluator accepts an optional `data_root`. Its default is derived from the
repository/workspace layout and points to the local BPPLIB directory. Startup
must fail with a clear `FileNotFoundError` if the root, extracted directory,
manifest, or required instance is absent.

The parser reads the BPPLIB BPP format:

1. First non-empty line: number of items.
2. Second non-empty line: bin capacity.
3. Remaining lines: exactly that many integer item sizes.

It rejects non-positive sizes, sizes greater than capacity, malformed counts,
and non-integer values. Dataset paths in the manifest are relative to the
`extracted` directory so the protocol does not depend on absolute paths.

## Fixed Data Split

The split manifest records relative path, stable instance ID, family,
subseries, size, and split for every selected instance. Selection is
deterministic and committed; runtime code never resamples it.

### Training: 30 instances

Only approximately 100-item instances participate in evolutionary fitness:

| Subseries | Size | Count |
|---|---:|---:|
| Falkenauer U | 120 | 8 |
| Falkenauer T | 120 | 8 |
| Scholl 1 | 100 | 7 |
| Scholl 2 | 100 | 7 |

Within each stratum, sort normalized relative paths and select indices using a
fixed seed recorded in the manifest metadata. Once generated, the explicit
paths, rather than the seed, are authoritative.

### Same-distribution validation: 30 instances

Use disjoint instances from the same four strata, with counts 8, 8, 7, and 7.
They do not participate in EoH selection or parameter tuning fitness.

### Cross-family test

Use all 17 Waescher instances, all 28 Hard28 instances, and all 200 Schwerin
instances. These families are absent from training.

### Cross-scale test

Use a fixed, balanced sample from Falkenauer U sizes 250, 500, and 1000;
Falkenauer T sizes 249 and 501; and Scholl 1/2 sizes 200 and 500. Select four
instances per size/subseries stratum, producing 36 cross-scale instances. None
may overlap training or validation.

## Best-Known Metadata

Raw BPPLIB BPP files do not contain reference objective values. Maintain a
separate `best_known.csv` with this schema:

```text
instance_id,best_known_bins,status,source,checked_date
```

Allowed statuses are:

- `OPTIMAL`: the value is proved optimal by the cited source.
- `BEST_KNOWN`: a cited incumbent without proof of optimality.
- `LOWER_BOUND`: only a cited or reproducibly computed lower bound is known.
- `UNVERIFIED`: no acceptable source has been confirmed.

Every instance in the split manifest must have exactly one metadata row.
Training is fail-closed: its 30 rows must be `OPTIMAL` or `BEST_KNOWN` and have
a non-empty source. Validation and test reports separate results by reference
status; they never label a gap against `LOWER_BOUND` as an optimality gap.

Source priority is BPPLIB/University of Bologna, original instance-author
material, peer-reviewed exact-algorithm result tables, and then independently
cross-checked benchmark repositories. Values are matched by exact instance ID
and recorded without modifying raw files.

## Fitness and Reports

For a valid packing with `B` bins and an accepted reference `R`, compute:

```text
relative_gap = (B - R) / R
```

Training fitness is the negative macro-average gap:

1. Average gaps within each of the four training subseries.
2. Average the four subseries means with equal weight.
3. Negate the result because EoH maximizes fitness.

This prevents a subseries count from controlling the objective. Any exception,
timeout, malformed return, infeasible packing, missing training reference, or
non-finite score makes the candidate invalid and returns `None`.

Offline validation and test utilities report, overall and per family:

- mean and median reference gap;
- best-reference hit rate;
- mean bins used;
- invalid/timeout rate;
- mean runtime per instance;
- reference-status coverage.

## GUI Behavior

The existing GUI discovers first-level directories below `llm4ad/task`, so
`experiment` appears as a task category. Selecting it lists
`bp_1d_eoh_full`; `bp_1d_common` is excluded because it has no runnable task
configuration. GUI discovery will list only directories containing
`paras.yaml`, preventing support packages from appearing.

`paras.yaml` exposes only stable runtime parameters:

- evaluator class name;
- timeout in seconds;
- optional data root;
- split, defaulting to `train`.

The default GUI experiment uses only the training split. Validation and test
splits are explicit offline evaluation actions and must not influence EoH
search.

## LLaMEA Preparation

The later LLaMEA task will define:

```python
class BinPackingOptimizer:
    def __init__(self):
        ...

    def __call__(self, instance: dict) -> dict:
        ...
```

It will declare `candidate_type = "class"`, identify the class as
`BinPackingOptimizer`, and support only `LLaMEA`. Its evaluator will adapt the
class call to the same common evaluation core. No split, reference value,
feasibility rule, or aggregation formula changes between methods.

## Verification

Tests must cover:

- BPPLIB parsing and malformed-file rejection;
- deterministic manifest contents and zero split overlap;
- exact expected counts for every split and stratum;
- one-to-one manifest/BKS metadata coverage;
- valid packing normalization;
- duplicate, missing, out-of-range, non-integer, empty-bin, capacity, and
  `num_bins` mismatch rejection;
- macro-averaged fitness direction and weighting;
- method compatibility (`EoH` accepted, `LLaMEA` rejected);
- dynamic export through `llm4ad.task`;
- GUI discovery of `experiment/bp_1d_eoh_full` without exposing
  `bp_1d_common`;
- a baseline complete solver smoke test on all 30 training instances;
- parity hooks that the future LLaMEA evaluator can reuse.

Verification includes focused unit tests, the GUI compatibility tests, Python
compilation, and the relevant existing task/evaluator tests. No long-running
GUI experiment is launched automatically.

## Scope Boundaries

This work creates the EoH task and shared protocol. It does not run a paid LLM
search, implement the LLaMEA class task, alter raw benchmark archives, or claim
unverified reference values as optimal. The LLaMEA task is a later change built
against the frozen common contract.
