# BP 1D LLaMEA Class Task Design

## Goal

Add `bp_1d_llamea_class`, a GUI-discoverable experiment task that lets LLaMEA generate a complete `BinPackingOptimizer` class. The task must use the exact frozen BPPLIB protocol already used by `bp_1d_eoh_full`, produce identical fitness for equivalent solutions, and reject every unsupported method or invalid candidate result.

## Scope and constraints

- Reuse `bp_1d_common/dataset.py`, `evaluation_core.py`, `split_manifest.json`, and `best_known.csv` without changing the manifest, references, benchmark files, split ranges, or fitness definition.
- Move reference parsing and coverage validation out of the EoH evaluator into a neutral `bp_1d_common/references.py` module. Both evaluators import that implementation; neither evaluator imports the other.
- Preserve all existing EoH behavior.
- Treat `timeout_seconds: 30` as the total limit for one candidate evaluation through the existing `Evaluation`/`SecureEvaluator` execution path, not a per-instance allowance.
- Do not run paid LLM searches or long GUI experiments.
- Do not modify or delete unrelated working-tree content, especially `data/` and `fssp_llamea_class/`.

## Architecture

### Shared references

`bp_1d_common/references.py` owns:

- the immutable `ReferenceValue` data object;
- CSV parsing and validation;
- accepted status validation;
- duplicate and empty `instance_id` rejection;
- positive numeric reference validation;
- coverage validation for a selected split;
- the stricter sourced `OPTIMAL`/`BEST_KNOWN` requirement for training.

The EoH evaluator keeps its public scoring behavior but replaces its local parser and coverage method with these shared functions. The LLaMEA evaluator uses the same functions and default CSV path.

### Candidate contract and template

The task metadata is:

```python
candidate_type = "class"
candidate_name = "BinPackingOptimizer"
candidate_call_signature = ("instance",)
supported_methods = ("LLaMEA",)
```

`template.py` contains exactly one top-level target class named `BinPackingOptimizer`. Its constructor requires no arguments, and `__call__(self, instance)` implements deterministic Best Fit Decreasing while retaining `(original_index, size)` identity. It returns only `num_bins` and `bins`; bins contain original 1-based indices.

The description states that candidates are complete optimizers and may contain construction, improvement, and local-search member methods. It explicitly warns against `items.index`, item sizes as output, dictionary-shaped bins, and trivial renaming of FFD/BFD. It exposes no test instances.

### Evaluation flow

`BP1DLLaMEAClassEvaluation` loads instances with `load_split` and references through the shared loader. For each instance it:

1. builds a fresh dictionary whose scalar fields come from the immutable dataset object and whose `items` value is a new list;
2. constructs a new `optimizer_class()`;
3. calls `optimizer(instance_dict)`;
4. validates the result with `validate_packing`;
5. checks that the shared reference is usable;
6. computes the shared `relative_gap`;
7. aggregates the four training subseries using `macro_average_fitness`.

Any constructor, call, validation, reference, timeout, or non-finite-score failure invalidates the complete candidate and returns `None`. The evaluator never repairs output and never skips failed instances. Each instance dictionary is built independently so mutation cannot affect the stored protocol or later instances.

### Discovery and method compatibility

The package contains `__init__.py`, `evaluation.py`, `template.py`, and `paras.yaml`. Dynamic task exports discover the evaluation class from `evaluation.py`; GUI task discovery sees the directory because it contains `paras.yaml`. `bp_1d_common` remains hidden because it has no `paras.yaml`. Existing method compatibility validation accepts LLaMEA and explicitly rejects EoH through `supported_methods`.

## Testing strategy

Development follows red-green-refactor. Tests first establish the reference extraction regression behavior, then the new task behavior.

Coverage includes:

- shared parser status, duplicate ID, value, source, and split-coverage validation;
- unchanged EoH score after extraction;
- exact metadata and template AST/signatures;
- feasibility of the baseline across all 30 real training instances when data are present;
- one optimizer object and one independent input dictionary/list per instance;
- rejection of dictionary bins, sizes instead of indices, duplicate/missing/out-of-range/non-integer indices, empty bins, capacity overflow, mismatched counts, constructor errors, and call errors;
- exact EoH/LLaMEA fitness parity to at least 12 decimal places using the same deterministic algorithm and split;
- unchanged 30-instance/four-subseries training protocol;
- GUI discovery and method compatibility;
- execution of the class template through the standard SecureEvaluator/LLaMEA adapter.

Temporary BPPLIB fixtures cover evaluator behavior independently of local benchmark availability. If the committed real dataset is incomplete, the real-data smoke test is reported as unavailable rather than downloading, fabricating, or altering data.

## Verification and delivery

Run the new test module, the requested LLaMEA regression modules, and compileall. Record the exact parity fitness, total tests and outcomes, any missing real-data paths, absence of long experiments, modified/untracked files, and final `git status`. No commit is created unless separately requested.
