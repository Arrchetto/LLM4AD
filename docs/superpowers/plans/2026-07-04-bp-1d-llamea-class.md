# BP 1D LLaMEA Class Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LLaMEA-only complete optimizer class task for frozen 1D Bin Packing experiments with exact EoH fitness parity.

**Architecture:** Extract best-known reference handling into `bp_1d_common/references.py`, keeping dataset and fitness primitives unchanged. Add an independent class-mode evaluator and deterministic BFD class template that share references and scoring with EoH, then verify discovery and standard LLaMEA adapter execution.

**Tech Stack:** Python 3.11, `unittest`, LLM4AD `Evaluation`/SecureEvaluator, YAML task configuration.

---

## File structure

- Create `llm4ad/task/experiment/bp_1d_common/references.py`: reference model, CSV loader, and split coverage validation.
- Modify `llm4ad/task/experiment/bp_1d_common/__init__.py`: export common reference APIs.
- Modify `llm4ad/task/experiment/bp_1d_eoh_full/evaluation.py`: consume shared references without changing scoring.
- Create `llm4ad/task/experiment/bp_1d_llamea_class/{__init__.py,evaluation.py,template.py,paras.yaml}`: class task.
- Modify `tests/test_bp_1d_eoh_full.py`: extraction regression tests.
- Create `tests/test_bp_1d_llamea_class.py`: contract, validation, parity, discovery, and adapter tests.

### Task 1: Extract shared reference handling

**Files:**
- Create: `llm4ad/task/experiment/bp_1d_common/references.py`
- Modify: `llm4ad/task/experiment/bp_1d_common/__init__.py`
- Modify: `llm4ad/task/experiment/bp_1d_eoh_full/evaluation.py`
- Test: `tests/test_bp_1d_eoh_full.py`

- [ ] **Step 1: Write failing reference regression tests**

Add tests importing `ReferenceValue`, `load_references`, and `validate_reference_coverage` from `bp_1d_common.references`. Temporary CSVs assert valid parsing, duplicate rejection, invalid status rejection, missing coverage rejection, and training rejection for unsourced/non-final references. Assert the EoH evaluator stores the common `ReferenceValue` type.

- [ ] **Step 2: Verify RED**

Run `.venv/bin/python -m unittest tests.test_bp_1d_eoh_full -v`.

Expected: new tests fail because `bp_1d_common.references` does not exist.

- [ ] **Step 3: Implement the neutral reference API**

Implement an immutable `ReferenceValue(bins, status, source)`, `load_references(path)`, and `validate_reference_coverage(instances, references, split)`. Keep accepted statuses and training rules equivalent to the prior EoH implementation. Replace EoH-local definitions with imports and call the common validator.

- [ ] **Step 4: Verify GREEN and EoH regression**

Run the same module and expect all tests to pass with unchanged EoH baseline fitness.

### Task 2: Establish class task contract and template

**Files:**
- Create: `tests/test_bp_1d_llamea_class.py`
- Create: `llm4ad/task/experiment/bp_1d_llamea_class/__init__.py`
- Create: `llm4ad/task/experiment/bp_1d_llamea_class/template.py`
- Create: `llm4ad/task/experiment/bp_1d_llamea_class/paras.yaml`

- [ ] **Step 1: Write failing metadata/template tests**

Test exported evaluator metadata, parse `template_program` with `ast`, assert exactly one top-level `BinPackingOptimizer`, a no-required-argument constructor, and exact `__call__(self, instance)`. Execute the template and validate its result on a duplicate-size fixture. Assert the description documents the full optimizer and index contract.

- [ ] **Step 2: Verify RED**

Run `.venv/bin/python -m unittest tests.test_bp_1d_llamea_class -v`.

Expected: import failure because the package is absent.

- [ ] **Step 3: Implement minimal template/config/package**

Create deterministic Best Fit Decreasing with stable size-descending index order and best residual-space placement. Configure `BP1DLLaMEAClassEvaluation`, 30-second timeout, null data root, and train split. Expose template program, task description, and a directly importable baseline class.

- [ ] **Step 4: Verify template tests GREEN**

Run the new module and expect contract/template tests to pass while evaluator behavior remains RED until Task 3.

### Task 3: Implement isolated class evaluation

**Files:**
- Create: `llm4ad/task/experiment/bp_1d_llamea_class/evaluation.py`
- Modify: `tests/test_bp_1d_llamea_class.py`

- [ ] **Step 1: Write failing evaluator tests with temporary protocol fixtures**

Build temporary manifest, extracted BPPLIB files, and best-known CSV. Test successful scoring, fresh optimizer construction, independent instance/items objects, constructor/call exceptions, and every required malformed output. Include duplicate sizes so size-returning or `items.index`-style output is rejected.

- [ ] **Step 2: Verify RED**

Run the new module and confirm failures are due to missing evaluator behavior.

- [ ] **Step 3: Implement evaluator**

Define class metadata and initialize `Evaluation` with `use_numba_accelerate=False`. Load frozen instances/references and validate coverage. For each instance create a fresh optimizer and payload with `list(instance.items)`, validate through `validate_packing`, compute shared `relative_gap`, then shared `macro_average_fitness`. Catch all candidate failures and return `None`; explicitly reject non-finite final fitness. `evaluate_program` delegates to `evaluate`.

- [ ] **Step 4: Verify GREEN**

Run the new module and expect all temporary-fixture behavior tests to pass.

### Task 4: Prove parity, discovery, compatibility, and real-data baseline

**Files:**
- Modify: `tests/test_bp_1d_llamea_class.py`

- [ ] **Step 1: Write failing integration tests**

Add a deterministic EoH wrapper around the class baseline and compare both evaluators on the same split with `assertAlmostEqual(..., places=12)`. Assert 30 training entries and four subseries, dynamic task export, GUI discovery excluding common support, LLaMEA acceptance/EoH rejection, all-training feasibility, and `generate_evaluator` execution of the class template.

- [ ] **Step 2: Verify RED**

Run the new module and confirm any missing integration behavior fails for the expected reason.

- [ ] **Step 3: Make minimal integration corrections**

Adjust only package exports, metadata, or template/evaluator boundaries demonstrated by failures. Do not change GUI discovery unless its existing `paras.yaml` convention fails.

- [ ] **Step 4: Verify GREEN and record parity**

Run the new module, expect all tests to pass, and capture exact EoH/LLaMEA fitness.

### Task 5: Full requested verification

**Files:** No planned source changes.

- [ ] **Step 1: Run targeted tests**

Run `.venv/bin/python -m unittest tests.test_bp_1d_llamea_class -v` and `.venv/bin/python -m unittest tests.test_llamea_adapter tests.test_llamea_evaluation tests.test_gui_llamea_compat -v`.

- [ ] **Step 2: Run compile verification**

Run `.venv/bin/python -m compileall -q llm4ad tests GUI/run_gui.py`.

- [ ] **Step 3: Run focused EoH regression and inspect final state**

Run `.venv/bin/python -m unittest tests.test_bp_1d_eoh_full -v`, `git diff --check`, and `git status --short`.

Expected: tests pass, compileall is silent, diff check is clean, original `fssp_llamea_class/` remains unmodified/untracked, and no benchmark/reference files changed.

- [ ] **Step 4: Deliver evidence**

Report files, exact class contract, parity fitness, test counts/results, real-data status, skipped long/paid experiments, and final git status. Do not commit.
